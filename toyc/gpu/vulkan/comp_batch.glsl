#version 450

// Data-parallel batch evaluator: one shader invocation per instance.
//
// Same flat instruction set as comp.glsl, but the variable pool and the
// output hold one entry PER INSTANCE and every invocation evaluates the
// whole program for its own instance (gl_GlobalInvocationID.x).
//
// Flat buffer layout: each instruction is 4 ints
//   [op, dest, src1, src2]
//
// Opcodes (must match toyc/gpu/instructions.py):
//   0 = ADD   : regs[dest] = regs[src1] + regs[src2]
//   1 = SUB   : regs[dest] = regs[src1] - regs[src2]
//   2 = MUL   : regs[dest] = regs[src1] * regs[src2]
//   3 = DIV   : regs[dest] = regs[src1] / regs[src2] (0.0 divisor -> error)
//   4 = LOAD  : regs[dest] = const_pool[src1]
//   5 = VAR   : regs[dest] = var_pool[instance * n_vars + src1]
//
// Buffer layout (all packed as int/float into the same SSBO):
//   [0]            : number of instructions (N)
//   [1]            : number of constants    (C)
//   [2]            : number of variables    (V)
//   [3]            : number of instances    (M)
//   [4 .. 4+N*4-1] : instructions           (N × 4 ints)
//   [const_base..] : constant pool          (C floats, bitcast, shared)
//   [var_base ..]  : variable pool          (M × V floats, instance-major)
//   [out_base ..]  : outputs                (M floats, written by shader)
//   [out_base+M]   : error flag             (1 int: 0 ok, 1 division by zero
//                                           in ANY instance)

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) buffer DataBuf {
    int data[];
};

// Maximum registers — covers the largest AST the flattener can produce
// in a single dispatch (one register per Flattener.new_reg() call).
#define MAX_REGS 256

void main() {
    int n_instrs = data[0];
    int n_consts = data[1];
    int n_vars   = data[2];
    int n_inst   = data[3];

    int idx = int(gl_GlobalInvocationID.x);
    if (idx >= n_inst) {
        return;
    }

    int instr_base = 4;
    int const_base = instr_base + n_instrs * 4;
    int var_base   = const_base + n_consts;
    int out_base   = var_base   + n_inst * n_vars;
    int err_slot   = out_base   + n_inst;

    float regs[MAX_REGS];

    for (int i = 0; i < n_instrs; i++) {
        int op   = data[instr_base + i * 4 + 0];
        int dest = data[instr_base + i * 4 + 1];
        int src1 = data[instr_base + i * 4 + 2];
        int src2 = data[instr_base + i * 4 + 3];

        if (op == 0) {
            // ADD
            regs[dest] = regs[src1] + regs[src2];
        } else if (op == 1) {
            // SUB
            regs[dest] = regs[src1] - regs[src2];
        } else if (op == 2) {
            // MUL
            regs[dest] = regs[src1] * regs[src2];
        } else if (op == 3) {
            // DIV — mirror the CPU (Cpu._execute): a zero divisor
            // records the shared error flag instead of producing inf.
            // Every failing instance writes the same value, so no
            // atomic is needed.
            if (regs[src2] == 0.0) {
                data[err_slot] = 1;
                regs[dest] = 0.0;
            } else {
                regs[dest] = regs[src1] / regs[src2];
            }
        } else if (op == 4) {
            // LOAD: load constant (shared by all instances)
            regs[dest] = intBitsToFloat(data[const_base + src1]);
        } else if (op == 5) {
            // VAR: load this instance's variable
            regs[dest] = intBitsToFloat(data[var_base + idx * n_vars + src1]);
        }
    }

    // The last register written by the flattener holds the result.
    int result_reg = data[instr_base + (n_instrs - 1) * 4 + 1];
    data[out_base + idx] = floatBitsToInt(regs[result_reg]);
}
