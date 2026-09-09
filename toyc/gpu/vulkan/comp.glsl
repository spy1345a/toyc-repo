#version 450

// Flat buffer layout: each instruction is 4 ints
//   [op, dest, src1, src2]
//
// Opcodes (must match toyc/gpu/instructions.py):
//   0 = LOAD  : regs[dest] = const_pool[src1]
//   1 = VAR   : regs[dest] = var_pool[src1]
//   2 = ADD   : regs[dest] = regs[src1] + regs[src2]
//   3 = SUB   : regs[dest] = regs[src1] - regs[src2]
//   4 = MUL   : regs[dest] = regs[src1] * regs[src2]
//   5 = DIV   : regs[dest] = regs[src1] / regs[src2]
//
// Buffer layout (all packed as int/float into the same SSBO):
//   [0]            : number of instructions (N)
//   [1]            : number of constants    (C)
//   [2]            : number of variables    (V)
//   [3 .. 3+N*4-1] : instructions           (N × 4 ints)
//   [3+N*4 .. +C-1]: constant pool          (C floats, bitcast)
//   [3+N*4+C ..]   : variable pool          (V floats, bitcast)
//   [last]         : output slot            (1 float, written by shader)

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) buffer DataBuf {
    int data[];
};

// Maximum registers — covers the largest AST the flattener can produce
// in a single dispatch (one register per Flattener.new_reg() call).
#define MAX_REGS 256

void main() {
    int n_instrs   = data[0];
    int n_consts   = data[1];
    int n_vars     = data[2];

    int instr_base = 3;
    int const_base = instr_base + n_instrs * 4;
    int var_base   = const_base + n_consts;
    int out_slot   = var_base   + n_vars;

    float regs[MAX_REGS];

    for (int i = 0; i < n_instrs; i++) {
        int op   = data[instr_base + i * 4 + 0];
        int dest = data[instr_base + i * 4 + 1];
        int src1 = data[instr_base + i * 4 + 2];
        int src2 = data[instr_base + i * 4 + 3];

        if (op == 0) {
            // LOAD: load constant
            regs[dest] = intBitsToFloat(data[const_base + src1]);
        } else if (op == 1) {
            // VAR: load variable
            regs[dest] = intBitsToFloat(data[var_base + src1]);
        } else if (op == 2) {
            // ADD
            regs[dest] = regs[src1] + regs[src2];
        } else if (op == 3) {
            // SUB
            regs[dest] = regs[src1] - regs[src2];
        } else if (op == 4) {
            // MUL
            regs[dest] = regs[src1] * regs[src2];
        } else if (op == 5) {
            // DIV
            regs[dest] = regs[src1] / regs[src2];
        }
    }

    // The last register written by the flattener holds the result.
    // n_instrs - 1 gives the index of the last instruction; its dest
    // is the result register.
    int result_reg = data[instr_base + (n_instrs - 1) * 4 + 1];
    data[out_slot] = floatBitsToInt(regs[result_reg]);
}
