# compiler/instructions.py

# opcodes
ADD  = 0
SUB  = 1
MUL  = 2
DIV  = 3
LOAD = 4   # load a constant value into a register
VAR  = 5   # load a variable into a register

OP_NAMES = {ADD: "ADD", SUB: "SUB", MUL: "MUL",
            DIV: "DIV", LOAD: "LOAD", VAR: "VAR"}

class Instruction:
    def __init__(self, op, dest, src1=0, src2=0):
        self.op   = op     # opcode int
        self.dest = dest   # destination register index
        self.src1 = src1   # source register 1 (or value for LOAD)
        self.src2 = src2   # source register 2

    def __repr__(self):
        name = OP_NAMES.get(self.op, "???")
        return f"{name:4} r{self.dest} r{self.src1} r{self.src2}"

    def to_list(self):
        """Serialize to flat list of 4 ints for GPU upload."""
        return [self.op, self.dest, self.src1, self.src2]