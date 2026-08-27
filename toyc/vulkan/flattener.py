# compiler/flattener.py

from ..ast_nodes    import Number, Var, BinOp
from .instructions import Instruction, ADD, SUB, MUL, DIV, LOAD, VAR

class Flattener:
    def __init__(self):
        self.instructions = []
        self.reg_counter  = 0       # next free register
        self.var_map      = {}      # var name → register index
        self.const_values = []      # constant pool [1.0, 2.5, ...]

    def new_reg(self):
        """Allocate a new register, return its index."""
        r = self.reg_counter
        self.reg_counter += 1
        return r

    def flatten(self, node):
        """Recursively flatten an AST node, return the register holding result."""

        if isinstance(node, Number):
            dest  = self.new_reg()
            const_idx = len(self.const_values)
            self.const_values.append(float(node.value))
            self.instructions.append(Instruction(LOAD, dest, const_idx))
            return dest

        if isinstance(node, Var):
            # each unique variable gets one register
            if node.name not in self.var_map:
                dest = self.new_reg()
                var_idx = len(self.var_map)
                self.var_map[node.name] = (dest, var_idx)
                self.instructions.append(Instruction(VAR, dest, var_idx))
            return self.var_map[node.name][0]

        if isinstance(node, BinOp):
            left_reg  = self.flatten(node.left)
            right_reg = self.flatten(node.right)
            dest      = self.new_reg()

            op = {"+": ADD, "-": SUB, "*": MUL, "/": DIV}[node.op]
            self.instructions.append(Instruction(op, dest, left_reg, right_reg))
            return dest

        raise TypeError(f"Unknown node: {type(node)}")

    def get_flat(self):
        """Return flat int list ready for GPU upload.
           Layout: [op, dest, src1, src2,  op, dest, src1, src2, ...]
        """
        flat = []
        for instr in self.instructions:
            flat.extend(instr.to_list())
        return flat