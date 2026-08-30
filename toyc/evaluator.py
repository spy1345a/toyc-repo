# compiler/evaluator.py

from .ast_nodes import Number, Var, BinOp

class Evaluator:
    def __init__(self, env=None):
        # env holds variable values e.g {"a": 10, "b": 5}
        self.env = env or {}

    def eval(self, node):
        # base case — just return the number
        if isinstance(node, Number):
            return node.value

        # variable lookup
        if isinstance(node, Var):
            if node.name not in self.env:
                raise NameError(f"Undefined variable: {node.name!r}")
            return self.env[node.name]

        # recursive case — evaluate both sides then apply operator
        if isinstance(node, BinOp):
            left  = self.eval(node.left)
            right = self.eval(node.right)

            if node.op == "+": return left + right
            if node.op == "-": return left - right
            if node.op == "*": return left * right
            if node.op == "/":
                if right == 0:
                    raise ZeroDivisionError("Division by zero in expression")
                return left / right

            raise ValueError(f"Unknown operator: {node.op!r}")

        raise TypeError(f"Unknown AST node: {type(node)}")