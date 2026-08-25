# compiler/ast_nodes.py

from dataclasses import dataclass
from typing import Any

@dataclass
class Number:
    value: float | int

    def __repr__(self):
        return f"Number({self.value})"

    def to_dict(self):
        return {"type": "Number", "value": self.value}

    @staticmethod
    def from_dict(d):
        return Number(d["value"])


@dataclass
class Var:
    name: str

    def __repr__(self):
        return f"Var({self.name!r})"

    def to_dict(self):
        return {"type": "Var", "name": self.name}

    @staticmethod
    def from_dict(d):
        return Var(d["name"])


@dataclass
class BinOp:
    op:    str
    left:  Any
    right: Any

    def __repr__(self):
        return f"BinOp({self.op!r}, {self.left}, {self.right})"

    def to_dict(self):
        return {
            "type":  "BinOp",
            "op":    self.op,
            "left":  self.left.to_dict(),
            "right": self.right.to_dict(),
        }

    @staticmethod
    def from_dict(d):
        return BinOp(
            op    = d["op"],
            left  = node_from_dict(d["left"]),
            right = node_from_dict(d["right"]),
        )


def node_from_dict(d):
    """Reconstruct any AST node from a dict."""
    kind = d["type"]
    if kind == "Number": return Number.from_dict(d)
    if kind == "Var":    return Var.from_dict(d)
    if kind == "BinOp":  return BinOp.from_dict(d)
    raise ValueError(f"Unknown node type: {kind!r}")