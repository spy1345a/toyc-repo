import os
import pickle
from dataclasses import dataclass
from typing import Any

from .ast_nodes import Number, Var, BinOp
from .lexer import Lexer
from .parser import Parser


@dataclass
class Instr:
    op: str
    arg: Any = None

    def __repr__(self):
        return f"{self.op} {self.arg!r}" if self.arg is not None else self.op


MAGIC = b'\x54\x4F\x59\x43'


def write_bytecode(path: str, instructions: list) -> None:
    with open(path, "wb") as f:
        f.write(MAGIC)
        pickle.dump(instructions, f)


def read_bytecode(path: str) -> list:
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != MAGIC:
            raise ValueError(f"{path!r} is not a valid .toyc file (bad magic bytes)")
        return pickle.load(f)


def is_compiled_bytecode(path: str) -> bool:
    try:
        with open(path, "rb") as f:
            return f.read(4) == MAGIC
    except OSError:
        return False


def _source_to_ast(toy_path: str):
    with open(toy_path, "r", encoding="utf-8") as f:
        source = f.read()
    tokens = Lexer.tokenize(source)
    return Parser.parse(tokens)


class Compiler:
    @staticmethod
    def compile(node_or_path, path: str = "out.toyc") -> list:
        if isinstance(node_or_path, str):
            toy_path = node_or_path
            if not toy_path.endswith(".toy"):
                raise ValueError(f"Expected a .toy source file, got: {toy_path!r}")
            if not os.path.isfile(toy_path):
                raise FileNotFoundError(f"Source file not found: {toy_path!r}")

            node = _source_to_ast(toy_path)

            if path == "out.toyc":
                path = os.path.splitext(os.path.abspath(toy_path))[0] + ".toyc"
        else:
            node = node_or_path

        instructions: list[Instr] = []
        Compiler._emit(node, instructions)

        if path is not None:
            if not path.endswith(".toyc"):
                path = os.path.splitext(path)[0] + ".toyc"
            write_bytecode(path, instructions)

        return instructions

    @staticmethod
    def _emit(node, out: list) -> None:
        if isinstance(node, Number):
            out.append(Instr("PUSH", node.value))
        elif isinstance(node, Var):
            out.append(Instr("LOAD", node.name))
        elif isinstance(node, BinOp):
            Compiler._emit(node.left, out)
            Compiler._emit(node.right, out)
            op_map = {"+": "ADD", "-": "SUB", "*": "MUL", "/": "DIV"}
            out.append(Instr(op_map[node.op]))
        else:
            raise TypeError(f"Unknown AST node: {type(node).__name__}")