import os
import pickle
from concurrent.futures import ThreadPoolExecutor
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


def _resolve_workers(workers) -> int:
    """Normalise a workers argument: None/0 → all CPUs, else positive int."""
    if workers is None or workers == 0:
        return os.cpu_count() or 1
    if isinstance(workers, bool) or not isinstance(workers, int) \
            or workers < 1:
        raise ValueError(
            f"workers must be a positive int (or None/0 for all CPUs), "
            f"got {workers!r}")
    return workers


def _item_to_node(item):
    """Coerce a .toy path, inline source string, or AST node to an AST."""
    if isinstance(item, str):
        if item.endswith(".toy"):
            if not os.path.isfile(item):
                raise FileNotFoundError(f"Source file not found: {item!r}")
            return _source_to_ast(item)
        tokens = Lexer.tokenize(item)
        return Parser.parse(tokens)
    return item


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
    def compile_many(items, workers=1) -> list:
        """
        Compile many programs in memory (nothing is written to disk).

        *items* is a list of .toy paths, inline source strings, or AST
        nodes. workers=1 runs sequentially; >1 (or None/0 = all CPUs)
        compiles across a thread pool (lex/parse/emit touch no shared
        state, so this is thread-safe).

        Returns [list[Instr], ...] in input order.
        """
        items = list(items)
        workers = _resolve_workers(workers)

        def _one(item):
            instructions: list[Instr] = []
            Compiler._emit(_item_to_node(item), instructions)
            return instructions

        if workers == 1 or len(items) < 2:
            return [_one(item) for item in items]
        with ThreadPoolExecutor(max_workers=workers) as ex:
            return list(ex.map(_one, items))

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