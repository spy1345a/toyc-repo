# compiler/compiler.py
#
# Responsible for:
#   • Defining the Instr dataclass (the instruction set)
#   • .toyc binary serialisation / deserialisation helpers
#   • The Compiler class  (AST → list[Instr], with disk write)
#
# It knows nothing about execution — import vm.py for that.
#
# Two calling styles are supported:
#
#   A) AST-first  (you already have an AST node):
#         Compiler.compile(ast)                    → writes out.toyc
#         Compiler.compile(ast, path="foo.toyc")   → writes foo.toyc
#         Compiler.compile(ast, path=None)          → memory only
#
#   B) File-first (point straight at a .toy source file):
#         Compiler.compile("script.toy")            → writes script.toyc
#         Compiler.compile("script.toy", path=None) → memory only
#
#      The Lexer and Parser are invoked internally so the caller doesn't
#      need to tokenise or build an AST manually.

import os
import pickle
from dataclasses import dataclass
from typing      import Any

from .ast_nodes import Number, Var, BinOp
from .lexer     import Lexer
from .parser    import Parser


# ── instruction set ───────────────────────────────────────────────────────────
# These map 1-to-1 to SPIR-V ops later:
#   PUSH  → OpConstant
#   LOAD  → OpLoad
#   ADD   → OpFAdd
#   SUB   → OpFSub
#   MUL   → OpFMul
#   DIV   → OpFDiv

@dataclass
class Instr:
    op:  str
    arg: Any = None

    def __repr__(self):
        return f"{self.op} {self.arg!r}" if self.arg is not None else self.op


# ── .toyc binary format ───────────────────────────────────────────────────────
# Layout:
#   bytes 0-3 : MAGIC  b'\x54\x4F\x59\x43'  ("TOYC")
#   bytes 4+  : pickle.dumps(list[Instr])
#
# The magic header lets Cpu.run() detect compiled files by peeking at raw
# bytes, without relying on the file extension alone.

MAGIC = b'\x54\x4F\x59\x43'   # "TOYC"


def write_bytecode(path: str, instructions: list) -> None:
    """Serialise *instructions* to a .toyc file at *path*."""
    with open(path, "wb") as f:
        f.write(MAGIC)
        pickle.dump(instructions, f)


def read_bytecode(path: str) -> list:
    """Deserialise a .toyc file and return the list[Instr]."""
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != MAGIC:
            raise ValueError(
                f"{path!r} is not a valid .toyc file (bad magic bytes)"
            )
        return pickle.load(f)


def is_compiled_bytecode(path: str) -> bool:
    """Return True if *path* starts with the TOYC magic number."""
    try:
        with open(path, "rb") as f:
            return f.read(4) == MAGIC
    except OSError:
        return False


# ── helpers ───────────────────────────────────────────────────────────────────

def _source_to_ast(toy_path: str):
    """Read a .toy file and return its parsed AST using Lexer + Parser."""
    with open(toy_path, "r", encoding="utf-8") as f:
        source = f.read()
    tokens = Lexer.tokenize(source)
    return Parser.parse(tokens)


# ── Compiler: AST → bytecode ──────────────────────────────────────────────────

class Compiler:
    """Walk an AST and emit a flat list[Instr].

    Calling styles
    --------------
    # A) You already have tokens / an AST (original workflow):
    tokens = Lexer.tokenize(code)
    ast    = Parser.parse(tokens)
    Compiler.compile(ast)                     # writes out.toyc
    Compiler.compile(ast, path="expr.toyc")   # writes expr.toyc
    Compiler.compile(ast, path=None)          # memory only

    # B) Point straight at a .toy source file:
    Compiler.compile("script.toy")            # reads script.toy,
                                              # writes script.toyc next to it
    Compiler.compile("script.toy", path=None) # reads script.toy,
                                              # memory only (no .toyc written)
    """

    @staticmethod
    def compile(node_or_path, path: str = "out.toyc") -> list:
        """
        Compile source or an AST node into bytecode.

        Parameters
        ----------
        node_or_path : AST node  OR  str
            • An AST node  → compiled directly (original behaviour).
            • A str ending in ``.toy``  → the file is read, lexed, and parsed
              first; the default output path becomes a sibling .toyc with the
              same stem (overrides the ``path`` default of ``"out.toyc"``).

        path : str | None, optional
            Where to write the .toyc file.
            • ``"out.toyc"`` (default) when *node_or_path* is an AST node.
            • Auto-derived sibling path   when *node_or_path* is a .toy file.
            • ``None``  → compile but do not write anything to disk.
            The .toyc extension is enforced regardless of what is passed.

        Returns
        -------
        list[Instr]
            Flat bytecode list (always returned).
        """

        # ── resolve the input ────────────────────────────────────────────────

        if isinstance(node_or_path, str):
            # File-first mode
            toy_path = node_or_path
            if not toy_path.endswith(".toy"):
                raise ValueError(
                    f"Expected a .toy source file, got: {toy_path!r}"
                )
            if not os.path.isfile(toy_path):
                raise FileNotFoundError(f"Source file not found: {toy_path!r}")

            node = _source_to_ast(toy_path)

            # Derive sibling .toyc path unless the caller overrode it
            if path == "out.toyc":   # still the default → replace with sibling
                path = os.path.splitext(os.path.abspath(toy_path))[0] + ".toyc"

        else:
            # AST-first mode — node_or_path is an AST node
            node = node_or_path
            # path stays as whatever the caller passed ("out.toyc" or explicit)

        # ── emit bytecode ────────────────────────────────────────────────────

        instructions: list[Instr] = []
        Compiler._emit(node, instructions)

        # ── write to disk ────────────────────────────────────────────────────

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
            Compiler._emit(node.left,  out)   # push left operand
            Compiler._emit(node.right, out)   # push right operand
            op_map = {"+": "ADD", "-": "SUB", "*": "MUL", "/": "DIV"}
            out.append(Instr(op_map[node.op]))

        else:
            raise TypeError(f"Unknown AST node: {type(node).__name__}")