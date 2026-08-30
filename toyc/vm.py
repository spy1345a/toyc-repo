# compiler/vm.py
#
# Responsible for:
#   • The Cpu class       (execute bytecode on the host CPU)
#   • Input resolution    (list[Instr], .toyc path, .toy path) — shared by
#                          all backends so file loading lives in one place
#   • Re-exporting GpuVulkan so callers can import both backends from here:
#
#       from compiler.vm import Cpu, GpuVulkan
#
# Calling styles (identical interface for both backends):
#
#   Cpu.run("script.toy")          → lex+parse+compile in memory, run on CPU
#   Cpu.run("script.toyc")         → load compiled file, run on CPU
#   Cpu.run(bytecode_list)         → run in memory directly
#
#   GpuVulkan.run("script.toy")    → same input forms, dispatches on GPU
#   GpuVulkan.run("script.toyc")   → load compiled file, dispatch on GPU

import os
from typing import Any

from .compiler import (
    Instr,
    Compiler,
    read_bytecode,
    is_compiled_bytecode,
)
from .lexer  import Lexer
from .parser import Parser

# Re-export so `from compiler.vm import GpuVulkan` works
from .gpu.vulkan import GpuVulkan

from .gpu.opengl import GpuOpengl

__all__ = ["Cpu", "GpuVulkan"]


# ── Cpu: execute bytecode ─────────────────────────────────────────────────────

class Cpu:
    """Stack-based virtual machine.

    ``Cpu.run()`` accepts three forms of *program*:

    1. ``list[Instr]``
          Already-compiled bytecode; executed directly, no I/O.

    2. ``str`` ending in ``.toyc``
          Load the compiled file from disk and execute it.
          No lexing, parsing, or compiling happens.

    3. ``str`` ending in ``.toy``
          Read the source file, lex it, parse it, compile it in memory,
          then execute.  Nothing is written to disk.
    """

    @staticmethod
    def run(program, env: dict = None, silent: bool = False) -> Any:
        """
        Execute *program* and return the top-of-stack result.

        By default the result is printed to stdout automatically.
        Pass ``silent=True`` when you are capturing the return value yourself
        and do not want it printed as well.

        Parameters
        ----------
        program : list[Instr] | str
            Bytecode list, a .toyc compiled file path, or a .toy source path.
        env : dict, optional
            Variable bindings available to LOAD instructions  {name: value}.
        silent : bool, optional
            False (default) → result is printed before returning.
            True            → result is returned quietly, no stdout output.

        Examples
        --------
        Cpu.run("out.toyc")                        # prints result automatically
        value = Cpu.run("out.toyc", silent=True)   # capture only, no print
        """
        instructions = Cpu._resolve(program)
        result       = Cpu._execute(instructions, env or {})

        if not silent:
            print(result)

        return result

    # ── input resolution ──────────────────────────────────────────────────────

    @staticmethod
    def _resolve(program) -> list:
        """Return a list[Instr] no matter what form *program* arrives in."""

        # ① Already compiled in memory — use directly
        if isinstance(program, list):
            return program

        if not isinstance(program, str):
            raise TypeError(
                f"Cpu.run() expects a list[Instr] or a file path str, "
                f"got {type(program).__name__}"
            )

        path = program

        # ② .toyc path — load compiled bytecode, skip all compile steps
        if path.endswith(".toyc"):
            if not os.path.isfile(path):
                raise FileNotFoundError(f"Bytecode file not found: {path!r}")
            return read_bytecode(path)

        # ③ .toy source path — lex → parse → compile in memory, no disk write
        if path.endswith(".toy"):
            if not os.path.isfile(path):
                raise FileNotFoundError(f"Source file not found: {path!r}")
            return Cpu._compile_toy(path)

        # ④ Unknown extension — peek at magic bytes as a last resort
        if os.path.isfile(path) and is_compiled_bytecode(path):
            return read_bytecode(path)

        raise ValueError(
            f"Cannot load {path!r}: expected a .toy source or .toyc bytecode file"
        )

    @staticmethod
    def _compile_toy(toy_path: str) -> list:
        """
        Lex → parse → compile a .toy source file entirely in memory.
        Nothing is written to disk (path=None passed to Compiler.compile).
        """
        with open(toy_path, "r", encoding="utf-8") as f:
            source = f.read()

        tokens = Lexer.tokenize(source)
        ast    = Parser.parse(tokens)
        # path=None → memory only, no .toyc file created
        return Compiler.compile(ast, path=None)

    # ── core interpreter ──────────────────────────────────────────────────────

    @staticmethod
    def _execute(instructions: list, env: dict) -> Any:
        stack: list = []

        for instr in instructions:
            if instr.op == "PUSH":
                stack.append(instr.arg)

            elif instr.op == "LOAD":
                if instr.arg not in env:
                    raise NameError(f"Undefined variable: {instr.arg!r}")
                stack.append(env[instr.arg])

            elif instr.op == "ADD":
                b, a = stack.pop(), stack.pop()
                stack.append(a + b)

            elif instr.op == "SUB":
                b, a = stack.pop(), stack.pop()
                stack.append(a - b)

            elif instr.op == "MUL":
                b, a = stack.pop(), stack.pop()
                stack.append(a * b)

            elif instr.op == "DIV":
                b, a = stack.pop(), stack.pop()
                if b == 0:
                    raise ZeroDivisionError("Division by zero in VM")
                stack.append(a / b)

            else:
                raise RuntimeError(f"Unknown opcode: {instr.op!r}")

        if not stack:
            raise RuntimeError("Execution finished with an empty stack")

        return stack[-1]   # top of stack is the result