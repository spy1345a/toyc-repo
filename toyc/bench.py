# toyc/bench.py
#
# Tabular benchmarking for CPU vs GPU backends.
#
# Everything here is dependency-free (stdlib only). Each profiling call
# returns a list of plain ``dict`` rows — one row per run — ready for
# pandas / plotting with zero conversion::
#
#     import pandas as pd
#     from toyc.bench import compare
#
#     a = 10
#     b = 20
#     rows = compare(["1 + 2 * 7", "a + b * 2"], repeats=5)
#     # env is auto-collected from your variables — no envs= needed
#     df = pd.DataFrame(rows)          # straight in, no reshaping
#     df.groupby("backend")["total"].mean().plot.bar()   # CPU vs GPU
#     df.to_csv("timings.csv", index=False)              # or use to_csv()
#
# All timing values are SECONDS (perf_counter). Multiply by 1e3 for ms.
# Stages that did not run in a given row are None (-> NaN in pandas).

import contextlib
import csv
import inspect
import io
import os
import struct
import tempfile

from .lexer import Lexer, IDENT
from .vm import Cpu, GpuVulkan

__all__ = ["profile", "compare", "compare_batch", "to_csv"]

# Union of every stage key both backends can report. Columns are always
# present in the same order so CSVs concat cleanly.
_STAGE_COLUMNS = ("resolve", "execute", "flatten", "select", "init",
                  "exec", "teardown", "decode")


def _expr_names(program) -> list[str]:
    """Variable names used by a program, in first-use order."""
    if isinstance(program, list):
        # CPU bytecode: LOAD instructions carry the variable name.
        names: list[str] = []
        for ins in program:
            if getattr(ins, "op", None) == "LOAD" \
                    and ins.arg not in names:
                names.append(ins.arg)
        return names
    if isinstance(program, str) and os.path.isfile(program):
        if not program.endswith(".toy"):
            return []  # .toyc cache: variable values already baked in
        try:
            with open(program, "r", encoding="utf-8") as f:
                text = f.read()
        except OSError:
            return []
    elif isinstance(program, str):
        text = program
    else:
        return []
    names = []
    for tok in Lexer.tokenize(text):
        if tok.type == IDENT and tok.value not in names:
            names.append(tok.value)
    return names


def _auto_env(program) -> dict:
    """
    Build an env dict for *program* from the caller's Python variables.
    Called directly from profile()/compare(), so the user frame is two
    levels up. Raises NameError naming the missing variables.
    """
    names = _expr_names(program)
    if not names:
        return {}
    frame = inspect.currentframe()
    try:
        caller = frame.f_back.f_back
        ns = dict(caller.f_globals)
        ns.update(caller.f_locals)
    finally:
        del frame
    env, missing, bad = {}, [], []
    for name in names:
        if name not in ns or name.startswith("__"):
            missing.append(name)
        elif isinstance(ns[name], bool) \
                or isinstance(ns[name], (int, float)):
            env[name] = ns[name]
        else:
            bad.append(name)
    if missing:
        raise NameError(
            f"Variable(s) {missing} in {program!r} are not defined in "
            "Python. Assign them (e.g. a = 10) or pass envs explicitly."
        )
    if bad:
        raise TypeError(
            f"Variable(s) {bad} in {program!r} must be numbers, got: "
            + ", ".join(f"{n}={ns[n]!r}" for n in bad)
        )
    return env


def _temp_toy(expression: str) -> tuple:
    """Write *expression* to a temp .toy file. Returns (path, cleanup=True)."""
    tmp = tempfile.NamedTemporaryFile("w", suffix=".toy", delete=False)
    try:
        tmp.write(expression)
        tmp.close()
        return tmp.name, True
    except Exception:
        tmp.close()
        raise


def _cleanup_toy(prog: str) -> None:
    """Remove a temp .toy file and its sibling .toyc cache, if present."""
    try:
        os.remove(prog)
    except OSError:
        pass
    try:
        os.remove(os.path.splitext(prog)[0] + ".toyc")
    except OSError:
        pass


def _label(program: str) -> str:
    """Short human-readable label for a program (expression text)."""
    if isinstance(program, str) and os.path.isfile(program):
        try:
            with open(program, "r", encoding="utf-8") as f:
                text = f.read().strip()
            if program.endswith(".toy") and len(text) < 120:
                return text
        except OSError:
            pass
    return program if isinstance(program, str) else repr(program)


def _row(backend: str, program: str, repeat: int,
         result, timing: dict) -> dict:
    row = {
        "backend": backend,
        "program": _label(program),
        "repeat":  repeat,
        "result":  result,
        "total":   timing.get("total"),
    }
    for stage in _STAGE_COLUMNS:
        row[stage] = timing.get(stage)
    return row


def profile(backend: str, program, env: dict = None, repeats: int = 3,
            debug: bool = False, verbose: bool = False) -> list[dict]:
    """
    Run *program* ``repeats`` times on ``backend`` ("cpu" or "gpu") and
    return one row-dict per run: ``backend, program, repeat, result,
    total`` plus one column per timing stage (seconds, None if N/A).

    *program* may be an expression string (written to a temp .toy file),
    a .toy path, a .toyc path, or — for "cpu" only — a list[Instr].

    If *env* is None (default), variable values are auto-collected from
    your Python variables, so this just works::

        a = 10
        b = 20
        profile("gpu", "a + b * 2")   # uses a=10, b=20

    Pass env explicitly to override, or for names not in scope.

    The shared GPU session is warmed up first so every timed repeat is
    steady-state. Per-run timing printouts are suppressed unless
    ``verbose=True``.
    """
    if backend not in ("cpu", "gpu"):
        raise ValueError(f"backend must be 'cpu' or 'gpu', got {backend!r}")

    if env is None:
        env = _auto_env(program)

    if backend == "gpu":
        GpuVulkan.startup(debug=debug)

    if isinstance(program, str) and not os.path.isfile(program) \
            and not program.endswith((".toy", ".toyc")):
        prog, cleanup = _temp_toy(program)
    else:
        prog, cleanup = program, False

    def _cpu():
        return Cpu.run(prog, env=env, silent=True, timed=True)

    def _gpu():
        raw, timing = GpuVulkan.run(prog, env=env, silent=True,
                                    debug=debug, timed=True)
        return struct.unpack("f", raw)[0], timing

    rows: list[dict] = []
    try:
        for i in range(repeats):
            run = _cpu if backend == "cpu" else _gpu
            if verbose:
                result, timing = run()
            else:
                with contextlib.redirect_stdout(io.StringIO()):
                    result, timing = run()
            rows.append(_row(backend, program, i, result, timing))
    finally:
        if cleanup:
            _cleanup_toy(prog)
    return rows


def compare(programs, envs=None, repeats: int = 3,
            debug: bool = False, verbose: bool = False) -> list[dict]:
    """
    Profile each program on BOTH backends. *programs* is a list of
    expressions / paths; *envs* an optional parallel list of env dicts
    (or a single shared dict). Omit *envs* entirely to auto-collect
    values from your Python variables::

        a = 10
        b = 20
        compare(["1 + 2 * 7", "a + b * 2"], repeats=5)

    Returns concatenated row-dicts, CPU rows first per program, ready
    for ``pd.DataFrame(rows)``.
    """
    if envs is None:
        # Plain loop (not a comprehension: comprehensions run in their
        # own frame, which would hide the caller's variables from
        # _auto_env's frame inspection).
        envs = []
        for prog in programs:
            envs.append(_auto_env(prog))
    elif isinstance(envs, dict):
        envs = [envs] * len(programs)
    if len(envs) != len(programs):
        raise ValueError("envs must be a dict or match programs in length")

    rows: list[dict] = []
    for prog, env in zip(programs, envs):
        rows += profile("cpu", prog, env=env, repeats=repeats)
        rows += profile("gpu", prog, env=env, repeats=repeats,
                        debug=debug, verbose=verbose)
    return rows


def to_csv(rows: list[dict], path: str) -> str:
    """Write row-dicts to *path* as CSV (pandas-readable). Returns path."""
    if not rows:
        raise ValueError("no rows to write")
    fieldnames = ["backend", "program", "n", "repeat", "result", "total",
                  "per_eval", "max_err", *_STAGE_COLUMNS]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames,
                                extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ("" if v is None else v)
                             for k, v in row.items()
                             if k in fieldnames})
    return path


def _batch_row(backend: str, program: str, n: int, repeat: int,
               total: float, per_eval: float, max_err: float,
               timing: dict) -> dict:
    """One compare_batch row: vector results collapse to max_err."""
    row = {
        "backend":  backend,
        "program":  _label(program),
        "n":        n,
        "repeat":   repeat,
        "result":   None,
        "total":    total,
        "per_eval": per_eval,
        "max_err":  max_err,
    }
    for stage in _STAGE_COLUMNS:
        row[stage] = timing.get(stage)
    return row


def _as_env_sets(program, env_sets) -> list[dict]:
    """
    Normalise batch variable sets. Accepts a list of env dicts, or a
    column dict of equal-length value lists (zipped into sets)::

        compare_batch("a + b * 2", [{"a": 1, "b": 2}, {"a": 3, "b": 4}])
        compare_batch("a + b * 2", {"a": [1, 3], "b": [2, 4]})  # same
    """
    if isinstance(env_sets, dict):
        if not env_sets:
            raise ValueError("compare_batch needs at least one variable set")
        lengths = set()
        for name, vals in env_sets.items():
            if not isinstance(vals, (list, tuple)):
                raise ValueError(
                    f"Column {name!r} must be a list/tuple of values, "
                    f"got {type(vals).__name__}. Pass a list of env dicts "
                    "for explicit sets."
                )
            lengths.add(len(vals))
        if len(lengths) != 1:
            raise ValueError(
                "All value columns must have the same length, got "
                f"{sorted(lengths)}"
            )
        names = list(env_sets)
        return [dict(zip(names, vals)) for vals in zip(*env_sets.values())]
    if not isinstance(env_sets, (list, tuple)) or not env_sets:
        raise ValueError("compare_batch needs a non-empty list of env dicts")
    return list(env_sets)


def compare_batch(program, env_sets, repeats: int = 3,
                  debug: bool = False, verbose: bool = False) -> list[dict]:
    """
    Profile ONE program over many variable sets: sequential ``Cpu.run``
    loop ("cpu") versus data-parallel ``GpuVulkan.run_batch``
    ("gpu-batch", one shader invocation per instance, chunked at the
    recommended batch size).

    *env_sets* is a list of env dicts, or a column dict of equal-length
    value lists::

        compare_batch("a + b * 2", [{"a": 1, "b": 2}, {"a": 3, "b": 4}])
        compare_batch("a + b * 2", {"a": [1, 3], "b": [2, 4]})

    Returns one row-dict per repeat per backend with ``n`` (instance
    count), ``total`` / ``per_eval`` seconds, ``max_err`` (max abs
    GPU-vs-CPU difference; 0.0 on CPU rows, proving the GPU results),
    plus per-stage columns — ready for ``pd.DataFrame(rows)``.
    """
    sets = _as_env_sets(program, env_sets)
    n = len(sets)

    if isinstance(program, str) and not os.path.isfile(program) \
            and not program.endswith(".toy"):
        prog, cleanup = _temp_toy(program)
    else:
        prog, cleanup = program, False

    GpuVulkan.startup(debug=debug)

    def _cpu_all():
        total = resolve = execute = 0.0
        values = []
        for env in sets:
            out = Cpu.run(prog, env=env, silent=True, timed=True)
            values.append(out[0])
            total   += out[1]["total"]
            resolve += out[1]["resolve"]
            execute += out[1]["execute"]
        return values, {"total": total, "resolve": resolve,
                        "execute": execute}

    def _gpu_all():
        return GpuVulkan.run_batch(prog, sets, silent=True,
                                   debug=debug, timed=True)

    rows: list[dict] = []
    try:
        for i in range(repeats):
            if verbose:
                cpu_vals, cpu_t = _cpu_all()
                gpu_vals, gpu_t = _gpu_all()
            else:
                with contextlib.redirect_stdout(io.StringIO()):
                    cpu_vals, cpu_t = _cpu_all()
                    gpu_vals, gpu_t = _gpu_all()
            max_err = max(abs(g - c)
                          for g, c in zip(gpu_vals, cpu_vals))
            rows.append(_batch_row("cpu", program, n, i, cpu_t["total"],
                                   cpu_t["total"] / n, 0.0, cpu_t))
            rows.append(_batch_row("gpu-batch", program, n, i,
                                   gpu_t["total"],
                                   gpu_t["total"] / n, max_err, gpu_t))
    finally:
        if cleanup:
            _cleanup_toy(prog)
    return rows
