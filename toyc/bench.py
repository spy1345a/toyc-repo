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

__all__ = ["profile", "compare", "compare_batch", "summarize",
             "to_csv"]

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


def _check_backends(backends, valid: tuple) -> list:
    """Normalise the backends selector (str or sequence) and validate."""
    if isinstance(backends, str):
        backends = [backends]
    backends = list(backends)
    if not backends:
        raise ValueError(f"backends must be a non-empty subset of {list(valid)}")
    for b in backends:
        if b not in valid:
            raise ValueError(
                f"unknown backend {b!r}; choose from {list(valid)}")
    return backends


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
        "n":        1,
        "repeat":   repeat,
        "result":   result,
        "total":    timing.get("total"),
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
            backends=("cpu", "gpu"),
            debug: bool = False, verbose: bool = False) -> list[dict]:
    """
    Profile each program on the selected backends (default both).
    *programs* is a list of expressions / paths; *envs* an optional
    parallel list of env dicts (or a single shared dict). Omit *envs*
    entirely to auto-collect values from your Python variables::

        a = 10
        b = 20
        compare(["1 + 2 * 7", "a + b * 2"], repeats=5)

    Pass ``backends=["cpu"]`` (or ``["gpu"]``) to run and save one
    backend at a time instead of both together::

        cpu_rows = compare(equations, backends=["cpu"])
        to_csv(cpu_rows, "cpu.csv")
        gpu_rows = compare(equations, backends=["gpu"])
        to_csv(gpu_rows, "gpu.csv")

    Returns concatenated row-dicts (one backend block per program, in
    *backends* order), ready for ``pd.DataFrame(rows)``.
    """
    backends = _check_backends(backends, ("cpu", "gpu"))
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
        for backend in backends:
            rows += profile(backend, prog, env=env, repeats=repeats,
                            debug=debug, verbose=verbose)
    return rows


def summarize(rows: list[dict]) -> list[dict]:
    """
    Collapse row-dicts to one summary row per (backend, program):
    ``backend, program, n, runs, mean_total, std_total, mean_per_eval,
    mean_max_err`` (times in seconds). std over a single repeat is 0.0.
    Handy for printing tables and for ``pd.DataFrame(summarize(rows))``.
    """
    import statistics
    groups: dict = {}
    for row in rows:
        key = (row.get("backend"), row.get("program"))
        groups.setdefault(key, []).append(row)
    summary = []
    for (backend, program), rs in groups.items():
        totals = [r["total"] for r in rs]
        per = [r["per_eval"] if r.get("per_eval") is not None
               else r["total"] for r in rs]
        errs = [r["max_err"] for r in rs
                if r.get("max_err") is not None]
        summary.append({
            "backend":       backend,
            "program":       program,
            "n":             rs[0].get("n"),
            "runs":          len(rs),
            "mean_total":    statistics.mean(totals),
            "std_total":     statistics.stdev(totals)
                             if len(totals) > 1 else 0.0,
            "mean_per_eval": statistics.mean(per),
            "mean_max_err":  statistics.mean(errs) if errs else None,
        })
    return summary


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


def _is_env_dict(obj) -> bool:
    """True if *obj* is a single variable set (dict of scalar values)."""
    return isinstance(obj, dict) and all(
        not isinstance(v, (list, tuple)) for v in obj.values())


def compare_batch(programs, env_sets, repeats: int = 3,
                  backends=("cpu", "gpu-batch"),
                  debug: bool = False, verbose: bool = False) -> list[dict]:
    """
    Profile programs over many variable sets: sequential ``Cpu.run``
    loop ("cpu") versus data-parallel ``GpuVulkan.run_batch``
    ("gpu-batch", one shader invocation per instance).

    Pass ``backends=["cpu"]`` (or ``["gpu-batch"]``) to run and save
    one backend at a time instead of both together::

        cpu_rows = compare_batch(eq, sets, backends=["cpu"])
        to_csv(cpu_rows, "batch_cpu.csv")
        gpu_rows = compare_batch(eq, sets, backends=["gpu-batch"])
        to_csv(gpu_rows, "batch_gpu.csv")

    Single program::

        compare_batch("a + b * 2", [{"a": 1, "b": 2}, {"a": 3, "b": 4}])
        compare_batch("a + b * 2", {"a": [1, 3], "b": [2, 4]})  # same

    Multiple equations simply share the one list you pass — each
    equation picks its own variables out of every set::

        sets = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
        compare_batch(["a + b", "a * b"], sets)   # both use both sets

    Different sets per equation: pass one entry per program, each a
    list of env dicts or a column dict::

        compare_batch(["a + b", "a * b"],
                      [[{"a": 1, "b": 2}], {"a": [1], "b": [2]}])

    Returns one row-dict per repeat per backend with ``n`` (instance
    count), ``total`` / ``per_eval`` seconds, ``max_err`` (max abs
    GPU-vs-CPU difference when both backends run, else None; 0.0 on
    CPU rows, proving the GPU results), plus per-stage columns —
    ready for ``pd.DataFrame(rows)``.
    """
    backends = _check_backends(backends, ("cpu", "gpu-batch"))
    if isinstance(programs, str):
        return _compare_batch_one(programs, env_sets, repeats,
                                  debug, verbose, backends)
    if isinstance(env_sets, dict) or (
            isinstance(env_sets, (list, tuple)) and len(env_sets) > 0
            and all(_is_env_dict(e) for e in env_sets)):
        # One shared sets-spec for every program: a column dict, or a
        # plain list of env dicts (each equation picks the variables it
        # needs out of every set).
        shared = env_sets
        rows: list[dict] = []
        for prog in programs:
            rows += _compare_batch_one(prog, shared, repeats,
                                       debug, verbose, backends)
        return rows
    if len(env_sets) != len(programs):
        raise ValueError(
            "env_sets must be a column dict, a single list of env dicts "
            "shared by all programs, or one entry per program "
            f"(got {len(env_sets)} entries for {len(programs)} programs)"
        )
    rows = []
    for prog, sets in zip(programs, env_sets):
        rows += _compare_batch_one(prog, sets, repeats, debug, verbose,
                                   backends)
    return rows


def _compare_batch_one(program, env_sets, repeats: int = 3,
                       debug: bool = False, verbose: bool = False,
                       backends=("cpu", "gpu-batch")) -> list[dict]:
    sets = _as_env_sets(program, env_sets)
    n = len(sets)

    if isinstance(program, str) and not os.path.isfile(program) \
            and not program.endswith(".toy"):
        prog, cleanup = _temp_toy(program)
    else:
        prog, cleanup = program, False

    if "gpu-batch" in backends:
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
            results = {}
            if verbose:
                if "cpu" in backends:
                    results["cpu"] = _cpu_all()
                if "gpu-batch" in backends:
                    results["gpu-batch"] = _gpu_all()
            else:
                with contextlib.redirect_stdout(io.StringIO()):
                    if "cpu" in backends:
                        results["cpu"] = _cpu_all()
                    if "gpu-batch" in backends:
                        results["gpu-batch"] = _gpu_all()
            if "cpu" in results and "gpu-batch" in results:
                max_err = max(abs(g - c) for g, c in
                              zip(results["gpu-batch"][0], results["cpu"][0]))
            else:
                max_err = None
            for backend in backends:
                values, timing = results[backend]
                rows.append(_batch_row(
                    backend, program, n, i, timing["total"],
                    timing["total"] / n,
                    0.0 if backend == "cpu" else max_err, timing))
    finally:
        if cleanup:
            _cleanup_toy(prog)
    return rows
