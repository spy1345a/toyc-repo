# toyc/bench.py
#
# Two-function benchmarking: bench() for single-dispatch runs,
# batch_bench() for data-parallel batch runs.
#
# Both take ONE equation plus the required knobs — program, backend,
# n, repeat — and generate the n test value sets themselves (uniform
# 1.0–10.0), so there is nothing else to write::
#
#     from toyc.bench import bench, batch_bench
#
#     rows = bench("a + b * 2", backend="cpu", n=100, repeat=5)
#     rows = batch_bench("a + b * 2", backend="vulkan", n=2000, repeat=3)
#
# An optional ``env`` argument replaces the generator: a template dict
# pins its names in every set (the rest stay auto-generated), and a
# list of dicts supplies every instance outright::
#
#     rows = bench("x / k", backend="cpu", n=100, repeat=5, seed=7,
#                  env={"k": 5})                       # pin k, sweep x
#     rows = batch_bench("x * k", backend="vulkan", n=3, repeat=2,
#                        env=[{"x": 1.0, "k": 4}, {"x": 2.0, "k": 4},
#                             {"x": 3.0, "k": 4}])     # all supplied
#
# Everything is dependency-free (stdlib only). Each call returns plain
# dict rows ready for pandas / plotting with zero conversion::
#
#     import pandas as pd
#     df = pd.DataFrame(rows)
#     df.groupby("backend")["total_time_taken"].mean().plot.bar()
#
# All timing values are SECONDS (perf_counter). Multiply by 1e3 for ms.
# Stages that did not run in a given row are None (-> NaN in pandas).

import contextlib
import csv
import io
import os
import random
import tempfile

from .lexer import Lexer, IDENT
from .vm import Cpu, GpuVulkan

__all__ = ["bench", "batch_bench", "summarize", "to_csv"]

_BACKENDS = ("cpu", "vulkan")

# Union of every stage key both backends can report. Columns are always
# present in the same order so CSVs concat cleanly.
_STAGE_COLUMNS = ("resolve", "execute", "flatten", "select", "init",
                  "exec", "teardown", "decode")


def _check_backend(backend: str) -> str:
    if backend not in _BACKENDS:
        raise ValueError(
            f"backend must be one of {list(_BACKENDS)}, "
            f"got {backend!r}")
    return backend


def _check_counts(n: int, repeat: int) -> None:
    if not isinstance(n, int) or isinstance(n, bool) or n < 1:
        raise ValueError(f"n must be a positive int, got {n!r}")
    if not isinstance(repeat, int) or isinstance(repeat, bool) \
            or repeat < 1:
        raise ValueError(f"repeat must be a positive int, got {repeat!r}")


def _expr_names(program: str) -> list:
    """Variable names used by an equation, in first-use order."""
    if os.path.isfile(program):
        with open(program, "r", encoding="utf-8") as f:
            text = f.read()
    else:
        text = program
    names = []
    for tok in Lexer.tokenize(text):
        if tok.type == IDENT and tok.value not in names:
            names.append(tok.value)
    return names


def _gen_sets(names: list, n: int, seed=None) -> list:
    """
    Generate *n* variable sets with a for loop (uniform 1.0–10.0, so
    divisors stay safely non-zero). One set per instance to evaluate.
    """
    rng = random.Random(seed)
    sets = []
    for _ in range(n):
        env = {}
        for name in names:
            env[name] = rng.uniform(1.0, 10.0)
        sets.append(env)
    return sets


def _resolve_sets(names: list, n: int, seed, env) -> list:
    """
    Build the variable-set list handed to the backends.

    ``env=None``      -> ``_gen_sets(names, n, seed)`` — auto-generated
                         uniform 1.0–10.0 sets, unchanged behavior.
    ``env=<dict>``    -> *template* mode: ``n`` copies of the dict, i.e.
                         names present in it stay pinned at the exact
                         user value in every set; the remaining (free)
                         program variables are filled per instance via
                         ``_gen_sets`` (seeded by ``seed``).
    ``env=<list[dict]>`` -> the user supplies every instance outright;
                         used as-is after validation (the caller derives
                         the row count from ``len(sets)``).

    Missing program variables in a supplied env raise ``ValueError``
    here, before any timing runs (the backends raise ``NameError`` for
    the same condition — this catches it earlier with a clearer
    message). The resolved list length is always ``n`` except in list
    mode, where it is ``len(env)``.
    """
    if env is None:
        return _gen_sets(names, n, seed)

    if isinstance(env, dict):
        # Pinned names keep their exact value; only the free program
        # variables are randomized (never re-randomize pinned values).
        free = [name for name in names if name not in env]
        sets = []
        for extras in _gen_sets(free, n, seed):
            s = dict(extras)
            s.update(env)
            sets.append(s)
        return sets

    # list[dict] (or other iterable of dict) → use as-is, validated.
    if isinstance(env, str) or not hasattr(env, "__iter__"):
        raise TypeError(
            "env must be None, a dict (template), or a non-empty "
            f"list of dicts; got {type(env).__name__}")
    env_list = list(env)
    if not env_list:
        raise ValueError(
            "env must not be empty; the backends need at least one "
            "variable set")
    for i, e in enumerate(env_list):
        if not isinstance(e, dict):
            raise TypeError(
                f"env[{i}] must be a dict, got {type(e).__name__}")
        for name in names:
            if name not in e:
                raise ValueError(
                    f"env[{i}] is missing variable {name!r} used by "
                    f"the program (the backends raise NameError for "
                    f"this) — supply every variable in every env dict, "
                    f"or pass a template dict instead")
    return env_list


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


def _prepare(program: str) -> tuple:
    """Accept one equation string or .toy path. Returns (path, cleanup)."""
    if isinstance(program, (list, tuple)):
        raise ValueError(
            "pass a single equation string, e.g. "
            'bench("a + b * 2", backend="cpu", n=100, repeat=5)')
    if not isinstance(program, str):
        raise TypeError(
            f"program must be an equation string or .toy path, "
            f"got {type(program).__name__}")
    if os.path.isfile(program):
        return program, False
    return _temp_toy(program)


def _label(program: str) -> str:
    """Short human-readable label for a program (expression text)."""
    if os.path.isfile(program):
        try:
            with open(program, "r", encoding="utf-8") as f:
                text = f.read().strip()
            if program.endswith(".toy") and len(text) < 120:
                return text
        except OSError:
            pass
    return program


def _stages(timing: dict) -> dict:
    return {stage: timing.get(stage) for stage in _STAGE_COLUMNS}


def bench(program, backend, n, repeat, seed=None, verbose=False,
          debug=False, env=None) -> list:
    """
    Run ONE equation ``n`` times per repeat (single-dispatch path) on
    ``backend`` ("cpu" or "vulkan"), repeating the whole thing
    ``repeat`` times. Test values are auto-generated (n sets, uniform
    1.0–10.0; pass ``seed`` to reproduce them) unless ``env`` is given:

    ``env=None``       auto-generate all n sets (default, as above).
    ``env={...}``      one template dict: names present in it stay
                       pinned at their exact value in all n sets, the
                       remaining program variables are filled per
                       instance (seeded by ``seed``).
    ``env=[{...}, …]`` supplies every instance outright; ``n`` becomes
                       informational (``len(env)`` drives the row count
                       and the ``n`` column).

    Returns one row-dict per run: ``backend, program, mode="single",
    n, repeat, inst, result, total, total_time_taken, per_eval``
    (= total, one evaluation) plus per-stage columns (seconds, None
    if N/A; ``threads``/``batch_index``/``chunk_n`` stay None on
    single rows) — ready for ``pd.DataFrame(rows)``.
    Per-run printouts are suppressed unless ``verbose=True``.

    Examples
    --------
    rows = bench("x / k", backend="cpu", n=100, repeat=5, seed=7,
                 env={"k": 5})                    # pin k, sweep x
    rows = bench("x", backend="cpu", n=3, repeat=1,
                 env=[{"x": 1.0}, {"x": 2.0}, {"x": 3.0}])
    """
    if isinstance(program, (list, tuple)):
        raise ValueError(
            "pass a single equation string, e.g. "
            'bench("a + b * 2", backend="cpu", n=100, repeat=5)')
    _check_backend(backend)
    _check_counts(n, repeat)
    names = _expr_names(program)
    sets = _resolve_sets(names, n, seed, env)
    n = len(sets)
    prog, cleanup = _prepare(program)
    if backend == "vulkan":
        GpuVulkan.startup(debug=debug)

    rows = []
    try:
        for r in range(repeat):
            for i, env in enumerate(sets):
                if backend == "cpu":
                    call = lambda: Cpu.run(prog, env=env, silent=True,
                                           timed=True)
                    result, timing = _run_quiet(call, verbose)
                else:
                    def call():
                        return GpuVulkan.run(
                            prog, env=env, silent=True, debug=debug,
                            timed=True, cache=False)
                    result, timing = _run_quiet(call, verbose)
                row = {
                    "backend":  backend,
                    "program":  _label(program),
                    "mode":     "single",
                    "n":        n,
                    "repeat":   r,
                    "inst":     i,
                    "result":   result,
                    "total_time_taken": timing.get("total"),
                    "per_eval": timing.get("total"),
                    "check_err": None,
                    "num_batches": None,
                    "batch_size":  None,
                    "batch_index": None,
                    "chunk_n":     None,
                    "threads":     None,
                }
                row.update(_stages(timing))
                rows.append(row)
    finally:
        if cleanup:
            _cleanup_toy(prog)
    return rows


def batch_bench(program, backend, n, repeat, seed=None, verbose=False,
                debug=False, threads=None, env=None) -> list:
    """
    Run ONE equation over ``n`` instances per repeat on ``backend``
    ("cpu" or "vulkan"), repeating the whole thing ``repeat`` times.
    Test values are auto-generated like in bench() unless ``env`` is
    given — same three forms as there:

    ``env=None``       auto-generate all n sets (default, uniform
                       1.0–10.0; ``seed`` reproduces them).
    ``env={...}``      one template dict: names present in it stay
                       pinned at their exact value in every set, the
                       remaining program variables are filled per
                       instance (seeded by ``seed``).
    ``env=[{...}, …]`` supplies every instance outright; the row count
                       comes from ``len(env)`` and the ``n`` column
                       reports it.

    Examples
    --------
    rows = batch_bench("x * k", backend="vulkan", n=3, repeat=2,
                       env=[{"x": 1.0, "k": 4}, {"x": 2.0, "k": 4},
                            {"x": 3.0, "k": 4}])  # every instance
    rows = batch_bench("x / k", backend="cpu", n=100, repeat=5,
                       seed=7, env={"k": 5})       # pin k, sweep x

    CPU rows (one per repeat): the loop runs on ``threads`` workers
    (None = all CPUs, 1 = sequential) via Cpu.run_batch; the worker
    count used lands in the ``threads`` column (None on vulkan rows).

    Vulkan rows (one per dispatch chunk per repeat): each row records
    ``batch_index`` (which chunk, 0-based), ``chunk_n`` (instances in
    it), and that chunk's own flatten+exec time. ``num_batches`` /
    ``batch_size`` describe the whole run. ``check_err`` is the abs
    error of the chunk's first instance against a CPU reference run.

    Ready for ``pd.DataFrame(rows)``.
    """
    if isinstance(program, (list, tuple)):
        raise ValueError(
            "pass a single equation string, e.g. "
            'batch_bench("a + b * 2", backend="vulkan", n=2000, '
            "repeat=3)")
    _check_backend(backend)
    _check_counts(n, repeat)
    names = _expr_names(program)
    sets = _resolve_sets(names, n, seed, env)
    n = len(sets)
    prog, cleanup = _prepare(program)
    if backend == "vulkan":
        GpuVulkan.startup(debug=debug)

    rows = []
    try:
        for r in range(repeat):
            if backend == "cpu":
                def call():
                    return Cpu.run_batch(prog, sets, workers=threads,
                                         silent=True, timed=True)
                cpu_vals, timing = _run_quiet(call, verbose)
                rows.append({
                    "backend":   backend,
                    "program":   _label(program),
                    "mode":      "batch",
                    "n":         n,
                    "repeat":    r,
                    "batch_index": None,
                    "chunk_n":     None,
                    "inst":      None,
                    "result":    _first_value(cpu_vals),
                    "total_time_taken": timing.get("total"),
                    "per_eval":  timing.get("total") / n,
                    "check_err": None,
                    "num_batches": None,
                    "batch_size":  None,
                    "threads":     timing.get("workers"),
                    "resolve":     timing.get("resolve"),
                    "execute":     timing.get("execute"),
                    "flatten":   None,
                    "select":    None,
                    "init":      None,
                    "exec":      None,
                    "teardown":  None,
                    "decode":    None,
                })
            else:
                def call():
                    return GpuVulkan.run_batch(
                        prog, sets, silent=True, debug=debug,
                        timed=True, cache=False)
                values, timing = _run_quiet(call, verbose)
                chunks = timing.get("chunks") or [{
                    "batch_index": 0, "chunk_n": n,
                    "flatten": timing.get("flatten", 0.0) or 0.0,
                    "exec":    timing.get("exec", 0.0) or 0.0,
                }]
                off = 0
                for chunk in chunks:
                    cn = chunk["chunk_n"]
                    ctotal = (chunk.get("flatten") or 0.0) \
                        + (chunk.get("exec") or 0.0)
                    ref = Cpu.run(prog, env=sets[off], silent=True)
                    rows.append({
                        "backend":   backend,
                        "program":   _label(program),
                        "mode":      "batch",
                        "n":         n,
                        "repeat":    r,
                        "batch_index": chunk.get("batch_index"),
                        "chunk_n":     cn,
                        "inst":      None,
                        "result":    _first_value(values[off:off + cn]),
                        "total_time_taken": ctotal,
                        "per_eval":  ctotal / cn if cn else 0.0,
                        "check_err": abs(values[off] - ref),
                        "num_batches": timing.get("num_batches"),
                        "batch_size":  timing.get("batch_size"),
                        "threads":     None,
                        "resolve":   None,
                        "execute":   None,
                        "flatten":   chunk.get("flatten"),
                        "select":    None,
                        "init":      None,
                        "exec":      chunk.get("exec"),
                        "teardown":  None,
                        "decode":    None,
                    })
                    off += cn
    finally:
        if cleanup:
            _cleanup_toy(prog)
    return rows


def _first_value(values: list):
    """First value of a result vector — one scalar for the table."""
    return values[0] if values else None


def _run_quiet(call, verbose: bool):
    """Run *call*, suppressing its stdout printouts unless verbose."""
    if verbose:
        return call()
    with contextlib.redirect_stdout(io.StringIO()):
        return call()


def summarize(rows: list) -> list:
    """
    Collapse row-dicts to one summary row per (backend, program):
    ``backend, program, mode, n, runs, mean_total, std_total,
    mean_per_eval, mean_check_err, total_time_taken, threads`` (times
    in seconds; None where not applicable). ``total_time_taken`` is
    the summed wall time of the whole group — the number to compare
    backends by.
    Handy for printing tables and for ``pd.DataFrame(summarize(rows))``.
    """
    import statistics
    groups = {}
    for row in rows:
        key = (row.get("backend"), row.get("program"), row.get("mode"))
        groups.setdefault(key, []).append(row)
    summary = []
    for (backend, program, mode), rs in groups.items():
        totals = [r["total_time_taken"] for r in rs]
        per = [r["per_eval"] if r.get("per_eval") is not None
               else r["total_time_taken"] for r in rs]
        errs = [r["check_err"] for r in rs
                if r.get("check_err") is not None]
        summary.append({
            "backend":        backend,
            "program":        program,
            "mode":           mode,
            "n":              rs[0].get("n"),
            "runs":           len(rs),
            "mean_total":     statistics.mean(totals),
            "std_total":      statistics.stdev(totals)
                              if len(totals) > 1 else 0.0,
            "mean_per_eval":  statistics.mean(per),
            "mean_check_err": statistics.mean(errs) if errs else None,
            "total_time_taken": sum(totals),
            "num_batches":    rs[0].get("num_batches"),
            "batch_size":     rs[0].get("batch_size"),
            "threads":        rs[0].get("threads"),
        })
    return summary


def to_csv(rows: list, path: str) -> str:
    """Append row-dicts to *path* as CSV (pandas-readable). Creates file if needed. Returns path."""
    if not rows:
        raise ValueError("no rows to write")
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    fieldnames = ["backend", "program", "mode", "n", "repeat", "inst", "result",
                  "total_time_taken", "per_eval", "check_err",
                  "num_batches", "batch_size", "batch_index", "chunk_n",
                  "threads", *_STAGE_COLUMNS]
    file_exists = os.path.isfile(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        for row in rows:
            writer.writerow({k: ("" if v is None else v)
                             for k, v in row.items()
                             if k in fieldnames})
    return path
