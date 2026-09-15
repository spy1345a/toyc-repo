# toyc/bench.py
#
# Two-function benchmarking: bench() for single-dispatch runs,
# batch_bench() for data-parallel batch runs.
#
# Both take ONE equation plus the required knobs — program, backend,
# n, repeat — and generate the n test value sets themselves, so there
# is nothing else to write::
#
#     from toyc.bench import bench, batch_bench
#
#     rows = bench("a + b * 2", backend="cpu", n=100, repeat=5)
#     rows = batch_bench("a + b * 2", backend="vulkan", n=2000, repeat=3)
#
# Everything is dependency-free (stdlib only). Each call returns plain
# dict rows ready for pandas / plotting with zero conversion::
#
#     import pandas as pd
#     df = pd.DataFrame(rows)
#     df.groupby("backend")["total"].mean().plot.bar()
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
          debug=False) -> list:
    """
    Run ONE equation ``n`` times per repeat (single-dispatch path) on
    ``backend`` ("cpu" or "vulkan"), repeating the whole thing
    ``repeat`` times. Test values are auto-generated (n sets, uniform
    1.0–10.0; pass ``seed`` to reproduce them).

    Returns one row-dict per run: ``backend, program, mode="single",
    n, repeat, inst, result, total, total_time_taken, per_eval``
    (= total, one evaluation) plus per-stage columns (seconds, None
    if N/A) — ready for ``pd.DataFrame(rows)``.
    Per-run printouts are suppressed unless ``verbose=True``.
    """
    if isinstance(program, (list, tuple)):
        raise ValueError(
            "pass a single equation string, e.g. "
            'bench("a + b * 2", backend="cpu", n=100, repeat=5)')
    _check_backend(backend)
    _check_counts(n, repeat)
    names = _expr_names(program)
    sets = _gen_sets(names, n, seed)
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
                    "total":    timing.get("total"),
                    "total_time_taken": timing.get("total"),
                    "per_eval": timing.get("total"),
                    "check_err": None,
                    "num_batches": None,
                    "batch_size":  None,
                }
                row.update(_stages(timing))
                rows.append(row)
    finally:
        if cleanup:
            _cleanup_toy(prog)
    return rows


def batch_bench(program, backend, n, repeat, seed=None, verbose=False,
                debug=False) -> list:
    """
    Run ONE equation over ``n`` instances per repeat on ``backend``
    ("cpu" = sequential loop, "vulkan" = one data-parallel
    ``run_batch`` dispatch, chunked automatically), repeating the whole
    thing ``repeat`` times. Test values are auto-generated like in
    bench().

    Returns one row-dict per repeat: ``backend, program,
    mode="batch", n, repeat, total, total_time_taken, per_eval``
    (= total/n), ``num_batches`` (dispatch chunks used),
    ``batch_size`` (instances per chunk) plus per-stage columns.
    ``check_err`` on vulkan rows is the abs error of instance
    0 against a CPU reference run (None on cpu rows).
    """
    if isinstance(program, (list, tuple)):
        raise ValueError(
            "pass a single equation string, e.g. "
            'batch_bench("a + b * 2", backend="vulkan", n=2000, '
            "repeat=3)")
    _check_backend(backend)
    _check_counts(n, repeat)
    names = _expr_names(program)
    sets = _gen_sets(names, n, seed)
    prog, cleanup = _prepare(program)
    if backend == "vulkan":
        GpuVulkan.startup(debug=debug)

    rows = []
    try:
        for r in range(repeat):
            if backend == "cpu":
                def call():
                    total = resolve = execute = 0.0
                    for env in sets:
                        out = Cpu.run(prog, env=env, silent=True,
                                      timed=True)
                        total   += out[1]["total"]
                        resolve += out[1]["resolve"]
                        execute += out[1]["execute"]
                    return None, {"total": total, "resolve": resolve,
                                  "execute": execute}
                _, timing = _run_quiet(call, verbose)
                check_err = None
            else:
                def call():
                    return GpuVulkan.run_batch(
                        prog, sets, silent=True, debug=debug,
                        timed=True, cache=False)
                values, timing = _run_quiet(call, verbose)
                ref = Cpu.run(prog, env=sets[0], silent=True)
                check_err = abs(values[0] - ref)
            row = {
                "backend":   backend,
                "program":   _label(program),
                "mode":      "batch",
                "n":         n,
                "repeat":    r,
                "inst":      None,
                "result":    None,
                "total":     timing.get("total"),
                "total_time_taken": timing.get("total"),
                "per_eval":  timing.get("total") / n,
                "check_err": check_err,
                "num_batches": timing.get("num_batches"),
                "batch_size":  timing.get("batch_size"),
            }
            row.update(_stages(timing))
            rows.append(row)
    finally:
        if cleanup:
            _cleanup_toy(prog)
    return rows


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
    mean_per_eval, mean_check_err, total_time_taken`` (times in
    seconds; None where not applicable). ``total_time_taken`` is the
    summed wall time of the whole group — the number to compare
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
        totals = [r["total"] for r in rs]
        per = [r["per_eval"] if r.get("per_eval") is not None
               else r["total"] for r in rs]
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
        })
    return summary


def to_csv(rows: list, path: str) -> str:
    """Write row-dicts to *path* as CSV (pandas-readable). Returns path."""
    if not rows:
        raise ValueError("no rows to write")
    fieldnames = ["backend", "program", "mode", "n", "repeat", "inst", "result",
                  "total", "total_time_taken", "per_eval", "check_err",
                  "num_batches", "batch_size", *_STAGE_COLUMNS]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames,
                                extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ("" if v is None else v)
                             for k, v in row.items()
                             if k in fieldnames})
    return path
