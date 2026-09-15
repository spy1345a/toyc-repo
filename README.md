# toyc

Toy expression compiler (`+ - * /`, parens, variables) with CPU and
Vulkan GPU backends. Both execute the same AST — compare the results.

```python
from toyc import Cpu, GpuVulkan

Cpu.run("1 + 2")          # 3   (ints stay ints)
GpuVulkan.run("1 + 2")    # 3.0 (GPU is float32)
```

## Install

```bash
pip install toyc
sudo apt install glslang-tools   # builds the bundled shaders on first run
```

Python 3.10+. Needs a Vulkan driver.

## Usage

```python
Cpu.run("program.toy")                          # file
Cpu.run("program.toy", env={"a": 1})            # variables
Cpu.run("program.toy", silent=True)             # capture, don't print
Cpu.run("out.toyc")                             # compiled file
Compiler.compile("program.toy")                 # write program.toyc

GpuVulkan.run("program.toy", env={"a": 10.0, "b": 5.0})
GpuVulkan.run("program.toy", env=[{...}, {...}])  # auto-batch → [..]
GpuVulkan.run("program.toy", cache=False)         # skip .toyc write
GpuVulkan.run("program.toyc")                     # cached result, no dispatch
```

`run()` takes an inline string, a `.toy` path, or a `.toyc` path.
`.toyc` caches hold a scalar (single) or a float list (batch).

## GPU session

Setup costs ~15–40 ms once; a dispatch ~0.5 ms. One shared session
per process, thread-safe, never torn down per call:

```python
GpuVulkan.startup()    # optional warmup; first call does it lazily
GpuVulkan.run(...)
GpuVulkan.shutdown()   # also automatic at process exit
```

## Batch (where the GPU wins)

One expression, many variable sets, one dispatch per chunk:

```python
results = GpuVulkan.run_batch("prog.toy", [{"a": 1.0}, {"a": 2.0}])
```

Chunking at the recommended batch size is automatic. Singles → CPU
(~0.1 ms vs ~1 ms); bulk → GPU (~23× at N=2000 on RX 580).

## Timing & bench

`timed=True` prints a report and returns `(result, timing)` (seconds):

```python
cpu_val, cpu_t = Cpu.run("program.toy", timed=True)
gpu_val, gpu_t = GpuVulkan.run("program.toy", timed=True)
```

`toyc.bench` needs no loops from you — one equation + knobs, test
values auto-generated:

```python
from toyc.bench import bench, batch_bench, summarize, to_csv

rows = bench("a + b * 2", backend="cpu", n=100, repeat=5)
rows = batch_bench("a + b * 2", backend="vulkan", n=2000, repeat=3)
to_csv(rows, "timings.csv")

import pandas as pd                       # your plotting, your deps
pd.DataFrame(rows).groupby("backend")["total"].mean().plot.bar()
```

Rows carry `backend, program, mode, n, repeat, total,
total_time_taken, per_eval, check_err` + per-stage columns. CPU batch
rows add `threads` (workers used). GPU batch rows come one per chunk
and add `num_batches` (chunks in the run), `batch_size` (instances
per chunk), `batch_index` (which chunk, 0-based) and `chunk_n`
(instances in it).
`summarize(rows)` collapses to one row per (backend, program).

CPU batch runs multithreaded automatically
(`batch_bench(..., threads=None)` = all CPUs, `1` = sequential),
and `Cpu.run_batch(prog, envs, workers=…)` / `Compiler.compile_many(
items, workers=…)` expose the same control directly. Note: CPython
threads share the GIL, so threaded CPU suits big batches and typically
gains far less than the worker count — measure, don't assume.

## Semantics (CPU ≡ GPU)

| Case | Behavior |
|------|----------|
| Value | Numerically equal (`15` vs `15.0` — GPU is float32, ~1e-6 error) |
| `1/0`, `1/(a-a)`, `0/0` | `ZeroDivisionError("Division by zero in VM")` |
| Missing variable | `NameError("Undefined variable: 'q'")` |

Pipeline: Lexer → Parser → AST → Evaluator/Cpu, or Flattener →
compute shader (`toyc/gpu/vulkan/comp*.glsl`).

## License

See [LICENSE](toyc/LICENSE).
