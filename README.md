# toyc

A toy expression compiler with CPU and GPU (Vulkan compute) backends,
written in Python. Source text is lexed, parsed to an AST, and executed
either by a tree-walking CPU evaluator or by a compute shader acting as
a register machine — then both results can be compared.

## Install

```bash
pip install toyc
```

You also need a Vulkan driver plus one GLSL→SPIR-V compiler (only used
to build the bundled shaders, automatically at first run):

```bash
sudo apt install glslang-tools   # glslangValidator (preferred)
# or
sudo apt install glslc           # shaderc (fallback)
```

Requires Python 3.10+ and the `vulkan` + `pyopengl` packages (installed
as dependencies).

## The language

Arithmetic expressions with correct precedence: `+ - * /`, parentheses,
int/float literals, and variables.

```
1 + 2 * 3        → 7
(a * b) + (c * d) / (a - b)
```

## Quickstart

```python
from toyc import Cpu, GpuVulkan

Cpu.run("program.toy")          # 15        (CPU tree-walk / bytecode VM)
GpuVulkan.run("program.toy")    # 15.0      (Vulkan compute shader)
```

## How it works (pipeline)

```
source text ──► Lexer ──► Parser ──► AST ──┬──► Evaluator / Cpu ──► CPU result
                                            │
                                            └──► Flattener ──► flat bytecode
                                                  [op, dest, src1, src2] ──► GPU ──► GPU result
```

| Stage | Module | What it does |
|-------|--------|--------------|
| Lexer | `toyc/lexer.py` | Tokenizes numbers, identifiers and `+ - * / ( )` |
| Parser | `toyc/parser.py` | Recursive-descent parser, correct precedence, builds the AST (`Number`, `Var`, `BinOp`) |
| Evaluator | `toyc/evaluator.py` | Tree-walking interpreter, the CPU reference |
| Compiler | `toyc/compiler.py` | Emits stack bytecode (`PUSH`/`LOAD`/`ADD`/`SUB`/`MUL`/`DIV`), saves/loads `.toyc` files |
| Cpu | `toyc/vm.py` | Stack VM executing the bytecode |
| Flattener | `toyc/gpu/flattener.py` | Lowers the AST to register bytecode + constant/variable pools |
| ISA | `toyc/gpu/instructions.py` | Opcodes `ADD=0 SUB=1 MUL=2 DIV=3 LOAD=4 VAR=5` — must match the shaders |
| Shaders | `toyc/gpu/vulkan/comp.glsl` / `comp_batch.glsl` | Single-value and data-parallel batch interpreters, compiled to `.spv` on first use |
| Vulkan backend | `toyc/gpu/vulkan/backend.py` | Instance/device/pipeline management, dispatch, readback |
| GPU detection | `toyc/gpu/vulkan/gpu_detect.py` | Device profiles, VRAM-based batch recommendation, JSON cache |

## CPU backend

```python
from toyc import Cpu, Compiler, Lexer, Parser

Cpu.run("program.toy")                          # lex+parse+compile+run, prints 15
Cpu.run("program.toy", env={"a": 1})            # variable bindings
Cpu.run("program.toy", silent=True)             # capture without printing
Cpu.run("out.toyc")                             # run a compiled file
Compiler.compile("program.toy")                 # write program.toyc to disk

tokens = Lexer.tokenize("a + b * 2")
ast    = Parser.parse(tokens)
```

`Cpu.run` accepts a `.toy` path, a `.toyc` path, or an in-memory
`list[Instr]`.

## GPU backend (single value)

```python
from toyc import GpuVulkan

GpuVulkan.run("program.toy")                    # 15.0
GpuVulkan.run("program.toy", env={"a": 10.0, "b": 5.0})
GpuVulkan.run("program.toy", silent=True)       # capture without printing
GpuVulkan.run("program.toyc")                   # cached 4-byte result, no dispatch
GpuVulkan.compile("program.toy")                # evaluate + cache program.toyc
```

How a single run works: the AST is flattened to
`[n_instrs, n_consts, n_vars, op, dest, src1, src2, …]` plus a constant
pool, your `env` values, and an output + error slot. That buffer is
uploaded to one storage buffer, **one** compute workgroup runs the
`comp` shader (a 256-register interpreter loop), and the output float
is mapped back. The `.toyc` file written next to the source holds the
cached 4-byte result, so `run("program.toyc")` decodes it without
touching the GPU.

### Session lifetime

Creating the Vulkan instance + device + pipelines costs ~15–40 ms;
a dispatch costs ~1 ms. So the backend keeps **one shared session per
process** instead of rebuilding it per call:

```python
GpuVulkan.startup()    # build the session now (optional; first call does it lazily)
GpuVulkan.run(...)     # reuses the live session — never tears it down
GpuVulkan.shutdown()   # release it (also happens automatically at process exit)
```

The session is guarded by a lock, so concurrent `run()` calls from
threads share it safely. Recompiling the shaders transparently rebuilds
the pipelines (the cache key includes both `.spv` mtimes).

## GPU backend (batch — where the GPU wins)

`run_batch` evaluates **one expression for many variable sets with one
dispatch per chunk** — one shader invocation per instance
(`local_size_x = 64`, `comp_batch` shader):

```python
envs = [{"a": 1.0, "b": 2.0}, {"a": 3.0, "b": 4.0}, ...]  # N sets
results = GpuVulkan.run_batch("prog.toy", envs)          # [3.0, 7.0, ...]
```

- The AST is flattened **once**; the variable pool holds all N sets
  (instance-major) and the shader writes N outputs. Bigger `envs`
  lists are split into chunks of the **recommended batch size**
  (`gpu_detect.batch`, e.g. 1,638 on an 8 GB card) with a VRAM-budget
  safety cap — never silently wrong.
- Returns one float per instance, in input order.
- Same error contract as everywhere: `NameError` for a variable
  missing from any set, `ZeroDivisionError` if **any** instance
  divides by zero (signalled through a shared error slot, since a
  shader cannot throw).

Rule of thumb (RX 580): single expressions → CPU (~0.1 ms vs ~1 ms
GPU dispatch + overhead); bulk evaluation → `run_batch`, where the
fixed per-call overhead amortises and throughput wins (~10× at N=500
and growing).

## Timing, debug output, and benchmarking

Every entry point takes `timed=False` (print a report + return timing
data as a second value) and the GPU calls also take `debug=False`
(print device/VRAM diagnostics):

```python
cpu_val, cpu_t = Cpu.run("program.toy", timed=True)
# [CPU timing] total=0.105 ms (resolve=0.099 ms, execute=0.006 ms)

gpu_raw, gpu_t = GpuVulkan.run("program.toy", timed=True)
# [GPU timing] total=0.843 ms (flatten=0.069 ms, select=0.010 ms,
#              init=0.027 ms, exec=0.666 ms, teardown=0.000 ms, decode=0.000 ms)
```

Timing dicts hold **seconds** (`perf_counter`); stages that didn't run
are `0.0`. Untimed calls return exactly what they always did, so
existing code is unaffected.

### `toyc.bench` — table-ready profiling for pandas

Dependency-free (stdlib only). Every call returns plain row-dicts —
one per run — that drop straight into a DataFrame:

```python
from toyc.bench import profile, compare, compare_batch, to_csv

a = 10
b = 20
rows = compare(["1 + 2 * 7", "a + b * 2"], repeats=5)
# env auto-collected from your variables — no envs= needed

import pandas as pd
df = pd.DataFrame(rows)
df.groupby("backend")["total"].mean().plot.bar()   # CPU vs GPU
```

| Function | What it does | Row columns |
|----------|--------------|-------------|
| `profile(backend, program, env?, repeats?, …)` | Repeat one program on `"cpu"` or `"gpu"` | `backend, program, repeat, result, total` + stage columns |
| `compare(programs, envs?, repeats?, …)` | Each program on **both** backends | same as above, concatenated |
| `compare_batch(program, env_sets, repeats?, …)` | One program over N sets: sequential CPU loop (`"cpu"`) vs `run_batch` (`"gpu-batch"`) | `backend, program, n, repeat, total, per_eval, max_err` + stage columns |
| `to_csv(rows, path)` | Write rows to CSV (`pd.read_csv`-ready) | fixed column order |

- `env_sets` accepts a list of env dicts **or** a column dict of
  equal-length value lists: `{"a": [1, 3], "b": [2, 4]}`.
- `max_err` is the max abs GPU-vs-CPU difference in that repeat
  (`0.0` on CPU rows) — the table proves the GPU results, so vectors
  aren't stored per row.
- Single-value `result` is the decoded float; batch vectors collapse
  to `max_err` (see above).
- The GPU session is warmed up before repeats, per-run prints are
  suppressed unless `verbose=True`, and temp `.toy` files are cleaned
  up automatically.

### Plotting later (matplotlib — yours to add)

The library ships no plotters on purpose. When you're ready, the hook
is the row table — seconds in `total`/`per_eval`, one row per repeat:

```python
import matplotlib.pyplot as plt
import pandas as pd
from toyc.bench import compare_batch

rows = compare_batch("a + b * 2",
                     {"a": list(range(2000)), "b": list(range(2000))},
                     repeats=3)
df = pd.DataFrame(rows)
df["ms"] = df["total"] * 1e3
df.groupby("backend")["ms"].mean().plot.bar()   # cpu vs gpu-batch
plt.ylabel("ms")
plt.show()
```

Useful columns to plot: `total` vs `n` (scaling curve), `per_eval`
(cost per instance), `exec` (pure dispatch time), `max_err` (validity).
`to_csv(rows, "timings.csv")` saves anything for later sessions.

## Semantics (CPU ≡ GPU, guaranteed)

| Case | Both backends |
|------|---------------|
| Value | Numerically equal; CPU keeps `int` for int-only expressions (`15`), GPU always returns float32-based `float` (`15.0`) |
| Division by zero (`1/0`, `1/(a-a)`, `0/0`) | `ZeroDivisionError("Division by zero in VM")` |
| Missing variable | `NameError("Undefined variable: 'q'")` |
| Float precision | GPU computes in float32 — expect ~1e-6 relative error vs CPU float64 on large magnitudes |

## License

See [LICENSE](toyc/LICENSE).
