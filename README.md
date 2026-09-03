# toyc

A toy compiler with Vulkan and OpenGL GPU backends, written in Python.

## Install

```bash
pip install toyc
```

## What's inside

- **Lexer / Parser / AST** — tokenizes and parses a simple toy language
- **Evaluator** — tree-walk interpreter
- **VM** — bytecode virtual machine

## To do 
- **GPU backends** — Vulkan and OpenGL compute backends via `toyc.gpu`

## Usage

```python
from  toyc  import Parser , Lexer , Flattener , vm
from toyc import Compiler ,Cpu 
# example code
code = "1 + 2 * 3"

# tokonizer of the code
token = Lexer.tokenize(code)
print(token ,"\n")

# prashing building an ats tree 
ats = Parser.parse(token)
print (ats,"\n")

Cpu.run(program="program.toy")

```

# Gpu detection (not compliling yet, andno opengl vulkan only)
```python
from toyc import gpu.GpuValkan

db = gpu.Gpuvalkan
```
ps: you can call it it wont print anything yet, iam still wokrcing on the class and its function

## License

See [LICENSE](toyc/LICENSE).