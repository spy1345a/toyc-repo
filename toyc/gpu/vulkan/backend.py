# Instruction → SPIR-V opcode mapping (matches compiler.py comments):
#   PUSH  → OpConstant
#   LOAD  → OpLoad
#   ADD   → OpFAdd
#   SUB   → OpFSub
#   MUL   → OpFMul
#   DIV   → OpFDiv

#__________self created imports____________
import gpu_detect

#__________________________________________

import vulkan as vk

# detect all GPUs (auto-saves JSON next to script)
db = gpu_detect.detect()

# pick the best GPU
best = gpu_detect.select_device(db)

# get batch size — prints + returns
n = gpu_detect.batch(best)

# n is now just an int you can use
print(f"will process {n} elements per batch")

class GpuVulkan:
    pass