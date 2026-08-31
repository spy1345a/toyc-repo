# Instruction → SPIR-V opcode mapping (matches compiler.py comments):
#   PUSH  → OpConstant
#   LOAD  → OpLoad
#   ADD   → OpFAdd
#   SUB   → OpFSub
#   MUL   → OpFMul
#   DIV   → OpFDiv

#__________self created imports____________
from gpu_detect import detect, save, load, select_device, suggest_batch_size

#__________________________________________

import vulkan as vk

# First run — detect and cache
db = detect(verbose=True)
save(db)

# Later runs — load from cache
db = load()

# Pick the best device and calculate batch size
dev = select_device(db, prefer="discrete")
batch = suggest_batch_size(dev, element_bytes=4 * 1024 * 1024)  # 4 MB per item
print(batch)

class GpuVulkan:
    pass