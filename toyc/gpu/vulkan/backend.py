# Instruction → SPIR-V opcode mapping (matches compiler.py comments):
#   PUSH  → OpConstant
#   LOAD  → OpLoad
#   ADD   → OpFAdd
#   SUB   → OpFSub
#   MUL   → OpFMul
#   DIV   → OpFDiv

#__________self created imports____________
from . import gpu_detect

#__________________________________________

#___________dependency imports ____________
import vulkan as vk
from .. import flattener

#___________________________________________

class GpuVulkan:

    @staticmethod
    def _sel_gpu():
        # detect all Vulkan-capable GPUs on this machine
        gpu_db = gpu_detect.detect()

        if not gpu_db:
            raise RuntimeError(
                "No Vulkan-capable GPU found. "
                "Ensure your drivers are installed and Vulkan is supported."
            )

        # prefer discrete GPU, falls back to whatever has the most VRAM
        best_gpu = gpu_detect.select_device(gpu_db)

        if best_gpu is None:
            raise RuntimeError(
                "GPU detection succeeded but device selection failed. "
                "This should not happen — please file a bug."
            )

        # recommended batch size = 80% of device-local VRAM
        batch_size = gpu_detect.batch(best_gpu)

        return best_gpu, batch_size
