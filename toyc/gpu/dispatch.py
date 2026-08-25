# compiler/gpu/dispatch.py

import subprocess
import os
import time
import array
import threading
import vulkan as vk
import cffi

SHADER_PATH = os.path.join(os.path.dirname(__file__), "shaders", "executor.comp")
SPV_PATH    = SHADER_PATH + ".spv"

_spv_cache = None
_spv_lock  = threading.Lock()


def compile_shader():
    """Compile GLSL → SPIR-V using glslc.

    Cached in memory and on disk: glslc only runs when the cached .spv is
    missing or older than the .glsl source.
    """
    global _spv_cache
    with _spv_lock:
        if _spv_cache is not None:
            return _spv_cache

        up_to_date = (
            os.path.exists(SPV_PATH)
            and os.path.getmtime(SPV_PATH) >= os.path.getmtime(SHADER_PATH)
        )
        if not up_to_date:
            result = subprocess.run(
                ["glslc", SHADER_PATH, "-o", SPV_PATH],
                capture_output=True, text=True
            )
            if result.returncode != 0:
                raise RuntimeError(f"glslc failed:\n{result.stderr}")

        with open(SPV_PATH, "rb") as f:
            _spv_cache = f.read()
        return _spv_cache


def find_memory_type(physical_device, type_filter, properties):
    """Find a suitable memory type on the GPU."""
    mem_props = vk.vkGetPhysicalDeviceMemoryProperties(physical_device)
    for i in range(mem_props.memoryTypeCount):
        if (type_filter & (1 << i)) and \
           (mem_props.memoryTypes[i].propertyFlags & properties) == properties:
            return i
    raise RuntimeError("No suitable memory type found")


def make_buffer(device, physical_device, data_list, dtype="int"):
    fmt  = "i" if dtype == "int" else "f"
    raw  = array.array(fmt, data_list)
    size = raw.buffer_info()[1] * raw.itemsize

    buf_info = vk.VkBufferCreateInfo(
        size        = size,
        usage       = vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        sharingMode = vk.VK_SHARING_MODE_EXCLUSIVE,
    )
    buf = vk.vkCreateBuffer(device, buf_info, None)

    mem_reqs = vk.vkGetBufferMemoryRequirements(device, buf)
    alloc    = vk.VkMemoryAllocateInfo(
        allocationSize  = mem_reqs.size,
        memoryTypeIndex = find_memory_type(
            physical_device,
            mem_reqs.memoryTypeBits,
            vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
            vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
        )
    )
    mem = vk.vkAllocateMemory(device, alloc, None)
    vk.vkBindBufferMemory(device, buf, mem, 0)

    # write via CFFI buffer directly
    ptr        = vk.vkMapMemory(device, mem, 0, size, 0)
    data_bytes = raw.tobytes()
    ffi        = cffi.FFI()
    ffi.memmove(ptr, data_bytes, size)
    vk.vkUnmapMemory(device, mem)

    return buf, mem, size


class GPUExecutor:
    def __init__(self):
        self.last_timings = {}
        self._init_vulkan()

    def _init_vulkan(self):
        """Create Vulkan instance, pick physical device, create logical device."""

        app_info  = vk.VkApplicationInfo(
            pApplicationName   = "ToyCompiler",
            applicationVersion = vk.VK_MAKE_VERSION(0, 1, 0),
            apiVersion         = vk.VK_API_VERSION_1_0,
        )
        inst_info     = vk.VkInstanceCreateInfo(pApplicationInfo=app_info)
        self.instance = vk.vkCreateInstance(inst_info, None)

        self.physical_device = vk.vkEnumeratePhysicalDevices(self.instance)[0]
        props = vk.vkGetPhysicalDeviceProperties(self.physical_device)
        print(f"GPU: {props.deviceName}")

        queue_families    = vk.vkGetPhysicalDeviceQueueFamilyProperties(
            self.physical_device
        )
        self.queue_family = next(
            i for i, qf in enumerate(queue_families)
            if qf.queueFlags & vk.VK_QUEUE_COMPUTE_BIT
        )

        queue_info  = vk.VkDeviceQueueCreateInfo(
            queueFamilyIndex = self.queue_family,
            queueCount       = 1,
            pQueuePriorities = [1.0],
        )
        device_info = vk.VkDeviceCreateInfo(
            pQueueCreateInfos = [queue_info],
        )
        self.device = vk.vkCreateDevice(self.physical_device, device_info, None)
        self.queue  = vk.vkGetDeviceQueue(self.device, self.queue_family, 0)

    def run(self, flat_instructions, constants, variables):
        """
        Upload instruction list + data to VRAM, dispatch shader, return result.

        flat_instructions : list of ints   e.g [5,0,0,0, 5,1,1,0, ...]
        constants         : list of floats e.g [2.0]
        variables         : list of floats e.g [10.0, 5.0]

        Per-call stage timings (ms) are stored in self.last_timings with keys:
        compile, buffers, setup, dispatch, readback.
        """
        t = {}

        # ── compile shader (cached) ────────────────────────────────────
        t0 = time.perf_counter()
        spv = compile_shader()
        t["compile"] = (time.perf_counter() - t0) * 1000.0

        # ── create buffers + track their sizes ────────────────────────
        t0 = time.perf_counter()
        instr_buf, instr_mem, instr_size = make_buffer(
            self.device, self.physical_device, flat_instructions, "int"
        )
        const_buf, const_mem, const_size = make_buffer(
            self.device, self.physical_device, constants or [0.0], "float"
        )
        var_buf,   var_mem,   var_size   = make_buffer(
            self.device, self.physical_device, variables, "float"
        )
        res_buf,   res_mem,   res_size   = make_buffer(
            self.device, self.physical_device, [0.0], "float"
        )
        t["buffers"] = (time.perf_counter() - t0) * 1000.0

        # ── descriptor set layout ─────────────────────────────────────
        t0 = time.perf_counter()
        bindings = [
            vk.VkDescriptorSetLayoutBinding(
                binding         = i,
                descriptorType  = vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                descriptorCount = 1,
                stageFlags      = vk.VK_SHADER_STAGE_COMPUTE_BIT,
            )
            for i in range(4)
        ]
        layout_info     = vk.VkDescriptorSetLayoutCreateInfo(pBindings=bindings)
        desc_set_layout = vk.vkCreateDescriptorSetLayout(
            self.device, layout_info, None
        )

        # ── pipeline layout ───────────────────────────────────────────
        pipeline_layout = vk.vkCreatePipelineLayout(
            self.device,
            vk.VkPipelineLayoutCreateInfo(pSetLayouts=[desc_set_layout]),
            None
        )

        # ── shader module ─────────────────────────────────────────────
        shader_module = vk.vkCreateShaderModule(
            self.device,
            vk.VkShaderModuleCreateInfo(codeSize=len(spv), pCode=spv),
            None
        )

        # ── compute pipeline ──────────────────────────────────────────
        stage = vk.VkPipelineShaderStageCreateInfo(
            stage  = vk.VK_SHADER_STAGE_COMPUTE_BIT,
            module = shader_module,
            pName  = "main",
        )
        pipeline = vk.vkCreateComputePipelines(
            self.device,
            vk.VK_NULL_HANDLE,
            1,
            [vk.VkComputePipelineCreateInfo(stage=stage, layout=pipeline_layout)],
            None
        )[0]

        # ── descriptor pool + set ─────────────────────────────────────
        desc_pool = vk.vkCreateDescriptorPool(
            self.device,
            vk.VkDescriptorPoolCreateInfo(
                maxSets    = 1,
                pPoolSizes = [vk.VkDescriptorPoolSize(
                    type            = vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    descriptorCount = 4,
                )],
            ),
            None
        )
        desc_set = vk.vkAllocateDescriptorSets(
            self.device,
            vk.VkDescriptorSetAllocateInfo(
                descriptorPool = desc_pool,
                pSetLayouts    = [desc_set_layout],
            )
        )[0]

        # ── bind buffers — use real sizes, NOT VK_WHOLE_SIZE ──────────
        buffers = [instr_buf,  const_buf,  var_buf,  res_buf]
        sizes   = [instr_size, const_size, var_size, res_size]

        writes = [
            vk.VkWriteDescriptorSet(
                dstSet          = desc_set,
                dstBinding      = i,
                descriptorType  = vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                descriptorCount = 1,
                pBufferInfo     = [vk.VkDescriptorBufferInfo(
                    buffer = buffers[i],
                    offset = 0,
                    range  = sizes[i],
                )],
            )
            for i in range(4)
        ]
        vk.vkUpdateDescriptorSets(self.device, len(writes), writes, 0, None)

        # ── command buffer ────────────────────────────────────────────
        cmd_pool = vk.vkCreateCommandPool(
            self.device,
            vk.VkCommandPoolCreateInfo(queueFamilyIndex=self.queue_family),
            None
        )
        cmd_buf = vk.vkAllocateCommandBuffers(
            self.device,
            vk.VkCommandBufferAllocateInfo(
                commandPool        = cmd_pool,
                level              = vk.VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                commandBufferCount = 1,
            )
        )[0]

        # record
        vk.vkBeginCommandBuffer(cmd_buf, vk.VkCommandBufferBeginInfo())
        vk.vkCmdBindPipeline(
            cmd_buf, vk.VK_PIPELINE_BIND_POINT_COMPUTE, pipeline
        )
        vk.vkCmdBindDescriptorSets(
            cmd_buf,
            vk.VK_PIPELINE_BIND_POINT_COMPUTE,
            pipeline_layout,
            0,
            1,
            [desc_set],
            0,
            None
        )
        vk.vkCmdDispatch(cmd_buf, 1, 1, 1)
        vk.vkEndCommandBuffer(cmd_buf)
        t["setup"] = (time.perf_counter() - t0) * 1000.0

        # ── submit + wait ─────────────────────────────────────────────
        t0 = time.perf_counter()
        fence = vk.vkCreateFence(self.device, vk.VkFenceCreateInfo(), None)
        vk.vkQueueSubmit(
            self.queue,
            1,
            [vk.VkSubmitInfo(pCommandBuffers=[cmd_buf])],
            fence
        )
        vk.vkWaitForFences(
            self.device,
            1,
            [fence],
            True,
            int(1e9)
        )
        t["dispatch"] = (time.perf_counter() - t0) * 1000.0

        # ── read back result ──────────────────────────────────────────
        t0 = time.perf_counter()
        ptr        = vk.vkMapMemory(self.device, res_mem, 0, res_size, 0)
        data_bytes = bytes(ptr)
        out        = array.array("f", data_bytes)
        vk.vkUnmapMemory(self.device, res_mem)
        t["readback"] = (time.perf_counter() - t0) * 1000.0

        self.last_timings = t
        return out[0]