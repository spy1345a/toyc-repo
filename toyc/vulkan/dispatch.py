# compiler/gpu/dispatch.py

import subprocess
import os
import time
import array
import struct
import threading
import vulkan as vk
import cffi

SHADER_PATH = os.path.join(os.path.dirname(__file__), "shaders", "executor.comp")
SPV_PATH    = SHADER_PATH + ".spv"

_spv_cache = None
_spv_lock  = threading.Lock()

_vk_api_lock = threading.Lock()


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


def make_buffer(device, physical_device, data_list, dtype="int", usage=None):
    fmt  = "i" if dtype == "int" else "f"
    raw  = array.array(fmt, data_list)
    size = raw.buffer_info()[1] * raw.itemsize

    if usage is None:
        usage = vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT

    buf_info = vk.VkBufferCreateInfo(
        size        = size,
        usage       = usage,
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
        self.timestamp_period = props.limits.timestampPeriod  # ns per tick

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

        # timestamp query pool (2 queries: before + after the dispatch)
        self.query_pool = vk.vkCreateQueryPool(
            self.device,
            vk.VkQueryPoolCreateInfo(
                queryType = vk.VK_QUERY_TYPE_TIMESTAMP,
                queryCount = 2,
            ),
            None
        )

        # host-visible buffer the query results get copied into (2 x uint64)
        self.ts_buf, self.ts_mem, self.ts_size = make_buffer(
            self.device, self.physical_device, [0, 0, 0, 0], "int",
            usage=vk.VK_BUFFER_USAGE_TRANSFER_DST_BIT
        )

    def run(self, flat_instructions, constants, variables):
        """Evaluate a single expression instance. Returns one float."""
        return self._execute(flat_instructions, constants, variables, 1,
                             len(variables))[0]

    def run_batch(self, flat_instructions, constants, variables, count,
                  num_vars):
        """
        Evaluate `count` instances of the same program data-parallel.

        variables : flat float list of len(count * num_vars), row-major
                    [inst0_var0, inst0_var1, ..., inst1_var0, ...]
        num_vars  : variables per instance (number of unique vars)
        Returns a list of `count` floats.
        """
        return self._execute(flat_instructions, constants, variables, count,
                             num_vars)

    LOCAL_SIZE = 64

    def _execute(self, flat_instructions, constants, variables, count,
                 num_vars):
        """
        Upload instruction list + data to VRAM, dispatch shader, return results.

        flat_instructions : list of ints   e.g [5,0,0,0, 5,1,1,0, ...]
        constants         : list of floats e.g [2.0]
        variables         : flat float list (count * num_vars)
        count             : number of instances to evaluate
        num_vars          : variables per instance

        Per-call stage timings (ms) are stored in self.last_timings with keys:
        compile, buffers, setup, dispatch, readback, gpu_exec.
        """
        t = {}
        var_data  = variables if variables else [0.0]
        const_data = list(constants or []) + [float(num_vars), float(count)]

        # ── compile shader (cached) ────────────────────────────────────
        t0 = time.perf_counter()
        spv = compile_shader()
        t["compile"] = (time.perf_counter() - t0) * 1000.0

        # ── create buffers + track their sizes ────────────────────────
        t0 = time.perf_counter()
        with _vk_api_lock:
            instr_buf, instr_mem, instr_size = make_buffer(
                self.device, self.physical_device, flat_instructions, "int"
            )
            const_buf, const_mem, const_size = make_buffer(
                self.device, self.physical_device, const_data, "float"
            )
            var_buf,   var_mem,   var_size   = make_buffer(
                self.device, self.physical_device, var_data, "float"
            )
            res_buf,   res_mem,   res_size   = make_buffer(
                self.device, self.physical_device, [0.0] * count, "float"
            )
        t["buffers"] = (time.perf_counter() - t0) * 1000.0

        # ── descriptor set layout ─────────────────────────────────────
        t0 = time.perf_counter()
        with _vk_api_lock:
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

            # ── pipeline layout ───────────────────────────────────────
            pipeline_layout = vk.vkCreatePipelineLayout(
                self.device,
                vk.VkPipelineLayoutCreateInfo(pSetLayouts=[desc_set_layout]),
                None
            )

            # ── shader module ─────────────────────────────────────────
            shader_module = vk.vkCreateShaderModule(
                self.device,
                vk.VkShaderModuleCreateInfo(codeSize=len(spv), pCode=spv),
                None
            )

            # ── compute pipeline ──────────────────────────────────────
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

            # ── descriptor pool + set ─────────────────────────────────
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

            # ── bind buffers — use real sizes, NOT VK_WHOLE_SIZE ──────
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

            # ── command buffer ────────────────────────────────────────
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
            vk.vkCmdResetQueryPool(cmd_buf, self.query_pool, 0, 2)
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
            vk.vkCmdWriteTimestamp(
                cmd_buf,
                vk.VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                self.query_pool,
                0
            )
            vk.vkCmdDispatch(
                cmd_buf,
                (count + GPUExecutor.LOCAL_SIZE - 1) // GPUExecutor.LOCAL_SIZE,
                1, 1
            )
            vk.vkCmdWriteTimestamp(
                cmd_buf,
                vk.VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                self.query_pool,
                1
            )
            vk.vkCmdCopyQueryPoolResults(
                cmd_buf,
                self.query_pool,
                0,
                2,
                self.ts_buf,
                0,
                8,
                vk.VK_QUERY_RESULT_64_BIT | vk.VK_QUERY_RESULT_WAIT_BIT
            )
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
        with _vk_api_lock:
            ptr        = vk.vkMapMemory(self.device, res_mem, 0, res_size, 0)
            data_bytes = bytes(ptr)
            out        = array.array("f", data_bytes)
            vk.vkUnmapMemory(self.device, res_mem)

            ts_ptr     = vk.vkMapMemory(self.device, self.ts_mem, 0, self.ts_size, 0)
            ts_bytes   = bytes(ts_ptr)
            vk.vkUnmapMemory(self.device, self.ts_mem)
        ts_start, ts_end = struct.unpack("QQ", ts_bytes)
        t["readback"]  = (time.perf_counter() - t0) * 1000.0
        t["gpu_exec"]  = (ts_end - ts_start) * self.timestamp_period / 1e6

        # ── free per-call resources (they are rebuilt on every run) ───
        with _vk_api_lock:
            vk.vkDestroyFence(self.device, fence, None)
            vk.vkDestroyCommandPool(self.device, cmd_pool, None)
            vk.vkDestroyDescriptorPool(self.device, desc_pool, None)
            vk.vkDestroyDescriptorSetLayout(self.device, desc_set_layout,
                                            None)
            vk.vkDestroyPipeline(self.device, pipeline, None)
            vk.vkDestroyPipelineLayout(self.device, pipeline_layout, None)
            vk.vkDestroyShaderModule(self.device, shader_module, None)
            for buf, mem in ((instr_buf, instr_mem), (const_buf, const_mem),
                             (var_buf, var_mem), (res_buf, res_mem)):
                vk.vkDestroyBuffer(self.device, buf, None)
                vk.vkFreeMemory(self.device, mem, None)

        self.last_timings = t
        return list(out)