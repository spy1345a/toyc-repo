import os
import struct
from typing import Any

import vulkan as vk

from . import gpu_detect
from .. import flattener
from ...compiler import (
    Instr,
    Compiler,
    write_bytecode,
    read_bytecode,
    is_compiled_bytecode,
)
from ...lexer  import Lexer
from ...parser import Parser


class GpuVulkan:

    # ── device selection ──────────────────────────────────────────────────────

    @staticmethod
    def _sel_gpu():
        """Return (DeviceProfile, batch_size) for the best available GPU."""
        gpu_db = gpu_detect.detect()

        if not gpu_db:
            raise RuntimeError(
                "No Vulkan-capable GPU found. "
                "Ensure your drivers are installed and Vulkan is supported."
            )

        profile = gpu_detect.select_device(gpu_db)

        if profile is None:
            raise RuntimeError(
                "GPU detection succeeded but device selection failed. "
                "This should not happen — please file a bug."
            )

        batch_size = gpu_detect.batch(profile)

        return profile, batch_size

    # ── SPIR-V auto-compile ───────────────────────────────────────────────────

    @staticmethod
    def _ensure_spv() -> str:
        """
        Return the path to comp.spv, compiling it from comp.glsl first if
        the .spv is missing or older than the .glsl source.

        Tries these compilers in order:
            1. glslangValidator  (glslang-tools package / Vulkan SDK)
            2. glslc             (shaderc / Vulkan SDK)

        Raises RuntimeError if neither is found.
        """
        import subprocess
        import shutil

        here     = os.path.dirname(os.path.abspath(__file__))
        glsl_path = os.path.join(here, "comp.glsl")
        spv_path  = os.path.join(here, "comp.spv")

        # skip recompile if .spv exists and is newer than .glsl
        if (os.path.isfile(spv_path) and os.path.isfile(glsl_path) and
                os.path.getmtime(spv_path) >= os.path.getmtime(glsl_path)):
            return spv_path

        if not os.path.isfile(glsl_path):
            raise FileNotFoundError(
                f"Shader source not found: {glsl_path!r}\n"
                "Place comp.glsl next to backend.py."
            )

        # find a compiler
        compiler = shutil.which("glslangValidator") or shutil.which("glslc")
        if compiler is None:
            raise RuntimeError(
                "No GLSL → SPIR-V compiler found.\n"
                "Install one with:\n"
                "  sudo apt install glslang-tools    # glslangValidator\n"
                "  sudo apt install glslc            # glslc (shaderc)"
            )

        if os.path.basename(compiler) == "glslangValidator":
            cmd = [compiler, "-V", glsl_path, "-o", spv_path]
        else:
            cmd = [compiler, "-fshader-stage=compute", glsl_path, "-o", spv_path]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"Shader compilation failed:\n{result.stderr or result.stdout}"
            )

        return spv_path

    # ── Vulkan logical device + pipeline init ─────────────────────────────────

    @staticmethod
    def _init_vulkan(profile):
        """
        Create a Vulkan instance, logical device, compute pipeline, command
        pool, and queue from a DeviceProfile.

        Returns a plain namespace with:
            .instance, .physical_device, .device,
            .queue, .cmd_pool,
            .descriptor_set_layout, .pipeline_layout, .pipeline
        """
        # instance
        app_info = vk.VkApplicationInfo(
            pApplicationName="toyc",
            applicationVersion=vk.VK_MAKE_VERSION(1, 0, 0),
            pEngineName="toyc",
            engineVersion=vk.VK_MAKE_VERSION(1, 0, 0),
            apiVersion=vk.VK_API_VERSION_1_0,
        )
        instance = vk.vkCreateInstance(
            vk.VkInstanceCreateInfo(pApplicationInfo=app_info), None
        )

        # pick the physical device that matches profile.index
        phys_devs      = vk.vkEnumeratePhysicalDevices(instance)
        physical_device = phys_devs[profile.index]

        # find a compute queue family
        compute_qf = next(
            (qf for qf in profile.queue_families if "COMPUTE" in qf.flags),
            None,
        )
        if compute_qf is None:
            raise RuntimeError(f"No compute queue family on {profile.name!r}")
        qf_index = compute_qf.index

        # logical device
        queue_info = vk.VkDeviceQueueCreateInfo(
            queueFamilyIndex=qf_index,
            queueCount=1,
            pQueuePriorities=[1.0],
        )
        device = vk.vkCreateDevice(
            physical_device,
            vk.VkDeviceCreateInfo(
                queueCreateInfoCount=1,
                pQueueCreateInfos=[queue_info],
            ),
            None,
        )
        queue    = vk.vkGetDeviceQueue(device, qf_index, 0)
        cmd_pool = vk.vkCreateCommandPool(
            device,
            vk.VkCommandPoolCreateInfo(
                flags=vk.VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
                queueFamilyIndex=qf_index,
            ),
            None,
        )

        # descriptor set layout: one storage buffer binding
        binding = vk.VkDescriptorSetLayoutBinding(
            binding=0,
            descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            descriptorCount=1,
            stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT,
        )
        ds_layout = vk.vkCreateDescriptorSetLayout(
            device,
            vk.VkDescriptorSetLayoutCreateInfo(bindingCount=1, pBindings=[binding]),
            None,
        )
        pipeline_layout = vk.vkCreatePipelineLayout(
            device,
            vk.VkPipelineLayoutCreateInfo(
                setLayoutCount=1, pSetLayouts=[ds_layout]
            ),
            None,
        )

        # auto-compile comp.glsl → comp.spv if needed, then load
        spv_path = GpuVulkan._ensure_spv()
        with open(spv_path, "rb") as f:
            spv = f.read()

        shader_module = vk.vkCreateShaderModule(
            device,
            vk.VkShaderModuleCreateInfo(codeSize=len(spv), pCode=spv),
            None,
        )
        pipeline = vk.vkCreateComputePipelines(
            device, None, 1,
            [vk.VkComputePipelineCreateInfo(
                stage=vk.VkPipelineShaderStageCreateInfo(
                    stage=vk.VK_SHADER_STAGE_COMPUTE_BIT,
                    module=shader_module,
                    pName="main",
                ),
                layout=pipeline_layout,
            )],
            None,
        )[0]
        vk.vkDestroyShaderModule(device, shader_module, None)

        class _Vk:
            pass

        ctx                    = _Vk()
        ctx.instance           = instance
        ctx.physical_device    = physical_device
        ctx.device             = device
        ctx.queue              = queue
        ctx.cmd_pool           = cmd_pool
        ctx.ds_layout          = ds_layout
        ctx.pipeline_layout    = pipeline_layout
        ctx.pipeline           = pipeline
        ctx.qf_index           = qf_index
        return ctx

    @staticmethod
    def _destroy_vulkan(ctx) -> None:
        """Clean up all Vulkan handles created by _init_vulkan."""
        vk.vkDestroyPipeline(ctx.device, ctx.pipeline, None)
        vk.vkDestroyPipelineLayout(ctx.device, ctx.pipeline_layout, None)
        vk.vkDestroyDescriptorSetLayout(ctx.device, ctx.ds_layout, None)
        vk.vkDestroyCommandPool(ctx.device, ctx.cmd_pool, None)
        vk.vkDestroyDevice(ctx.device, None)
        vk.vkDestroyInstance(ctx.instance, None)

    # ── Vulkan buffer helpers ─────────────────────────────────────────────────

    @staticmethod
    def _create_buffer(ctx, size, usage, mem_props_flags):
        buf = vk.vkCreateBuffer(
            ctx.device,
            vk.VkBufferCreateInfo(
                size=size,
                usage=usage,
                sharingMode=vk.VK_SHARING_MODE_EXCLUSIVE,
            ),
            None,
        )

        mem_reqs = vk.vkGetBufferMemoryRequirements(ctx.device, buf)
        phys_mem = vk.vkGetPhysicalDeviceMemoryProperties(ctx.physical_device)

        mem_type_index = None
        for i in range(phys_mem.memoryTypeCount):
            if (mem_reqs.memoryTypeBits & (1 << i)) and \
               (phys_mem.memoryTypes[i].propertyFlags & mem_props_flags) == mem_props_flags:
                mem_type_index = i
                break

        if mem_type_index is None:
            raise RuntimeError("No suitable Vulkan memory type found.")

        memory = vk.vkAllocateMemory(
            ctx.device,
            vk.VkMemoryAllocateInfo(
                allocationSize=mem_reqs.size,
                memoryTypeIndex=mem_type_index,
            ),
            None,
        )
        vk.vkBindBufferMemory(ctx.device, buf, memory, 0)
        return buf, memory

    @staticmethod
    def _upload_flat(ctx, flat_data: list[int]):
        packed = struct.pack(f"{len(flat_data)}i", *flat_data)
        size   = len(packed)

        buf, mem = GpuVulkan._create_buffer(
            ctx, size,
            vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
            vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        )

        ptr = vk.vkMapMemory(ctx.device, mem, 0, size, 0)
        vk.memmove(ptr, packed, size)
        vk.vkUnmapMemory(ctx.device, mem)

        return buf, mem

    @staticmethod
    def _run_compute(ctx, buf, size, x_groups):
        # descriptor pool + set
        pool = vk.vkCreateDescriptorPool(
            ctx.device,
            vk.VkDescriptorPoolCreateInfo(
                maxSets=1,
                poolSizeCount=1,
                pPoolSizes=[vk.VkDescriptorPoolSize(
                    type=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    descriptorCount=1,
                )],
            ),
            None,
        )
        ds = vk.vkAllocateDescriptorSets(
            ctx.device,
            vk.VkDescriptorSetAllocateInfo(
                descriptorPool=pool,
                descriptorSetCount=1,
                pSetLayouts=[ctx.ds_layout],
            ),
        )[0]
        vk.vkUpdateDescriptorSets(
            ctx.device, 1,
            [vk.VkWriteDescriptorSet(
                dstSet=ds,
                dstBinding=0,
                descriptorCount=1,
                descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                pBufferInfo=[vk.VkDescriptorBufferInfo(
                    buffer=buf, offset=0, range=size,
                )],
            )],
            0, [],
        )

        # record + submit
        cmd = vk.vkAllocateCommandBuffers(
            ctx.device,
            vk.VkCommandBufferAllocateInfo(
                commandPool=ctx.cmd_pool,
                level=vk.VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                commandBufferCount=1,
            ),
        )[0]
        vk.vkBeginCommandBuffer(
            cmd,
            vk.VkCommandBufferBeginInfo(
                flags=vk.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
            ),
        )
        vk.vkCmdBindPipeline(cmd, vk.VK_PIPELINE_BIND_POINT_COMPUTE, ctx.pipeline)
        vk.vkCmdBindDescriptorSets(
            cmd, vk.VK_PIPELINE_BIND_POINT_COMPUTE,
            ctx.pipeline_layout, 0, [ds], [],
        )
        vk.vkCmdDispatch(cmd, x_groups, 1, 1)
        vk.vkEndCommandBuffer(cmd)

        fence = vk.vkCreateFence(ctx.device, vk.VkFenceCreateInfo(), None)
        vk.vkQueueSubmit(ctx.queue, [vk.VkSubmitInfo(commandBuffers=[cmd])], fence)
        vk.vkWaitForFences(ctx.device, [fence], vk.VK_TRUE, int(1e9))

        vk.vkDestroyFence(ctx.device, fence, None)
        vk.vkFreeCommandBuffers(ctx.device, ctx.cmd_pool, [cmd])
        vk.vkDestroyDescriptorPool(ctx.device, pool, None)

    # ── internal: source → gpu buffer ────────────────────────────────────────

    @staticmethod
    def _to_flat(toy_path: str) -> list[int]:
        """
        Lex, parse, flatten a .toy file and pack into the GPU buffer layout:

            [n_instrs, n_consts, n_vars,
             op, dest, src1, src2,  ...×n_instrs,
             const_bits...,   (float bits packed as int)
             var_bits...,     (initialised to 0.0)
             0]               (output slot written by shader)
        """
        import struct as _struct

        with open(toy_path, "r", encoding="utf-8") as f:
            source = f.read()
        tokens = Lexer.tokenize(source)
        ast    = Parser.parse(tokens)

        fl = flattener.Flattener()
        fl.flatten(ast)

        flat_instrs  = fl.get_flat()
        const_values = fl.const_values
        n_vars       = len(fl.var_map)
        n_instrs     = len(flat_instrs) // 4
        n_consts     = len(const_values)

        buf = [n_instrs, n_consts, n_vars]
        buf.extend(flat_instrs)
        for v in const_values:
            buf.append(_struct.unpack("i", _struct.pack("f", float(v)))[0])
        for _ in range(n_vars):
            buf.append(0)
        buf.append(0)  # output slot

        return buf

    # ── internal: upload, dispatch, read result slot ──────────────────────────

    @staticmethod
    def _gpu_process_chunk(ctx, chunk: list[int]) -> bytes:
        """
        Upload *chunk* (GPU buffer layout), dispatch 1 workgroup,
        and return the 4-byte float result from the output slot.
        """
        size     = len(chunk) * 4
        buf, mem = GpuVulkan._upload_flat(ctx, chunk)

        try:
            GpuVulkan._run_compute(ctx, buf, size, x_groups=1)

            ptr       = vk.vkMapMemory(ctx.device, mem, 0, size, 0)
            all_bytes = bytes(vk.buffer(ptr, size))
            vk.vkUnmapMemory(ctx.device, mem)

        finally:
            vk.vkDestroyBuffer(ctx.device, buf, None)
            vk.vkFreeMemory(ctx.device, mem, None)

        return all_bytes[-4:]  # last slot = result float

    # ── public: compile ───────────────────────────────────────────────────────

    @staticmethod
    def compile(program: str) -> str:
        """
        Compile a .toy source file using the GPU and write a sibling .toyc.
        Returns the path of the written .toyc file.
        """
        if not program.endswith(".toy"):
            raise ValueError(f"Expected a .toy source file, got: {program!r}")
        if not os.path.isfile(program):
            raise FileNotFoundError(f"Source file not found: {program!r}")

        flat_data = GpuVulkan._to_flat(program)
        if not flat_data:
            raise RuntimeError("Flattener produced no data.")

        profile, _ = GpuVulkan._sel_gpu()
        ctx        = GpuVulkan._init_vulkan(profile)
        try:
            binary = GpuVulkan._gpu_process_chunk(ctx, flat_data)
        finally:
            GpuVulkan._destroy_vulkan(ctx)

        toyc_path = os.path.splitext(os.path.abspath(program))[0] + ".toyc"
        write_bytecode(toyc_path, binary)
        return toyc_path

    @staticmethod
    def compile_batch(program: str) -> str:
        """
        Same as compile() but splits flat data into VRAM-sized chunks.
        Returns the path of the written .toyc file.
        """
        if not program.endswith(".toy"):
            raise ValueError(f"Expected a .toy source file, got: {program!r}")
        if not os.path.isfile(program):
            raise FileNotFoundError(f"Source file not found: {program!r}")

        flat_data = GpuVulkan._to_flat(program)
        if not flat_data:
            raise RuntimeError("Flattener produced no data.")

        profile, batch_size = GpuVulkan._sel_gpu()
        ctx                 = GpuVulkan._init_vulkan(profile)

        # each instruction is 4 ints × 4 bytes = 16 bytes
        ints_per_batch = max(4, (batch_size // 16) * 4)

        try:
            binary = b""
            for offset in range(0, len(flat_data), ints_per_batch):
                chunk   = flat_data[offset : offset + ints_per_batch]
                binary += GpuVulkan._gpu_process_chunk(ctx, chunk)
        finally:
            GpuVulkan._destroy_vulkan(ctx)

        toyc_path = os.path.splitext(os.path.abspath(program))[0] + ".toyc"
        write_bytecode(toyc_path, binary)
        return toyc_path

    # ── public: run ───────────────────────────────────────────────────────────

    @staticmethod
    def run(program: str, silent: bool = False) -> Any:
        """
        Run a .toy or .toyc file on the GPU.
        .toy  → compile first (writes sibling .toyc), then execute.
        .toyc → load and execute directly.
        """
        if not isinstance(program, str):
            raise TypeError(
                f"GpuVulkan.run() expects a file path str, got {type(program).__name__}"
            )

        if program.endswith(".toy"):
            if not os.path.isfile(program):
                raise FileNotFoundError(f"Source file not found: {program!r}")
            toyc_path = GpuVulkan.compile(program)
        elif program.endswith(".toyc"):
            if not os.path.isfile(program):
                raise FileNotFoundError(f"Bytecode file not found: {program!r}")
            toyc_path = program
        elif os.path.isfile(program) and is_compiled_bytecode(program):
            toyc_path = program
        else:
            raise ValueError(
                f"Cannot load {program!r}: expected a .toy source or .toyc bytecode file"
            )

        instructions = read_bytecode(toyc_path)

        profile, _ = GpuVulkan._sel_gpu()
        ctx        = GpuVulkan._init_vulkan(profile)
        try:
            result = GpuVulkan._gpu_process_chunk(ctx, instructions)
        finally:
            GpuVulkan._destroy_vulkan(ctx)

        if not silent:
            print(result)

        return result