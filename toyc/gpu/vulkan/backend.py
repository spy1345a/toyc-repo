import atexit
import os
import struct
import threading
import time
import warnings
from typing import Any

try:
    import vulkan as vk
    _VULKAN_AVAILABLE = True
    _VULKAN_IMPORT_ERROR = None
except (ImportError, OSError) as _exc:
    # Missing bindings OR missing loader/SDK (vulkan raises OSError
    # when libvulkan/Vulkan SDK can't be found). The CPU backend does
    # not need Vulkan, so importing toyc must keep working — GPU entry
    # points raise a clear error via _require_vulkan() instead.
    vk = None  # type: ignore
    _VULKAN_AVAILABLE = False
    _VULKAN_IMPORT_ERROR = _exc
    warnings.warn(
        "Vulkan SDK/loader not found — GpuVulkan is unavailable and "
        "any GPU call will raise RuntimeError; the CPU backend works "
        "normally. Install a Vulkan driver + SDK for GPU support "
        f"(import failed with: {_exc})",
        UserWarning,
        stacklevel=2,
    )

_NO_VULKAN_MSG = (
    "Vulkan SDK/loader not found, so the GPU backend cannot run. "
    "Use the CPU backend (Cpu.run), or install a Vulkan driver and SDK. "
    "See https://vulkan.lunarg.com/sdk/home"
)


def _require_vulkan() -> None:
    """Raise a friendly error if the Vulkan binding/loader is missing."""
    if not _VULKAN_AVAILABLE:
        raise RuntimeError(_NO_VULKAN_MSG)

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

# 10-second fence timeout in nanoseconds — generous for any real shader.
_FENCE_TIMEOUT_NS = int(10e9)

# Workgroup width of the data-parallel batch shader (comp_batch.glsl).
_BATCH_GROUP = 64


def _f2i(values: list[float]) -> list[int]:
    """Bit-cast floats to int32s with ONE struct round-trip (bulk)."""
    if not values:
        return []
    return list(struct.unpack(f"{len(values)}i",
                              struct.pack(f"{len(values)}f", *values)))

# ── persistent Vulkan session ───────────────────────────────────────────────
# Creating a Vulkan instance + logical device + compute pipeline costs
# ~15 ms on this hardware, while the actual dispatch is ~1 ms. So the
# instance/device/pipeline (and the GPU database) are created once per
# process and shared by every dispatch instead of being thrown away.
#
# The RLock serialises dispatches: the shared command pool, queue and
# cached handles are not safe for concurrent use, and vkQueueSubmit on
# one queue requires external synchronisation. Threads still share the
# session safely — they just take turns on the ~1 ms dispatch.
_CTX_LOCK = threading.RLock()
_DB_CACHE = None          # gpu_detect.detect() result, process-wide
_CTX_CACHE = None         # [key, ctx] of the shared Vulkan session
_ATEXIT_REGISTERED = False


def _shutdown_session() -> None:
    """Destroy the shared Vulkan session, if one exists (idempotent)."""
    global _CTX_CACHE, _DB_CACHE
    with _CTX_LOCK:
        if _CTX_CACHE is not None:
            try:
                GpuVulkan._destroy_vulkan(_CTX_CACHE[1])
            except Exception:
                pass
            _CTX_CACHE = None
        _DB_CACHE = None


class GpuVulkan:

    # ── device selection ──────────────────────────────────────────────────────

    @staticmethod
    def _get_db(debug: bool = False):
        """Return the process-wide GPU database, detecting once."""
        global _DB_CACHE
        with _CTX_LOCK:
            if _DB_CACHE is None:
                _DB_CACHE = gpu_detect.detect(debug=debug)
            elif debug:
                # Already cached: still honour debug with a short report.
                for dev in _DB_CACHE.values():
                    print(f"  [{dev.index}] {dev.name}  → loaded from cache")
            return _DB_CACHE

    @staticmethod
    def _sel_gpu(debug: bool = False):
        """Return (DeviceProfile, batch_size) for the best available GPU."""
        _require_vulkan()
        gpu_db = GpuVulkan._get_db(debug=debug)

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

        batch_size = gpu_detect.batch(profile, debug=debug)
        return profile, batch_size

    @staticmethod
    def _ctx_key(profile) -> tuple:
        """Cache key: device + age of both shader binaries."""
        parts = [profile.index, profile.name]
        for name in ("comp", "comp_batch"):
            spv_path = GpuVulkan._ensure_spv(name)
            try:
                mtime = os.path.getmtime(spv_path)
            except OSError:
                mtime = 0.0
            parts.extend((spv_path, mtime))
        return tuple(parts)

    @staticmethod
    def _get_ctx(profile, debug: bool = False):
        """
        Return the shared Vulkan session for *profile*, creating it on
        first use. A shader recompile (new comp.spv) transparently
        rebuilds the pipeline. Callers must hold _CTX_LOCK (it is an
        RLock, so re-entry is safe) and must NOT destroy the session.
        """
        global _CTX_CACHE, _ATEXIT_REGISTERED
        with _CTX_LOCK:
            key = GpuVulkan._ctx_key(profile)
            if _CTX_CACHE is not None and _CTX_CACHE[0] == key:
                return _CTX_CACHE[1]
            old = _CTX_CACHE
            ctx = GpuVulkan._init_vulkan(profile)
            if old is not None:
                try:
                    GpuVulkan._destroy_vulkan(old[1])
                except Exception:
                    pass
            _CTX_CACHE = [key, ctx]
            if not _ATEXIT_REGISTERED:
                atexit.register(_shutdown_session)
                _ATEXIT_REGISTERED = True
            return ctx

    @staticmethod
    def shutdown() -> None:
        """Release the shared Vulkan session and cached GPU database."""
        _shutdown_session()

    @staticmethod
    def startup(debug: bool = False):
        """
        Create the shared Vulkan session now (or reuse it if it already
        exists) so the first run()/compile() pays no setup cost.

        The session stays alive across all calls — run()/compile() never
        tear it down — until shutdown() or process exit. Returns the
        selected DeviceProfile.
        """
        _require_vulkan()
        profile, _ = GpuVulkan._sel_gpu(debug=debug)
        with _CTX_LOCK:
            GpuVulkan._get_ctx(profile, debug=debug)
        return profile

    # ── SPIR-V auto-compile ───────────────────────────────────────────────────

    @staticmethod
    def _ensure_spv(name: str = "comp") -> str:
        """
        Return the path to ``<name>.spv``, compiling from ``<name>.glsl``
        first if the .spv is missing or older than the .glsl source.

        Tries these compilers in order:
            1. glslangValidator  (glslang-tools / Vulkan SDK)
            2. glslc             (shaderc / Vulkan SDK)

        Raises RuntimeError if neither is found.
        """
        import subprocess
        import shutil

        here      = os.path.dirname(os.path.abspath(__file__))
        glsl_path = os.path.join(here, f"{name}.glsl")
        spv_path  = os.path.join(here, f"{name}.spv")

        # Skip recompile if .spv exists and is newer than .glsl.
        if (os.path.isfile(spv_path) and os.path.isfile(glsl_path) and
                os.path.getmtime(spv_path) >= os.path.getmtime(glsl_path)):
            return spv_path

        if not os.path.isfile(glsl_path):
            raise FileNotFoundError(
                f"Shader source not found: {glsl_path!r}\n"
                f"Place {name}.glsl next to backend.py."
            )

        compiler = shutil.which("glslangValidator") or shutil.which("glslc")
        if compiler is None:
            raise RuntimeError(
                "No GLSL to SPIR-V compiler found.\n"
                "Install one with:\n"
                "  sudo apt install glslang-tools    # glslangValidator\n"
                "  sudo apt install glslc            # glslc (shaderc)"
            )

        if os.path.basename(compiler) == "glslangValidator":
            # -S comp: comp.glsl doesn't have a stage-classified suffix,
            # so the stage must be given explicitly.
            cmd = [compiler, "-S", "comp", "-V", glsl_path, "-o", spv_path]
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

        Returns a plain namespace with all handles needed for dispatch and
        teardown.
        """
        _require_vulkan()
        # instance — include sType so strict loaders (RADV) accept it.
        app_info = vk.VkApplicationInfo(
            sType=vk.VK_STRUCTURE_TYPE_APPLICATION_INFO,
            pApplicationName="toyc",
            applicationVersion=vk.VK_MAKE_VERSION(1, 0, 0),
            pEngineName="toyc",
            engineVersion=vk.VK_MAKE_VERSION(1, 0, 0),
            apiVersion=vk.VK_API_VERSION_1_0,
        )
        instance_info = vk.VkInstanceCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
            pApplicationInfo=app_info,
        )
        instance = vk.vkCreateInstance(instance_info, None)

        # Pick the physical device that matches profile.index.
        phys_devs       = vk.vkEnumeratePhysicalDevices(instance)
        physical_device = phys_devs[profile.index]

        # Find a compute queue family.
        compute_qf = next(
            (qf for qf in profile.queue_families if "COMPUTE" in qf.flags),
            None,
        )
        if compute_qf is None:
            raise RuntimeError(f"No compute queue family on {profile.name!r}")
        qf_index = compute_qf.index

        # Logical device.
        queue_info = vk.VkDeviceQueueCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
            queueFamilyIndex=qf_index,
            queueCount=1,
            pQueuePriorities=[1.0],
        )
        device_info = vk.VkDeviceCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            queueCreateInfoCount=1,
            pQueueCreateInfos=[queue_info],
        )
        device = vk.vkCreateDevice(physical_device, device_info, None)
        queue    = vk.vkGetDeviceQueue(device, qf_index, 0)
        cmd_pool_info = vk.VkCommandPoolCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            flags=vk.VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
            queueFamilyIndex=qf_index,
        )
        cmd_pool = vk.vkCreateCommandPool(device, cmd_pool_info, None)
        # One persistent primary command buffer, reset per dispatch.
        cmd_alloc_info = vk.VkCommandBufferAllocateInfo(
            sType=vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            commandPool=cmd_pool,
            level=vk.VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            commandBufferCount=1,
        )
        cmd = vk.vkAllocateCommandBuffers(device, cmd_alloc_info)[0]
        # One persistent fence, reset per dispatch.
        fence = vk.vkCreateFence(
            device,
            vk.VkFenceCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
            ),
            None,
        )

        # Descriptor set layout: one storage buffer binding.
        binding = vk.VkDescriptorSetLayoutBinding(
            binding=0,
            descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            descriptorCount=1,
            stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT,
        )
        ds_layout_info = vk.VkDescriptorSetLayoutCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            bindingCount=1,
            pBindings=[binding],
        )
        ds_layout = vk.vkCreateDescriptorSetLayout(device, ds_layout_info, None)
        pipeline_layout_info = vk.VkPipelineLayoutCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            setLayoutCount=1,
            pSetLayouts=[ds_layout],
        )
        pipeline_layout = vk.vkCreatePipelineLayout(
            device, pipeline_layout_info, None
        )

        # Auto-compile the shaders if needed, then build one pipeline
        # per shader (single-value + data-parallel batch). Both share
        # the pipeline layout (one storage buffer binding).
        def _make_pipeline(name: str):
            spv_path = GpuVulkan._ensure_spv(name)
            with open(spv_path, "rb") as f:
                spv = f.read()
            shader_module_info = vk.VkShaderModuleCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                codeSize=len(spv),
                pCode=spv,
            )
            shader_module = vk.vkCreateShaderModule(
                device, shader_module_info, None
            )
            # NOTE: stage_info / pipeline_info MUST stay alive in locals
            # until vkCreateComputePipelines returns.
            # VkComputePipelineCreateInfo embeds `stage` by value (not
            # pointer), so vulkan-py does not keep the inner pName
            # ("main") string buffer alive if stage is an inline
            # temporary — the driver then reads a freed pointer and RADV
            # segfaults inside vkCreateComputePipelines.
            stage_info = vk.VkPipelineShaderStageCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                stage=vk.VK_SHADER_STAGE_COMPUTE_BIT,
                module=shader_module,
                pName="main",
            )
            pipeline_info = vk.VkComputePipelineCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
                stage=stage_info,
                layout=pipeline_layout,
            )
            pipeline = vk.vkCreateComputePipelines(
                device, None, 1, [pipeline_info], None
            )[0]
            vk.vkDestroyShaderModule(device, shader_module, None)
            return pipeline

        pipeline       = _make_pipeline("comp")
        pipeline_batch = _make_pipeline("comp_batch")

        # One persistent descriptor pool (reset per dispatch) plus its
        # allocate-info, both referencing the session's pool + layout.
        pool_size = vk.VkDescriptorPoolSize(
            type=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            descriptorCount=1,
        )
        desc_pool = vk.vkCreateDescriptorPool(
            device,
            vk.VkDescriptorPoolCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                maxSets=1,
                poolSizeCount=1,
                pPoolSizes=[pool_size],
            ),
            None,
        )
        alloc_info = vk.VkDescriptorSetAllocateInfo(
            sType=vk.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            descriptorPool=desc_pool,
            descriptorSetCount=1,
            pSetLayouts=[ds_layout],
        )

        class _Vk:
            pass

        ctx                 = _Vk()
        ctx.instance        = instance
        ctx.physical_device = physical_device
        ctx.device          = device
        ctx.queue           = queue
        ctx.cmd_pool        = cmd_pool
        ctx.cmd             = cmd
        ctx.fence           = fence
        ctx.ds_layout       = ds_layout
        ctx.pipeline_layout = pipeline_layout
        ctx.pipeline        = pipeline
        ctx.pipeline_batch  = pipeline_batch
        ctx.desc_pool       = desc_pool
        ctx.alloc_info      = alloc_info
        ctx.qf_index        = qf_index
        # Reusable scratch storage, grown on demand (None until first use).
        ctx.scratch_buf  = None
        ctx.scratch_mem  = None
        ctx.scratch_size = 0
        return ctx

    @staticmethod
    def _destroy_vulkan(ctx) -> None:
        """Clean up all Vulkan handles created by _init_vulkan."""
        if ctx.scratch_buf is not None:
            vk.vkDestroyBuffer(ctx.device, ctx.scratch_buf, None)
            vk.vkFreeMemory(ctx.device, ctx.scratch_mem, None)
        vk.vkDestroyFence(ctx.device, ctx.fence, None)
        vk.vkDestroyDescriptorPool(ctx.device, ctx.desc_pool, None)
        vk.vkDestroyPipeline(ctx.device, ctx.pipeline_batch, None)
        vk.vkDestroyPipeline(ctx.device, ctx.pipeline, None)
        vk.vkDestroyPipelineLayout(ctx.device, ctx.pipeline_layout, None)
        vk.vkDestroyDescriptorSetLayout(ctx.device, ctx.ds_layout, None)
        vk.vkDestroyCommandPool(ctx.device, ctx.cmd_pool, None)
        vk.vkDestroyDevice(ctx.device, None)
        vk.vkDestroyInstance(ctx.instance, None)

    # ── Vulkan buffer helpers ─────────────────────────────────────────────────

    @staticmethod
    def _find_memory_type(ctx, mem_type_bits: int, required_flags: int) -> int:
        """Return the index of a suitable memory type, or raise RuntimeError."""
        phys_mem = vk.vkGetPhysicalDeviceMemoryProperties(ctx.physical_device)
        for i in range(phys_mem.memoryTypeCount):
            if (mem_type_bits & (1 << i)) and \
               (phys_mem.memoryTypes[i].propertyFlags & required_flags) == required_flags:
                return i
        raise RuntimeError(
            f"No Vulkan memory type found for bits=0x{mem_type_bits:X} "
            f"flags=0x{required_flags:X}"
        )

    @staticmethod
    def _create_buffer(ctx, size: int, usage: int, mem_props_flags: int):
        """Allocate a Vulkan buffer + bound device memory. Returns (buf, mem)."""
        buf = vk.vkCreateBuffer(
            ctx.device,
            vk.VkBufferCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                size=size,
                usage=usage,
                sharingMode=vk.VK_SHARING_MODE_EXCLUSIVE,
            ),
            None,
        )
        mem_reqs       = vk.vkGetBufferMemoryRequirements(ctx.device, buf)
        mem_type_index = GpuVulkan._find_memory_type(
            ctx, mem_reqs.memoryTypeBits, mem_props_flags
        )
        memory = vk.vkAllocateMemory(
            ctx.device,
            vk.VkMemoryAllocateInfo(
                sType=vk.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                allocationSize=mem_reqs.size,
                memoryTypeIndex=mem_type_index,
            ),
            None,
        )
        vk.vkBindBufferMemory(ctx.device, buf, memory, 0)
        return buf, memory

    @staticmethod
    def _ensure_scratch(ctx, size: int):
        """
        Return a reusable (buf, mem) pair holding at least *size* bytes,
        growing the session's scratch storage on demand. Callers must
        hold _CTX_LOCK (all dispatch paths do).
        """
        if ctx.scratch_buf is None or size > ctx.scratch_size:
            if ctx.scratch_buf is not None:
                vk.vkDestroyBuffer(ctx.device, ctx.scratch_buf, None)
                vk.vkFreeMemory(ctx.device, ctx.scratch_mem, None)
            buf, mem = GpuVulkan._create_buffer(
                ctx, size,
                vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            )
            capacity = vk.vkGetBufferMemoryRequirements(
                ctx.device, buf).size
            ctx.scratch_buf  = buf
            ctx.scratch_mem  = mem
            ctx.scratch_size = capacity
        return ctx.scratch_buf, ctx.scratch_mem

    @staticmethod
    def _upload_flat(ctx, flat_data: list[int]):
        """Pack flat_data as int32 and upload to the scratch buffer."""
        packed = struct.pack(f"{len(flat_data)}i", *flat_data)
        size   = len(packed)

        buf, mem = GpuVulkan._ensure_scratch(ctx, size)

        # vkMapMemory returns a cffi buffer (ffi.buffer) supporting the
        # buffer protocol — write via slice assignment, not ctypes.memmove.
        mapped = vk.vkMapMemory(ctx.device, mem, 0, size, 0)
        try:
            mapped[:] = packed
        finally:
            vk.vkUnmapMemory(ctx.device, mem)

        return buf, mem

    @staticmethod
    def _run_compute(ctx, buf, size: int, x_groups: int,
                     pipeline=None) -> None:
        """
        Bind buf to a compute pipeline and dispatch x_groups workgroups.

        Reuses the session's descriptor pool, command buffer and fence
        (reset per dispatch) instead of recreating them every call.
        Callers must hold _CTX_LOCK.

        A VkMemoryBarrier is inserted after vkCmdDispatch so the GPU write
        is visible to the CPU before we map and read back the result.
        """
        # ── descriptor set (pool reset, then re-allocated) ──────────────
        vk.vkResetDescriptorPool(ctx.device, ctx.desc_pool, 0)
        ds = vk.vkAllocateDescriptorSets(ctx.device, ctx.alloc_info)[0]
        buffer_info = vk.VkDescriptorBufferInfo(
            buffer=buf, offset=0, range=size,
        )
        write = vk.VkWriteDescriptorSet(
            sType=vk.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            dstSet=ds,
            dstBinding=0,
            descriptorCount=1,
            descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            pBufferInfo=[buffer_info],
        )
        vk.vkUpdateDescriptorSets(ctx.device, 1, [write], 0, None)

        # ── command buffer (persistent, reset per dispatch) ──────────────
        cmd = ctx.cmd
        vk.vkResetCommandBuffer(cmd, 0)
        begin_info = vk.VkCommandBufferBeginInfo(
            sType=vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            flags=vk.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        )
        vk.vkBeginCommandBuffer(cmd, begin_info)
        vk.vkCmdBindPipeline(
            cmd, vk.VK_PIPELINE_BIND_POINT_COMPUTE,
            ctx.pipeline if pipeline is None else pipeline,
        )
        vk.vkCmdBindDescriptorSets(
            cmd, vk.VK_PIPELINE_BIND_POINT_COMPUTE,
            ctx.pipeline_layout, 0, 1, [ds], 0, None,
        )
        vk.vkCmdDispatch(cmd, x_groups, 1, 1)

        # Memory barrier: shader writes must be visible before CPU reads.
        # Without this, RADV/AMDVLK can hand back stale cache contents.
        barrier = vk.VkMemoryBarrier(
            sType=vk.VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            srcAccessMask=vk.VK_ACCESS_SHADER_WRITE_BIT,
            dstAccessMask=vk.VK_ACCESS_HOST_READ_BIT,
        )
        vk.vkCmdPipelineBarrier(
            cmd,
            vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,  # srcStageMask
            vk.VK_PIPELINE_STAGE_HOST_BIT,            # dstStageMask
            0,                                         # dependencyFlags
            1, [barrier],
            0, None,
            0, None,
        )
        vk.vkEndCommandBuffer(cmd)

        # ── submit + wait (persistent fence, reset per dispatch) ────────
        fence = ctx.fence
        vk.vkResetFences(ctx.device, 1, [fence])
        submit = vk.VkSubmitInfo(
            sType=vk.VK_STRUCTURE_TYPE_SUBMIT_INFO,
            commandBufferCount=1,
            pCommandBuffers=[cmd],
        )
        vk.vkQueueSubmit(ctx.queue, 1, [submit], fence)
        vk_result = vk.vkWaitForFences(
            ctx.device, 1, [fence], vk.VK_TRUE,
            _FENCE_TIMEOUT_NS,
        )
        if vk_result == vk.VK_TIMEOUT:
            raise RuntimeError(
                "GPU compute timed out after "
                f"{_FENCE_TIMEOUT_NS / 1e9:.0f} s"
            )

    # ── internal: source -> GPU buffer ────────────────────────────────────────

    @staticmethod
    def _classify(program: str) -> str:
        """
        Classify a program string as "toy" (.toy file), "toyc" (.toyc
        file or magic bytes) or "inline" (expression source text).
        Raises FileNotFoundError for missing .toy/.toyc paths and
        ValueError for existing files of unknown type.
        """
        if program.endswith(".toy"):
            if not os.path.isfile(program):
                raise FileNotFoundError(
                    f"Source file not found: {program!r}")
            return "toy"
        if program.endswith(".toyc"):
            if not os.path.isfile(program):
                raise FileNotFoundError(
                    f"Bytecode file not found: {program!r}")
            return "toyc"
        if os.path.isfile(program):
            if is_compiled_bytecode(program):
                return "toyc"
            raise ValueError(
                f"Cannot load {program!r}: expected a .toy source, "
                f".toyc file, or inline expression")
        return "inline"

    @staticmethod
    def _flatten_source(source: str):
        """Lex, parse and flatten source text. Returns the Flattener."""
        tokens = Lexer.tokenize(source)
        ast    = Parser.parse(tokens)
        fl = flattener.Flattener()
        fl.flatten(ast)
        return fl

    @staticmethod
    def _to_flat(toy_path: str, env: dict = None) -> list[int]:
        """
        Lex, parse, flatten a .toy file and pack into the GPU buffer layout:

            [n_instrs, n_consts, n_vars,
             op, dest, src1, src2,  ...x n_instrs,
             const_bits...,   (float bits packed as int)
             var_bits...,     (float bits from env, in first-use order)
             0,               (output slot written by shader)
             0]               (error flag written by shader: 0 ok, 1 div-by-zero)
        """
        with open(toy_path, "r", encoding="utf-8") as f:
            source = f.read()
        return GpuVulkan._flat_from_source(source, env)

    @staticmethod
    def _flat_from_source(source: str, env: dict = None) -> list[int]:
        """Flatten source text and pack the single-value GPU buffer."""
        fl = GpuVulkan._flatten_source(source)

        flat_instrs  = fl.get_flat()
        const_values = fl.const_values
        n_vars       = len(fl.var_map)
        n_instrs     = len(flat_instrs) // 4
        n_consts     = len(const_values)

        buf = [n_instrs, n_consts, n_vars]
        buf.extend(flat_instrs)
        buf.extend(_f2i([float(v) for v in const_values]))
        if n_vars:
            env = env or {}
            var_floats = [0.0] * n_vars
            for name, (_, var_idx) in fl.var_map.items():
                if name not in env:
                    raise NameError(f"Undefined variable: {name!r}")
                var_floats[var_idx] = float(env[name])
            buf.extend(_f2i(var_floats))
        buf.append(0)  # output slot
        buf.append(0)  # error flag slot (0 ok, 1 division by zero)

        return buf

    @staticmethod
    def _to_flat_batch(toy_path: str, envs: list[dict]) -> list[int]:
        """
        Lex, parse, flatten a .toy file once and pack the data-parallel
        GPU buffer layout for ``len(envs)`` instances:

            [n_instrs, n_consts, n_vars, n_instances,
             op, dest, src1, src2,  ...x n_instrs,
             const_bits...,   (float bits, shared by all instances)
             var_bits...,     (n_instances × n_vars floats, instance-major:
                              instance i owns var_base + i*n_vars + v)
             out...,          (n_instances output floats)
             err]             (1 shared error flag: 0 ok, 1 div-by-zero)

        Every env dict must define every variable (else NameError, like
        the CPU). An empty *envs* raises ValueError.
        """
        if not envs:
            raise ValueError("run_batch needs at least one variable set")
        with open(toy_path, "r", encoding="utf-8") as f:
            source = f.read()
        return GpuVulkan._flat_batch_from_source(source, envs)

    @staticmethod
    def _flat_batch_from_source(source: str,
                                envs: list[dict]) -> list[int]:
        """Flatten source text and pack the data-parallel GPU buffer."""
        if not envs:
            raise ValueError("run_batch needs at least one variable set")
        fl = GpuVulkan._flatten_source(source)

        flat_instrs  = fl.get_flat()
        const_values = fl.const_values
        var_names    = [None] * len(fl.var_map)
        for name, (_, var_idx) in fl.var_map.items():
            var_names[var_idx] = name
        n_vars   = len(var_names)
        n_instrs = len(flat_instrs) // 4
        n_consts = len(const_values)
        n_inst   = len(envs)

        buf = [n_instrs, n_consts, n_vars, n_inst]
        buf.extend(flat_instrs)
        buf.extend(_f2i([float(v) for v in const_values]))
        var_floats = []
        for inst, env in enumerate(envs):
            env = env or {}
            for name in var_names:
                if name not in env:
                    raise NameError(
                        f"Undefined variable: {name!r} "
                        f"(instance {inst} of {n_inst})"
                    )
                var_floats.append(float(env[name]))
        buf.extend(_f2i(var_floats))
        for _ in range(n_inst):
            buf.append(0)  # output slots
        buf.append(0)  # shared error flag slot

        return buf

    # ── internal: upload, dispatch, read result slot ──────────────────────────

    @staticmethod
    def _gpu_process_chunk(ctx, chunk: list[int]) -> bytes:
        """
        Upload *chunk* (GPU buffer layout as list[int]), dispatch 1 workgroup,
        and return the 4-byte float result from the output slot.

        Raises ZeroDivisionError with the same message as Cpu if the
        shader hit a division by zero (signalled through the trailing
        error slot), so both backends behave identically.

        vkMapMemory returns a cffi buffer; the result is read via
        slicing, not ctypes pointer arithmetic.
        """
        if not chunk:
            raise ValueError("_gpu_process_chunk received an empty chunk")

        chunk = list(chunk)
        if len(chunk) < 3:
            raise RuntimeError(
                f"Corrupt GPU buffer: header needs 3 ints, got {len(chunk)}"
            )
        n_instrs, n_consts, n_vars = chunk[0], chunk[1], chunk[2]
        if n_instrs < 0 or n_consts < 0 or n_vars < 0:
            raise RuntimeError(f"Corrupt GPU buffer header: {chunk[:3]!r}")
        out_idx = 3 + n_instrs * 4 + n_consts + n_vars
        err_idx = out_idx + 1
        if err_idx > len(chunk):
            raise RuntimeError(
                f"Corrupt GPU buffer: header {chunk[:3]!r} needs "
                f"{err_idx + 1} ints, buffer holds {len(chunk)}"
            )
        # Pad a missing trailing error slot (buffers built before it existed).
        while len(chunk) <= err_idx:
            chunk.append(0)

        size     = len(chunk) * 4          # bytes
        buf, mem = GpuVulkan._upload_flat(ctx, chunk)

        # NOTE: buf/mem are the session's reusable scratch storage — they
        # are NOT destroyed here (only grown, and freed at shutdown).
        GpuVulkan._run_compute(ctx, buf, size, x_groups=1)

        # Map the whole buffer, read the output slot and the error
        # flag by header-derived offsets, copy them out before unmapping.
        mapped = vk.vkMapMemory(ctx.device, mem, 0, size, 0)
        try:
            out_bytes = bytes(mapped[out_idx * 4 : out_idx * 4 + 4])
            err = struct.unpack("i", mapped[err_idx * 4 : err_idx * 4 + 4])[0]
        finally:
            vk.vkUnmapMemory(ctx.device, mem)

        if err != 0:
            raise ZeroDivisionError("Division by zero in VM")

        return out_bytes  # 4-byte IEEE-754 float result

    @staticmethod
    def _gpu_process_batch(ctx, chunk: list[int]) -> list[float]:
        """
        Upload a batch-layout buffer (see _to_flat_batch), dispatch
        ``ceil(n_instances / 64)`` workgroups of the batch pipeline, and
        return one float per instance.

        Raises ZeroDivisionError with the same message as Cpu if ANY
        instance divided by zero.
        """
        _BATCH_GROUP = 64
        chunk = list(chunk)
        if len(chunk) < 4:
            raise RuntimeError(
                f"Corrupt GPU batch buffer: header needs 4 ints, "
                f"got {len(chunk)}"
            )
        n_instrs, n_consts, n_vars, n_inst = chunk[:4]
        if n_instrs < 0 or n_consts < 0 or n_vars < 0 or n_inst <= 0:
            raise RuntimeError(f"Corrupt GPU batch header: {chunk[:4]!r}")
        out_base = 4 + n_instrs * 4 + n_consts + n_inst * n_vars
        err_idx  = out_base + n_inst
        if err_idx >= len(chunk):
            raise RuntimeError(
                f"Corrupt GPU batch buffer: header {chunk[:4]!r} needs "
                f"{err_idx + 1} ints, buffer holds {len(chunk)}"
            )

        size     = len(chunk) * 4          # bytes
        buf, mem = GpuVulkan._upload_flat(ctx, chunk)

        # NOTE: buf/mem are the session's reusable scratch storage — they
        # are NOT destroyed here (only grown, and freed at shutdown).
        groups = (n_inst + _BATCH_GROUP - 1) // _BATCH_GROUP
        GpuVulkan._run_compute(ctx, buf, size, x_groups=groups,
                               pipeline=ctx.pipeline_batch)

        mapped = vk.vkMapMemory(ctx.device, mem, 0, size, 0)
        try:
            out = [struct.unpack("f", mapped[(out_base + i) * 4:
                                             (out_base + i) * 4 + 4])[0]
                   for i in range(n_inst)]
            err = struct.unpack("i", mapped[err_idx * 4:
                                            err_idx * 4 + 4])[0]
        finally:
            vk.vkUnmapMemory(ctx.device, mem)

        if err != 0:
            raise ZeroDivisionError("Division by zero in VM")

        return out

    # ── public: compile ───────────────────────────────────────────────────────

    @staticmethod
    def _eval_flat_once(flat_data: list[int], debug: bool = False) -> bytes:
        """Dispatch one flat buffer on the GPU and return 4-byte result."""
        result, _ = GpuVulkan._eval_flat_timed(flat_data, debug=debug)
        return result

    @staticmethod
    def _eval_flat_timed(flat_data: list[int],
                         debug: bool = False) -> tuple[bytes, dict]:
        """
        Dispatch one flat buffer on the shared GPU session. Returns
        ``(result_bytes, timing)`` where *timing* holds seconds for each
        stage: ``{"select", "init", "exec", "teardown"}``. ``init`` is
        ~0 after the first call (session reuse) and ``teardown`` is
        always 0.0 (the session is kept, not destroyed).
        """
        if not flat_data:
            raise RuntimeError("Flattener produced no data.")
        t0 = time.perf_counter()
        profile, _ = GpuVulkan._sel_gpu(debug=debug)
        t1 = time.perf_counter()
        with _CTX_LOCK:
            ctx = GpuVulkan._get_ctx(profile, debug=debug)
            t2 = time.perf_counter()
            result = GpuVulkan._gpu_process_chunk(ctx, flat_data)
            t3 = time.perf_counter()
        timing = {
            "select":   t1 - t0,
            "init":     t2 - t1,
            "exec":     t3 - t2,
            "teardown": 0.0,
        }
        return result, timing

    @staticmethod
    def _print_timing(timing: dict) -> None:
        """Print one-line GPU timing report (all values are seconds)."""
        extra = ""
        if "num_batches" in timing:
            extra = (f" n={timing.get('n')} "
                     f"batches={timing.get('num_batches')} "
                     f"batch_size={timing.get('batch_size')}")
        print(
            "[GPU timing] "
            f"total={timing['total'] * 1e3:.3f} ms "
            f"(flatten={timing.get('flatten', 0.0) * 1e3:.3f} ms, "
            f"select={timing.get('select', 0.0) * 1e3:.3f} ms, "
            f"init={timing.get('init', 0.0) * 1e3:.3f} ms, "
            f"exec={timing.get('exec', 0.0) * 1e3:.3f} ms, "
            f"teardown={timing.get('teardown', 0.0) * 1e3:.3f} ms, "
            f"decode={timing.get('decode', 0.0) * 1e3:.3f} ms)"
            f"{extra}"
        )

    @staticmethod
    def compile(program: str, env: dict = None, debug: bool = False,
                timed: bool = False) -> str:
        """
        Evaluate a .toy source file on the GPU and write the result as
        a sibling .toyc file: 4-byte result for a single *env* dict,
        result list for a list of env dicts (auto-batched via
        compile_batch).
        Returns the path of the written .toyc file, or
        ``(path, timing)`` when timed=True (*timing* holds seconds per
        stage: ``{"total", "flatten", "select", "init", "exec",
        "teardown"}``, plus ``"n"`` for batches).
        """
        if isinstance(env, (list, tuple)):
            return GpuVulkan.compile_batch(program, list(env),
                                           debug=debug, timed=timed)
        if not program.endswith(".toy"):
            raise ValueError(f"Expected a .toy source file, got: {program!r}")
        if not os.path.isfile(program):
            raise FileNotFoundError(f"Source file not found: {program!r}")
        _require_vulkan()

        t_start = time.perf_counter() if timed else 0.0
        flat_data = GpuVulkan._to_flat(program, env)
        t_flat = time.perf_counter() if timed else 0.0
        if timed:
            binary, stages = GpuVulkan._eval_flat_timed(flat_data,
                                                        debug=debug)
        else:
            binary = GpuVulkan._eval_flat_once(flat_data, debug=debug)

        toyc_path = os.path.splitext(os.path.abspath(program))[0] + ".toyc"
        write_bytecode(toyc_path, binary)

        if timed:
            timing = {
                "total":    time.perf_counter() - t_start,
                "flatten":  t_flat - t_start,
                "select":   stages["select"],
                "init":     stages["init"],
                "exec":     stages["exec"],
                "teardown": stages["teardown"],
            }
            GpuVulkan._print_timing(timing)
            return toyc_path, timing
        return toyc_path

    @staticmethod
    def compile_batch(program: str, env=None, debug: bool = False,
                      timed: bool = False) -> str:
        """
        Evaluate a .toy source file for many variable sets on the GPU
        (same data-parallel engine as run_batch, chunking automatic)
        and write the result list as a sibling .toyc file.

        *env* is one mapping per instance — a list of dicts, a single
        dict (one instance), or None (one instance, no variables).
        Accepts a
        .toy source file like compile() — chunking, VRAM budget checks
        and error semantics are handled automatically.

        Returns the path of the written .toyc file, or
        ``(path, timing)`` when timed=True (same timing dict as
        run_batch, including ``"n"``).
        """
        if not program.endswith(".toy"):
            raise ValueError(f"Expected a .toy source file, got: {program!r}")
        if not os.path.isfile(program):
            raise FileNotFoundError(f"Source file not found: {program!r}")
        _require_vulkan()

        results, timing = GpuVulkan._run_batch_core(
            program, env, debug=debug, timed=timed)

        toyc_path = os.path.splitext(os.path.abspath(program))[0] + ".toyc"
        write_bytecode(toyc_path, results)

        if timed:
            GpuVulkan._print_timing(timing)
            return toyc_path, timing
        return toyc_path

    # ── public: run ───────────────────────────────────────────────────────────

    @staticmethod
    def run(program: str, env: dict = None, silent: bool = False,
            debug: bool = False, timed: bool = False,
            cache: bool = True) -> Any:
        """
        Run a .toy or .toyc file — or an inline expression — on the GPU.
        .toy  -> single *env* dict: flatten, single dispatch, cache the
                 4-byte result in a sibling .toyc, then return it.
                 list of env dicts: auto-batched via run_batch (one
                 float per instance, sibling .toyc holds the list).
        "1+2" -> inline expression: same as .toy but nothing is written
                 to disk.
        .toyc -> cached 4-byte scalar result, or cached batch result
                 list (both written by compile()/run()) — decoded
                 directly without touching the GPU; anything else is
                 treated as a flat program buffer and dispatched once.

        Returns the decoded float value (or a list of floats for
        batched runs); with timed=True returns ``(result, timing)``
        instead (*timing* holds seconds per stage: ``{"total",
        "flatten", "select", "init", "exec", "teardown", "decode"}`` —
        stages that did not run are 0.0).
        Prints the decoded value(s) unless silent=True.
        Prints GPU diagnostics only when debug=True.
        Prints a timing report when timed=True.
        Pass cache=False in hot loops to skip the sibling .toyc write.

        The shared GPU session is reused across calls and is never torn
        down here — call startup() once upfront to warm it, shutdown()
        when you are done (otherwise it is released at process exit).
        """
        if not isinstance(program, str):
            raise TypeError(
                f"GpuVulkan.run() expects a file path str, "
                f"got {type(program).__name__}"
            )

        if isinstance(env, (list, tuple)):
            # Batch of variable sets — the batching is automatic.
            return GpuVulkan.run_batch(program, list(env), silent=silent,
                                       debug=debug, timed=timed,
                                       cache=cache)

        timing = {
            "total": 0.0, "flatten": 0.0, "select": 0.0, "init": 0.0,
            "exec": 0.0, "teardown": 0.0, "decode": 0.0,
        }
        t_start = time.perf_counter() if timed else 0.0

        if program.endswith(".toy"):
            if not os.path.isfile(program):
                raise FileNotFoundError(f"Source file not found: {program!r}")
            _require_vulkan()
            flat_data = GpuVulkan._to_flat(program, env)
            t_flat = time.perf_counter() if timed else 0.0
            if timed:
                result, stages = GpuVulkan._eval_flat_timed(flat_data,
                                                            debug=debug)
                timing.update(flatten=t_flat - t_start, **stages)
            else:
                result = GpuVulkan._eval_flat_once(flat_data, debug=debug)
            if cache:
                # Cache the result next to the source for run(.toyc).
                toyc_path = os.path.splitext(
                    os.path.abspath(program))[0] + ".toyc"
                write_bytecode(toyc_path, result)
        elif not program.endswith(".toyc") and not os.path.isfile(program):
            # Inline expression, e.g. GpuVulkan.run("1 + 2"). No file to
            # cache next to, so nothing is written.
            _require_vulkan()
            flat_data = GpuVulkan._flat_from_source(program, env)
            t_flat = time.perf_counter() if timed else 0.0
            if timed:
                result, stages = GpuVulkan._eval_flat_timed(flat_data,
                                                            debug=debug)
                timing.update(flatten=t_flat - t_start, **stages)
            else:
                result = GpuVulkan._eval_flat_once(flat_data, debug=debug)
        elif program.endswith(".toyc") or (
            os.path.isfile(program) and is_compiled_bytecode(program)
        ):
            toyc_path = program
            if not os.path.isfile(toyc_path):
                raise FileNotFoundError(
                    f"Bytecode file not found: {toyc_path!r}"
                )
            raw = read_bytecode(toyc_path)
            if isinstance(raw, (bytes, bytearray)):
                if len(raw) == 4:
                    # Cached scalar result — no GPU dispatch needed.
                    t_dec = time.perf_counter() if timed else 0.0
                    result = bytes(raw)
                    if timed:
                        timing["decode"] = time.perf_counter() - t_dec
                else:
                    n = len(raw) // 4
                    if n == 0:
                        raise RuntimeError(
                            f"Bytecode file is empty: {toyc_path!r}"
                        )
                    _require_vulkan()
                    instructions = list(struct.unpack(f"{n}i", raw[:n * 4]))
                    if timed:
                        result, stages = GpuVulkan._eval_flat_timed(
                            instructions, debug=debug)
                        timing.update(stages)
                    else:
                        result = GpuVulkan._eval_flat_once(instructions,
                                                           debug=debug)
            elif isinstance(raw, list) and raw \
                    and all(isinstance(x, float) for x in raw):
                # Cached batch result list — no GPU dispatch needed.
                t_dec = time.perf_counter() if timed else 0.0
                result = list(raw)
                if timed:
                    timing["decode"] = time.perf_counter() - t_dec
                if not silent:
                    print(result)
                if timed:
                    timing["total"] = time.perf_counter() - t_start
                    GpuVulkan._print_timing(timing)
                    return result, timing
                return result
            elif isinstance(raw, list) and raw and isinstance(raw[0], int):
                _require_vulkan()
                if timed:
                    result, stages = GpuVulkan._eval_flat_timed(
                        list(raw), debug=debug)
                    timing.update(stages)
                else:
                    result = GpuVulkan._eval_flat_once(list(raw),
                                                       debug=debug)
            else:
                raise TypeError(
                    f"{toyc_path!r} holds CPU bytecode (list[Instr]); "
                    "execute it with Cpu.run(), not GpuVulkan.run()."
                )
        else:
            raise ValueError(
                f"Cannot load {program!r}: "
                "expected a .toy source or .toyc bytecode file"
            )

        if isinstance(result, list):
            value = result
        else:
            value = struct.unpack("f", result)[0]

        if not silent:
            print(value)

        if timed:
            timing["total"] = time.perf_counter() - t_start
            GpuVulkan._print_timing(timing)
            return value, timing

        return value

    # ── public: run_batch ─────────────────────────────────────────────────────

    @staticmethod
    def _norm_envs(env) -> list[dict]:
        """Normalise batch variable sets: None → [{}], dict → [dict]."""
        if env is None:
            return [{}]
        if isinstance(env, dict):
            return [env]
        env = list(env)
        if not env:
            raise ValueError("run_batch needs at least one variable set")
        return env

    @staticmethod
    def _run_batch_core(program: str, env, debug: bool = False,
                        timed: bool = False) -> tuple:
        """
        Shared engine for run_batch()/compile_batch(). Validates the
        .toy program, normalises *env*, and evaluates every instance
        in recommended-batch-size chunks on the shared session.
        Returns ``(results, timing)``; *timing* holds seconds per stage
        plus ``"n"``.
        """
        if not isinstance(program, str):
            raise TypeError(
                f"Batch program must be a .toy path or inline expression "
                f"str, got {type(program).__name__}")
        kind = GpuVulkan._classify(program)
        if kind == "toyc":
            raise ValueError(
                f"run_batch needs source (.toy file or inline "
                f"expression), got bytecode: {program!r}")
        if kind == "inline":
            source = program
        else:
            with open(program, "r", encoding="utf-8") as f:
                source = f.read()
        env = GpuVulkan._norm_envs(env)

        t_start = time.perf_counter() if timed else 0.0
        profile, batch_size = GpuVulkan._sel_gpu(debug=debug)
        t_sel = time.perf_counter() if timed else 0.0
        chunk_size = max(1, batch_size)
        usable = int(profile.total_device_local_memory_bytes * 0.80)

        results: list[float] = []
        chunks: list[dict] = []
        t_flatten = t_exec = 0.0
        with _CTX_LOCK:
            ctx = GpuVulkan._get_ctx(profile, debug=debug)
            t_init = time.perf_counter() if timed else 0.0
            for off in range(0, len(env), chunk_size):
                chunk_envs = env[off:off + chunk_size]
                f0 = time.perf_counter() if timed else 0.0
                flat = GpuVulkan._flat_batch_from_source(source,
                                                         chunk_envs)
                f1 = time.perf_counter() if timed else 0.0
                if len(flat) * 4 > usable:
                    raise RuntimeError(
                        f"Batch chunk ({len(flat) * 4} bytes) exceeds "
                        f"usable VRAM budget ({usable} bytes)."
                    )
                e0 = time.perf_counter() if timed else 0.0
                results.extend(GpuVulkan._gpu_process_batch(ctx, flat))
                e1 = time.perf_counter() if timed else 0.0
                chunks.append({
                    "batch_index": off // chunk_size,
                    "chunk_n":     len(chunk_envs),
                    "flatten":     (f1 - f0) if timed else 0.0,
                    "exec":        (e1 - e0) if timed else 0.0,
                })
                if timed:
                    t_flatten += f1 - f0
                    t_exec += e1 - e0

        timing = {
            "total":    (time.perf_counter() - t_start) if timed else 0.0,
            "flatten":  t_flatten,
            "select":   (t_sel - t_start) if timed else 0.0,
            "init":     (t_init - t_sel) if timed else 0.0,
            "exec":     t_exec,
            "teardown": 0.0,
            "decode":   0.0,
            "n":        len(env),
            "num_batches": len(chunks),
            "batch_size":  chunk_size,
            "chunks":      chunks,
        }
        return results, timing

    # ── public: run_batch ─────────────────────────────────────────────────────

    @staticmethod
    def run_batch(program: str, env=None, silent: bool = False,
                  debug: bool = False, timed: bool = False,
                  cache: bool = True) -> list[float] | tuple:
        """
        Evaluate a .toy expression for many variable sets, data-parallel:
        one shader invocation per instance, dispatched in chunks of the
        recommended batch size (see gpu_detect.batch) — chunking is
        automatic.

        Parameters
        ----------
        program : str
            .toy source file path or inline expression (variables come
            from *env*).
        env : list[dict] | dict | None
            One variable mapping per instance, e.g.
            ``[{"a": 1.0, "b": 2.0}, {"a": 3.0, "b": 4.0}]``; a single
            dict runs one instance; None runs one instance with no
            variables (constant expressions).
        silent : bool — print the result list unless True.
        debug  : bool — print GPU diagnostics.
        timed  : bool — print a timing report and return
            ``(results, timing)``; *timing* holds seconds per stage plus
            ``"n"`` (instance count).
        cache  : bool — write the result list to the sibling .toyc file
            (.toy paths only; skipped for inline expressions, read back
            without dispatch by run(".toyc")). Pass cache=False in hot
            loops to skip disk I/O.

        Returns a list of one float per instance (in *env* order).
        Raises NameError for variables missing from an env dict and
        ZeroDivisionError if ANY instance divides by zero — same
        exceptions as the CPU backend.
        """
        results, timing = GpuVulkan._run_batch_core(
            program, env, debug=debug, timed=timed)

        if cache and not (
                isinstance(program, str)
                and GpuVulkan._classify(program) == "inline"):
            toyc_path = os.path.splitext(os.path.abspath(program))[0] \
                + ".toyc"
            write_bytecode(toyc_path, results)

        if not silent:
            print(results)

        if timed:
            GpuVulkan._print_timing(timing)
            return results, timing

        return results