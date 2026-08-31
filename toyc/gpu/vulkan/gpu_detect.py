"""
gpu_detect.py — Vulkan GPU detection and device-profile store.

Enumerates every physical device visible to Vulkan and records:
  • identity   – name, vendor/device IDs, driver version, API version
  • type        – discrete / integrated / virtual / CPU / other
  • memory      – heaps (size + flags) and memory types
  • queue families – count, flags, min-image granularity, timestamp bits
  • limits      – full VkPhysicalDeviceLimits dict (for batch-size math)
  • features    – full VkPhysicalDeviceFeatures dict

Everything lands in a plain Python dict keyed by device index so that
later code (batch-size selection, workload routing, etc.) can query it
without re-running Vulkan.  Optionally persisted to JSON.
"""

import json
import sys
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Any

try:
    import vulkan as vk
except ImportError:
    sys.exit(
        "vulkan-python not found.  Install it with:\n"
        "  pip install vulkan"
    )


# ── data model ────────────────────────────────────────────────────────────────

_DEVICE_TYPE_NAME = {
    vk.VK_PHYSICAL_DEVICE_TYPE_OTHER:          "other",
    vk.VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: "integrated",
    vk.VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:   "discrete",
    vk.VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:    "virtual",
    vk.VK_PHYSICAL_DEVICE_TYPE_CPU:            "cpu",
}

def _vk_const(name: str, fallback: int) -> int:
    """getattr with a fallback for constants missing in older vulkan-python builds."""
    return getattr(vk, name, fallback)

_MEM_HEAP_FLAGS = {
    _vk_const("VK_MEMORY_HEAP_DEVICE_LOCAL_BIT",   0x00000001): "DEVICE_LOCAL",
    _vk_const("VK_MEMORY_HEAP_MULTI_INSTANCE_BIT", 0x00000002): "MULTI_INSTANCE",  # Vulkan 1.1+
}

_MEM_TYPE_FLAGS = {
    _vk_const("VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT",     0x00000001): "DEVICE_LOCAL",
    _vk_const("VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT",     0x00000002): "HOST_VISIBLE",
    _vk_const("VK_MEMORY_PROPERTY_HOST_COHERENT_BIT",    0x00000004): "HOST_COHERENT",
    _vk_const("VK_MEMORY_PROPERTY_HOST_CACHED_BIT",      0x00000008): "HOST_CACHED",
    _vk_const("VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT", 0x00000010): "LAZILY_ALLOCATED",
    _vk_const("VK_MEMORY_PROPERTY_PROTECTED_BIT",        0x00000020): "PROTECTED",   # Vulkan 1.1+
}

_QUEUE_FLAGS = {
    _vk_const("VK_QUEUE_GRAPHICS_BIT",       0x00000001): "GRAPHICS",
    _vk_const("VK_QUEUE_COMPUTE_BIT",        0x00000002): "COMPUTE",
    _vk_const("VK_QUEUE_TRANSFER_BIT",       0x00000004): "TRANSFER",
    _vk_const("VK_QUEUE_SPARSE_BINDING_BIT", 0x00000008): "SPARSE_BINDING",
    _vk_const("VK_QUEUE_PROTECTED_BIT",      0x00000010): "PROTECTED",               # Vulkan 1.1+
}


def _decode_flags(value: int, flag_map: dict) -> list[str]:
    return [name for bit, name in flag_map.items() if value & bit]


def _version_tuple(packed: int) -> tuple[int, int, int]:
    return (packed >> 22, (packed >> 12) & 0x3FF, packed & 0xFFF)


def _version_str(packed: int) -> str:
    return "%d.%d.%d" % _version_tuple(packed)


@dataclass
class QueueFamily:
    index: int
    queue_count: int
    flags: list[str]
    timestamp_valid_bits: int
    min_image_transfer_granularity: dict  # {width, height, depth}

    # Convenience helpers used by batch-planning code
    @property
    def supports_compute(self) -> bool:
        return "COMPUTE" in self.flags

    @property
    def supports_graphics(self) -> bool:
        return "GRAPHICS" in self.flags

    @property
    def supports_transfer(self) -> bool:
        return "TRANSFER" in self.flags or self.supports_compute or self.supports_graphics


@dataclass
class MemoryHeap:
    index: int
    size_bytes: int
    size_mb: float
    flags: list[str]

    @property
    def is_device_local(self) -> bool:
        return "DEVICE_LOCAL" in self.flags


@dataclass
class MemoryType:
    index: int
    heap_index: int
    flags: list[str]

    @property
    def is_host_visible(self) -> bool:
        return "HOST_VISIBLE" in self.flags

    @property
    def is_coherent(self) -> bool:
        return "HOST_COHERENT" in self.flags

    @property
    def is_device_local(self) -> bool:
        return "DEVICE_LOCAL" in self.flags


@dataclass
class DeviceProfile:
    index: int                          # position in vkEnumeratePhysicalDevices
    name: str
    device_type: str
    vendor_id: int
    device_id: int
    driver_version: str
    api_version: str

    queue_families: list[QueueFamily]   = field(default_factory=list)
    memory_heaps:   list[MemoryHeap]    = field(default_factory=list)
    memory_types:   list[MemoryType]    = field(default_factory=list)
    limits:         dict[str, Any]      = field(default_factory=dict)
    features:       dict[str, Any]      = field(default_factory=dict)

    # ── derived helpers used by batch-planning ────────────────────────────────

    @property
    def total_device_local_memory_bytes(self) -> int:
        return sum(h.size_bytes for h in self.memory_heaps if h.is_device_local)

    @property
    def total_device_local_memory_mb(self) -> float:
        return self.total_device_local_memory_bytes / (1024 ** 2)

    @property
    def compute_queue_families(self) -> list[QueueFamily]:
        return [q for q in self.queue_families if q.supports_compute]

    @property
    def max_concurrent_compute_queues(self) -> int:
        return sum(q.queue_count for q in self.compute_queue_families)

    @property
    def max_workgroup_count(self) -> tuple[int, int, int]:
        lim = self.limits
        return (
            lim.get("maxComputeWorkGroupCount", [0, 0, 0])[0],
            lim.get("maxComputeWorkGroupCount", [0, 0, 0])[1],
            lim.get("maxComputeWorkGroupCount", [0, 0, 0])[2],
        )

    @property
    def max_workgroup_size(self) -> tuple[int, int, int]:
        lim = self.limits
        return (
            lim.get("maxComputeWorkGroupSize", [0, 0, 0])[0],
            lim.get("maxComputeWorkGroupSize", [0, 0, 0])[1],
            lim.get("maxComputeWorkGroupSize", [0, 0, 0])[2],
        )

    @property
    def max_workgroup_invocations(self) -> int:
        return self.limits.get("maxComputeWorkGroupInvocations", 0)


# ── detection ─────────────────────────────────────────────────────────────────

def _extract_limits(props) -> dict:
    """Pull every scalar / array field out of VkPhysicalDeviceLimits."""
    lim = props.limits
    return {
        "maxImageDimension1D":                  lim.maxImageDimension1D,
        "maxImageDimension2D":                  lim.maxImageDimension2D,
        "maxImageDimension3D":                  lim.maxImageDimension3D,
        "maxImageDimensionCube":                lim.maxImageDimensionCube,
        "maxImageArrayLayers":                  lim.maxImageArrayLayers,
        "maxTexelBufferElements":               lim.maxTexelBufferElements,
        "maxUniformBufferRange":                lim.maxUniformBufferRange,
        "maxStorageBufferRange":                lim.maxStorageBufferRange,
        "maxPushConstantsSize":                 lim.maxPushConstantsSize,
        "maxMemoryAllocationCount":             lim.maxMemoryAllocationCount,
        "maxSamplerAllocationCount":            lim.maxSamplerAllocationCount,
        "maxBoundDescriptorSets":               lim.maxBoundDescriptorSets,
        "maxPerStageDescriptorSamplers":        lim.maxPerStageDescriptorSamplers,
        "maxPerStageDescriptorUniformBuffers":  lim.maxPerStageDescriptorUniformBuffers,
        "maxPerStageDescriptorStorageBuffers":  lim.maxPerStageDescriptorStorageBuffers,
        "maxDescriptorSetStorageBuffers":       lim.maxDescriptorSetStorageBuffers,
        "maxComputeSharedMemorySize":           lim.maxComputeSharedMemorySize,
        "maxComputeWorkGroupCount":             list(lim.maxComputeWorkGroupCount),
        "maxComputeWorkGroupInvocations":       lim.maxComputeWorkGroupInvocations,
        "maxComputeWorkGroupSize":              list(lim.maxComputeWorkGroupSize),
        "subPixelPrecisionBits":                lim.subPixelPrecisionBits,
        "mipmapPrecisionBits":                  lim.mipmapPrecisionBits,
        "maxDrawIndexedIndexValue":             lim.maxDrawIndexedIndexValue,
        "maxDrawIndirectCount":                 lim.maxDrawIndirectCount,
        "maxSamplerLodBias":                    lim.maxSamplerLodBias,
        "maxSamplerAnisotropy":                 lim.maxSamplerAnisotropy,
        "maxViewports":                         lim.maxViewports,
        "maxViewportDimensions":                list(lim.maxViewportDimensions),
        "minMemoryMapAlignment":                lim.minMemoryMapAlignment,
        "minTexelBufferOffsetAlignment":        lim.minTexelBufferOffsetAlignment,
        "minUniformBufferOffsetAlignment":      lim.minUniformBufferOffsetAlignment,
        "minStorageBufferOffsetAlignment":      lim.minStorageBufferOffsetAlignment,
        "optimalBufferCopyOffsetAlignment":     lim.optimalBufferCopyOffsetAlignment,
        "optimalBufferCopyRowPitchAlignment":   lim.optimalBufferCopyRowPitchAlignment,
        "nonCoherentAtomSize":                  lim.nonCoherentAtomSize,
        "timestampComputeAndGraphics":          bool(lim.timestampComputeAndGraphics),
        "timestampPeriod":                      lim.timestampPeriod,
    }


def _extract_features(phys_dev) -> dict:
    """Return a flat bool dict from VkPhysicalDeviceFeatures."""
    f = vk.vkGetPhysicalDeviceFeatures(phys_dev)
    feature_names = [
        "robustBufferAccess", "fullDrawIndexUint32", "imageCubeArray",
        "independentBlend", "geometryShader", "tessellationShader",
        "sampleRateShading", "dualSrcBlend", "logicOp", "multiDrawIndirect",
        "drawIndirectFirstInstance", "depthClamp", "depthBiasClamp",
        "fillModeNonSolid", "depthBounds", "wideLines", "largePoints",
        "alphaToOne", "multiViewport", "samplerAnisotropy",
        "textureCompressionETC2", "textureCompressionASTC_LDR",
        "textureCompressionBC", "occlusionQueryPrecise",
        "pipelineStatisticsQuery", "vertexPipelineStoresAndAtomics",
        "fragmentStoresAndAtomics", "shaderTessellationAndGeometryPointSize",
        "shaderImageGatherExtended", "shaderStorageImageExtendedFormats",
        "shaderStorageImageMultisample", "shaderStorageImageReadWithoutFormat",
        "shaderStorageImageWriteWithoutFormat",
        "shaderUniformBufferArrayDynamicIndexing",
        "shaderSampledImageArrayDynamicIndexing",
        "shaderStorageBufferArrayDynamicIndexing",
        "shaderStorageImageArrayDynamicIndexing",
        "shaderClipDistance", "shaderCullDistance", "shaderFloat64",
        "shaderInt64", "shaderInt16", "shaderResourceResidency",
        "shaderResourceMinLod", "sparseBinding", "sparseResidencyBuffer",
        "sparseResidencyImage2D", "sparseResidencyImage3D",
        "sparseResidency2Samples", "sparseResidency4Samples",
        "sparseResidency8Samples", "sparseResidency16Samples",
        "sparseResidencyAliased", "variableMultisampleRate", "inheritedQueries",
    ]
    return {name: bool(getattr(f, name, False)) for name in feature_names}


def detect(verbose: bool = False) -> dict[int, DeviceProfile]:
    """
    Enumerate all Vulkan physical devices and return a dict
    mapping device index → DeviceProfile.

    Parameters
    ----------
    verbose : bool
        If True, print a summary table to stdout after detection.

    Returns
    -------
    dict[int, DeviceProfile]
        Keyed by the device's enumeration index (0, 1, …).
        Returns an empty dict if no Vulkan devices are found.
    """
    # --- create a minimal headless instance --------------------------------
    app_info = vk.VkApplicationInfo(
        sType=vk.VK_STRUCTURE_TYPE_APPLICATION_INFO,
        pApplicationName="gpu_detect",
        applicationVersion=vk.VK_MAKE_VERSION(1, 0, 0),
        pEngineName="gpu_detect",
        engineVersion=vk.VK_MAKE_VERSION(1, 0, 0),
        apiVersion=vk.VK_API_VERSION_1_0,
    )
    instance_info = vk.VkInstanceCreateInfo(
        sType=vk.VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        pApplicationInfo=app_info,
    )
    instance = vk.vkCreateInstance(instance_info, None)

    try:
        physical_devices = vk.vkEnumeratePhysicalDevices(instance)
    except vk.VkError:
        physical_devices = []

    if not physical_devices:
        vk.vkDestroyInstance(instance, None)
        return {}

    db: dict[int, DeviceProfile] = {}

    for idx, phys_dev in enumerate(physical_devices):
        props = vk.vkGetPhysicalDeviceProperties(phys_dev)

        # --- identity -------------------------------------------------------
        profile = DeviceProfile(
            index=idx,
            name=props.deviceName,
            device_type=_DEVICE_TYPE_NAME.get(props.deviceType, "unknown"),
            vendor_id=props.vendorID,
            device_id=props.deviceID,
            driver_version=_version_str(props.driverVersion),
            api_version=_version_str(props.apiVersion),
        )

        # --- queue families -------------------------------------------------
        qf_props = vk.vkGetPhysicalDeviceQueueFamilyProperties(phys_dev)
        for qi, qf in enumerate(qf_props):
            gran = qf.minImageTransferGranularity
            profile.queue_families.append(QueueFamily(
                index=qi,
                queue_count=qf.queueCount,
                flags=_decode_flags(qf.queueFlags, _QUEUE_FLAGS),
                timestamp_valid_bits=qf.timestampValidBits,
                min_image_transfer_granularity={
                    "width": gran.width, "height": gran.height, "depth": gran.depth
                },
            ))

        # --- memory ---------------------------------------------------------
        mem_props = vk.vkGetPhysicalDeviceMemoryProperties(phys_dev)
        for hi in range(mem_props.memoryHeapCount):
            heap = mem_props.memoryHeaps[hi]
            profile.memory_heaps.append(MemoryHeap(
                index=hi,
                size_bytes=heap.size,
                size_mb=round(heap.size / (1024 ** 2), 2),
                flags=_decode_flags(heap.flags, _MEM_HEAP_FLAGS),
            ))
        for ti in range(mem_props.memoryTypeCount):
            mt = mem_props.memoryTypes[ti]
            profile.memory_types.append(MemoryType(
                index=ti,
                heap_index=mt.heapIndex,
                flags=_decode_flags(mt.propertyFlags, _MEM_TYPE_FLAGS),
            ))

        # --- limits & features ----------------------------------------------
        profile.limits   = _extract_limits(props)
        profile.features = _extract_features(phys_dev)

        db[idx] = profile

    vk.vkDestroyInstance(instance, None)

    if verbose:
        _print_summary(db)

    return db


# ── persistence ───────────────────────────────────────────────────────────────

def save(db: dict[int, DeviceProfile], path: str | Path = "gpu_db.json") -> None:
    """Serialise the device database to JSON."""
    raw = {str(k): asdict(v) for k, v in db.items()}
    Path(path).write_text(json.dumps(raw, indent=2))


def load(path: str | Path = "gpu_db.json") -> dict[int, DeviceProfile]:
    """Deserialise a previously saved device database from JSON."""
    raw = json.loads(Path(path).read_text())
    db: dict[int, DeviceProfile] = {}
    for k, v in raw.items():
        v["queue_families"] = [QueueFamily(**q) for q in v["queue_families"]]
        v["memory_heaps"]   = [MemoryHeap(**h)  for h in v["memory_heaps"]]
        v["memory_types"]   = [MemoryType(**t)  for t in v["memory_types"]]
        db[int(k)] = DeviceProfile(**v)
    return db


# ── pretty-print ──────────────────────────────────────────────────────────────

def _print_summary(db: dict[int, DeviceProfile]) -> None:
    sep = "─" * 60
    for dev in db.values():
        print(sep)
        print(f"  [{dev.index}] {dev.name}")
        print(f"       type    : {dev.device_type}")
        print(f"       API     : {dev.api_version}  driver: {dev.driver_version}")
        print(f"       VRAM    : {dev.total_device_local_memory_mb:,.0f} MB  "
              f"(device-local)")
        print(f"       compute : {dev.max_concurrent_compute_queues} queue(s), "
              f"max {dev.max_workgroup_invocations} invocations/group")
        wgx, wgy, wgz = dev.max_workgroup_count
        print(f"       WG dims : {wgx} × {wgy} × {wgz}")
        print(f"       queues  :")
        for qf in dev.queue_families:
            print(f"                 [{qf.index}] count={qf.queue_count}  "
                  f"flags={','.join(qf.flags)}")
    print(sep)


# ── batch-planning helpers ────────────────────────────────────────────────────

def suggest_batch_size(
    profile: DeviceProfile,
    element_bytes: int,
    overhead_factor: float = 0.80,
) -> int:
    """
    Naive heuristic: fit as many elements as possible into
    `overhead_factor` of available device-local VRAM.

    Parameters
    ----------
    profile        : DeviceProfile  – target device
    element_bytes  : int            – bytes per single work item
    overhead_factor: float          – fraction of VRAM to target (default 80 %)

    Returns
    -------
    int  – suggested number of elements per batch
    """
    usable = profile.total_device_local_memory_bytes * overhead_factor
    return max(1, int(usable // element_bytes))


def select_device(
    db: dict[int, DeviceProfile],
    prefer: str = "discrete",
) -> DeviceProfile | None:
    """
    Return the best device matching `prefer` type,
    falling back to whichever has the most VRAM.
    """
    candidates = [d for d in db.values() if d.device_type == prefer]
    if not candidates:
        candidates = list(db.values())
    if not candidates:
        return None
    return max(candidates, key=lambda d: d.total_device_local_memory_bytes)


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Detecting Vulkan devices …\n")
    db = detect(verbose=True)

    if not db:
        print("No Vulkan-capable devices found.")
        sys.exit(1)

    save(db)
    print(f"\nDevice database written to gpu_db.json  ({len(db)} device(s))\n") 

    best = select_device(db)
    if best:
        bs = suggest_batch_size(best, element_bytes=4 * 1024 * 1024)  # 4 MB each
        print(f"Recommended device : [{best.index}] {best.name}")
        print(f"Suggested batch    : {bs:,} element(s)  "
              f"(4 MB each, 80 % of {best.total_device_local_memory_mb:,.0f} MB VRAM)")