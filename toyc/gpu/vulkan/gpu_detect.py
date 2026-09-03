"""
gpu_detect.py — Vulkan GPU detection and device-profile store.

Enumerates every physical device visible to Vulkan and records:
  • identity   – name, vendor/device IDs, driver version, API version
  • type        – discrete / integrated / virtual / CPU / other
  • memory      – heaps (size + flags) and memory types
  • queue families – count, flags, min-image granularity, timestamp bits
  • limits      – full VkPhysicalDeviceLimits dict (for batch-size math)
  • features    – full VkPhysicalDeviceFeatures dict
  • batch_size  – recommended batch size (4 MB elements, 80 % VRAM)

Caching behaviour
-----------------
Each GPU gets its own JSON file saved next to this script, named after
the GPU (spaces → underscores, e.g. "NVIDIA GeForce RTX 3060.json").
On the next call to detect(), if the file already exists for that GPU,
Vulkan enumeration is skipped entirely for that device and the profile
is loaded straight from disk.  Different GPUs → different JSON files.

Public API
----------
    db   = gpu_detect.detect(verbose=False)   # dict[int, DeviceProfile]
    size = gpu_detect.batch(profile)          # prints + returns int
    best = gpu_detect.select_device(db)       # DeviceProfile | None
"""

import json
import re
import sys
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Any

try:
    import vulkan as vk
    _VULKAN_AVAILABLE = True
except ImportError:
    _VULKAN_AVAILABLE = False

# Directory that contains this script — all JSON files land here.
_HERE = Path(__file__).resolve().parent

# Default bytes-per-element used for batch-size recommendations.
_DEFAULT_ELEMENT_BYTES = 4 * 1024 * 1024   # 4 MB
_DEFAULT_OVERHEAD      = 0.80              # use 80 % of VRAM


# ── data model ────────────────────────────────────────────────────────────────

_DEVICE_TYPE_NAME = {}  # filled lazily after vk import check

def _lazy_init_maps():
    global _DEVICE_TYPE_NAME, _MEM_HEAP_FLAGS, _MEM_TYPE_FLAGS, _QUEUE_FLAGS
    if _DEVICE_TYPE_NAME:
        return
    _DEVICE_TYPE_NAME = {
        vk.VK_PHYSICAL_DEVICE_TYPE_OTHER:          "other",
        vk.VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: "integrated",
        vk.VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:   "discrete",
        vk.VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:    "virtual",
        vk.VK_PHYSICAL_DEVICE_TYPE_CPU:            "cpu",
    }
    def _vk_const(name, fallback):
        return getattr(vk, name, fallback)
    _MEM_HEAP_FLAGS = {
        _vk_const("VK_MEMORY_HEAP_DEVICE_LOCAL_BIT",   0x1): "DEVICE_LOCAL",
        _vk_const("VK_MEMORY_HEAP_MULTI_INSTANCE_BIT", 0x2): "MULTI_INSTANCE",
    }
    _MEM_TYPE_FLAGS = {
        _vk_const("VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT",     0x01): "DEVICE_LOCAL",
        _vk_const("VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT",     0x02): "HOST_VISIBLE",
        _vk_const("VK_MEMORY_PROPERTY_HOST_COHERENT_BIT",    0x04): "HOST_COHERENT",
        _vk_const("VK_MEMORY_PROPERTY_HOST_CACHED_BIT",      0x08): "HOST_CACHED",
        _vk_const("VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT", 0x10): "LAZILY_ALLOCATED",
        _vk_const("VK_MEMORY_PROPERTY_PROTECTED_BIT",        0x20): "PROTECTED",
    }
    _QUEUE_FLAGS = {
        _vk_const("VK_QUEUE_GRAPHICS_BIT",       0x01): "GRAPHICS",
        _vk_const("VK_QUEUE_COMPUTE_BIT",        0x02): "COMPUTE",
        _vk_const("VK_QUEUE_TRANSFER_BIT",       0x04): "TRANSFER",
        _vk_const("VK_QUEUE_SPARSE_BINDING_BIT", 0x08): "SPARSE_BINDING",
        _vk_const("VK_QUEUE_PROTECTED_BIT",      0x10): "PROTECTED",
    }

_MEM_HEAP_FLAGS: dict = {}
_MEM_TYPE_FLAGS: dict = {}
_QUEUE_FLAGS:    dict = {}


def _decode_flags(value: int, flag_map: dict) -> list[str]:
    return [name for bit, name in flag_map.items() if value & bit]


def _version_str(packed: int) -> str:
    return "%d.%d.%d" % (packed >> 22, (packed >> 12) & 0x3FF, packed & 0xFFF)


def _gpu_json_path(gpu_name: str) -> Path:
    """Return the JSON path for a GPU name, e.g. 'NVIDIA RTX 3060' → script_dir/NVIDIA_RTX_3060.json"""
    safe = re.sub(r"[^\w\-.]", "_", gpu_name).strip("_")
    return _HERE / f"{safe}.json"


# ── dataclasses ───────────────────────────────────────────────────────────────

@dataclass
class QueueFamily:
    index: int
    queue_count: int
    flags: list[str]
    timestamp_valid_bits: int
    min_image_transfer_granularity: dict  # {width, height, depth}

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
    index: int
    name: str
    device_type: str
    vendor_id: int
    device_id: int
    driver_version: str
    api_version: str

    queue_families:       list[QueueFamily] = field(default_factory=list)
    memory_heaps:         list[MemoryHeap]  = field(default_factory=list)
    memory_types:         list[MemoryType]  = field(default_factory=list)
    limits:               dict[str, Any]    = field(default_factory=dict)
    features:             dict[str, Any]    = field(default_factory=dict)
    recommended_batch:    int               = 0   # saved to / loaded from JSON

    # ── derived helpers ────────────────────────────────────────────────────

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
        return tuple(lim.get("maxComputeWorkGroupCount", [0, 0, 0]))  # type: ignore[return-value]

    @property
    def max_workgroup_size(self) -> tuple[int, int, int]:
        lim = self.limits
        return tuple(lim.get("maxComputeWorkGroupSize", [0, 0, 0]))  # type: ignore[return-value]

    @property
    def max_workgroup_invocations(self) -> int:
        return self.limits.get("maxComputeWorkGroupInvocations", 0)


# ── serialisation helpers ─────────────────────────────────────────────────────

def _profile_to_dict(profile: DeviceProfile) -> dict:
    d = asdict(profile)
    d["recommended_batch"] = profile.recommended_batch
    return d


def _profile_from_dict(d: dict) -> DeviceProfile:
    d = dict(d)  # shallow copy so we can pop
    d["queue_families"] = [QueueFamily(**q) for q in d.get("queue_families", [])]
    d["memory_heaps"]   = [MemoryHeap(**h)  for h in d.get("memory_heaps",   [])]
    d["memory_types"]   = [MemoryType(**t)  for t in d.get("memory_types",   [])]
    return DeviceProfile(**d)


def _save_profile(profile: DeviceProfile) -> Path:
    """Write a single GPU profile to its own JSON file next to this script."""
    path = _gpu_json_path(profile.name)
    path.write_text(json.dumps(_profile_to_dict(profile), indent=2))
    return path


def _load_profile(gpu_name: str) -> DeviceProfile | None:
    """Try to load a cached profile for gpu_name. Returns None on miss."""
    path = _gpu_json_path(gpu_name)
    if not path.exists():
        return None
    try:
        return _profile_from_dict(json.loads(path.read_text()))
    except Exception:
        return None  # corrupt cache → fall through to Vulkan


# ── Vulkan extraction helpers ─────────────────────────────────────────────────

def _extract_limits(props) -> dict:
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


def _build_profile_from_vulkan(idx: int, phys_dev) -> DeviceProfile:
    """Query Vulkan for a single physical device and return a DeviceProfile."""
    _lazy_init_maps()
    props = vk.vkGetPhysicalDeviceProperties(phys_dev)

    profile = DeviceProfile(
        index=idx,
        name=props.deviceName,
        device_type=_DEVICE_TYPE_NAME.get(props.deviceType, "unknown"),
        vendor_id=props.vendorID,
        device_id=props.deviceID,
        driver_version=_version_str(props.driverVersion),
        api_version=_version_str(props.apiVersion),
    )

    # queue families
    for qi, qf in enumerate(vk.vkGetPhysicalDeviceQueueFamilyProperties(phys_dev)):
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

    # memory
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

    profile.limits   = _extract_limits(props)
    profile.features = _extract_features(phys_dev)
    return profile


# ── public API ────────────────────────────────────────────────────────────────

def detect(
    verbose: bool = False,
    element_bytes: int = _DEFAULT_ELEMENT_BYTES,
    overhead_factor: float = _DEFAULT_OVERHEAD,
) -> dict[int, "DeviceProfile"]:
    """
    Enumerate all Vulkan physical devices and return a dict
    mapping device index → DeviceProfile.

    For each GPU, if a JSON cache file already exists next to this script
    (named after the GPU), it is loaded without touching Vulkan.  Otherwise
    Vulkan is queried and the result is saved to that file automatically,
    including the recommended batch size.

    Parameters
    ----------
    verbose        : bool  – print a summary table to stdout
    element_bytes  : int   – bytes per work item used for batch recommendation
    overhead_factor: float – fraction of VRAM to target (default 0.80)

    Returns
    -------
    dict[int, DeviceProfile]
    """
    if not _VULKAN_AVAILABLE:
        sys.exit(
            "vulkan-python not found.  Install it with:\n"
            "  pip install vulkan"
        )

    # --- create a minimal headless Vulkan instance -------------------------
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

    # We need device names before we can check the cache, so do a quick
    # name-only pass first, then load or build each profile.
    db: dict[int, DeviceProfile] = {}

    for idx, phys_dev in enumerate(physical_devices):
        raw_name = vk.vkGetPhysicalDeviceProperties(phys_dev).deviceName

        cached = _load_profile(raw_name)
        if cached is not None:
            if verbose:
                print(f"  [{idx}] {raw_name}  → loaded from cache "
                      f"({_gpu_json_path(raw_name).name})")
            db[idx] = cached
            continue

        # Cache miss – query Vulkan in full
        profile = _build_profile_from_vulkan(idx, phys_dev)
        profile.recommended_batch = suggest_batch_size(
            profile, element_bytes, overhead_factor
        )
        path = _save_profile(profile)
        if verbose:
            print(f"  [{idx}] {profile.name}  → saved to {path.name}")
        db[idx] = profile

    vk.vkDestroyInstance(instance, None)

    if verbose:
        _print_summary(db)

    return db


def batch(
    profile: "DeviceProfile",
    element_bytes: int = _DEFAULT_ELEMENT_BYTES,
    overhead_factor: float = _DEFAULT_OVERHEAD,
) -> int:
    """
    Return the recommended batch size for *profile*, printing it at the
    same time.  If profile.recommended_batch is already set (e.g. loaded
    from cache), that value is used directly; otherwise it is computed,
    stored on the profile, and the JSON cache is updated.

    Assigning the return value and printing are both done in one call::

        n = gpu_detect.batch(profile)           # prints + returns
        print(n)                                 # still just the int
    """
    if profile.recommended_batch:
        size = profile.recommended_batch
    else:
        size = suggest_batch_size(profile, element_bytes, overhead_factor)
        profile.recommended_batch = size
        # update the JSON so the field is not missing next load
        _save_profile(profile)

    vram_mb = profile.total_device_local_memory_mb
    elem_mb = element_bytes / (1024 ** 2)
    print(
        f"Recommended batch  : {size:,} element(s)\n"
        f"  GPU              : [{profile.index}] {profile.name}\n"
        f"  VRAM             : {vram_mb:,.0f} MB  (device-local)\n"
        f"  Element size     : {elem_mb:.1f} MB\n"
        f"  Target occupancy : {overhead_factor * 100:.0f} %"
    )
    return size


def suggest_batch_size(
    profile: "DeviceProfile",
    element_bytes: int = _DEFAULT_ELEMENT_BYTES,
    overhead_factor: float = _DEFAULT_OVERHEAD,
) -> int:
    """
    Naïve heuristic: fit as many elements as possible into
    `overhead_factor` of available device-local VRAM.
    """
    usable = profile.total_device_local_memory_bytes * overhead_factor
    return max(1, int(usable // element_bytes))


def select_device(
    db: dict[int, "DeviceProfile"],
    prefer: str = "discrete",
) -> "DeviceProfile | None":
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


# ── pretty-print ──────────────────────────────────────────────────────────────

def _print_summary(db: dict[int, "DeviceProfile"]) -> None:
    sep = "─" * 60
    for dev in db.values():
        print(sep)
        print(f"  [{dev.index}] {dev.name}")
        print(f"       type    : {dev.device_type}")
        print(f"       API     : {dev.api_version}  driver: {dev.driver_version}")
        print(f"       VRAM    : {dev.total_device_local_memory_mb:,.0f} MB  (device-local)")
        print(f"       batch   : {dev.recommended_batch:,} element(s)  (recommended)")
        print(f"       compute : {dev.max_concurrent_compute_queues} queue(s), "
              f"max {dev.max_workgroup_invocations} invocations/group")
        wg = dev.max_workgroup_count
        print(f"       WG dims : {wg[0]} × {wg[1]} × {wg[2]}")
        print(f"       queues  :")
        for qf in dev.queue_families:
            print(f"                 [{qf.index}] count={qf.queue_count}  "
                  f"flags={','.join(qf.flags)}")
    print(sep)


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Detecting Vulkan devices …\n")
    db = detect(verbose=True)

    if not db:
        print("No Vulkan-capable devices found.")
        sys.exit(1)

    best = select_device(db)
    if best:
        print()
        n = batch(best)   # prints and returns; n holds the int