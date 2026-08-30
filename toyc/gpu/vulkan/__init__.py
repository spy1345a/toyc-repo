# compiler/gpu/vulkan/__init__.py
#
# Re-exports GpuVulkan so callers can do either:
#   from compiler.gpu.vulkan import GpuVulkan
#   from compiler.gpu.vulkan.backend import GpuVulkan

from .backend import GpuVulkan

__all__ = ["GpuVulkan"]