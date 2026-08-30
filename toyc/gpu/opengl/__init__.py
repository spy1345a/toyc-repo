# compiler/gpu/vulkan/__init__.py
#
# Re-exports GpuOpengl so callers can do either:
#   from compiler.gpu.vulkan import GpuOpengl
#   from compiler.gpu.vulkan.backend import GpuOpengl

from .backend import GpuOpengl

__all__ = ["GpuOpengl"]