# SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
XRT-based implementation of the HostRuntime
"""
import atexit
import logging
from collections import OrderedDict
import os
import time
import weakref
from pathlib import Path
from typing import TYPE_CHECKING
import numpy as np
import pyxrt

from ..hostruntime import HostRuntime, HostRuntimeError, KernelHandle, KernelResult

if TYPE_CHECKING:
    from aie.iron.device import Device
from .tensor import XRTTensor


# XRTKernelHandle(kernel, xclbin, context, insts_path)
class XRTKernelHandle(KernelHandle):
    """
    Handle for a loaded XRT kernel.
    """

    def __init__(self, kernel, xclbin, context, insts, insts_bo=None):
        """
        Initialize the XRTKernelHandle.

        Args:
            kernel: The XRT kernel object.
            xclbin: The XRT xclbin object.
            context: The XRT context object.
            insts: The instructions for the kernel.
            insts_bo (optional): The instruction buffer object. Defaults to None.
        """
        self.kernel = kernel
        self.xclbin = xclbin
        self.context = context
        self.insts = insts
        self.insts_bo = insts_bo


class XRTKernelResult(KernelResult):
    """A wrapper around data produced as the result of running a kernel with the PyXRT runtime"""

    def __init__(
        self,
        ret: pyxrt.ert_cmd_state,
        npu_time: int,
        trace_data: XRTTensor | None = None,
    ):
        super().__init__(npu_time, trace_data)
        self.ret = ret

    def is_success(self) -> bool:
        return self.ret == pyxrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED


class XRTHostRuntime(HostRuntime):
    """Singleton manager for AIE XRT resources."""

    # TODO: this is duplicated from the LIT helpers.
    # NPU Model mappings - centralized for easy updates
    # Maps generation name to list of model strings that may appear in xrt-smi
    NPU_MODELS = {
        "npu1": ["npu1", "Phoenix"],
        "npu2": ["npu4", "Strix", "npu5", "Strix Halo", "npu6", "Krackan"],
    }
    _tensor_class = XRTTensor

    def __init__(self):
        """
        Initialize the XRTHostRuntime.
        """
        self._device = pyxrt.device(0)
        self._device_type_str = self._device.get_info(pyxrt.xrt_info_device.name)

        self.npu_str = None
        for key, value in self.NPU_MODELS.items():
            if any([model in self._device_type_str for model in self.NPU_MODELS[key]]):
                self.npu_str = key
                break
        if not self.npu_str:
            raise RuntimeError(f"Unknown device type: {self._device_type_str}")

    @classmethod
    def read_insts(cls, insts_path: Path):
        """
        Reads instructions from the given file, with XRT-specific handling for ELF files.

        Args:
            insts_path (Path): Path to the instruction file.

        Returns:
            The instructions (either as bytes/array or XRT module).
        """
        # Overload the function in the generic class so we can use xrt-specific handling of elf files.
        ext = insts_path.suffix.lower()
        if ext == ".elf" and hasattr(pyxrt, "module"):
            elf = pyxrt.elf(str(insts_path))
            return pyxrt.module(elf)
        else:
            return super().read_insts(insts_path)

    def load(
        self,
        npu_kernel,
        **kwargs,
    ) -> XRTKernelHandle:
        """
        Load an NPU kernel into the XRT runtime.

        Args:
            npu_kernel: The NPU kernel to load.
            **kwargs: Additional arguments for loading.

        Returns:
            XRTKernelHandle: A handle to the loaded kernel.

        Raises:
            HostRuntimeError: If xclbin or insts files do not exist, or if kernel is not found.
        """
        xclbin_path = Path(npu_kernel.xclbin_path).resolve()
        insts_path = Path(npu_kernel.insts_path).resolve()
        kernel_name = npu_kernel.kernel_name

        if not xclbin_path.exists() or not xclbin_path.is_file():
            raise HostRuntimeError(
                f"xclbin {xclbin_path} does not exist or is not a file."
            )
        if not insts_path.exists() or not insts_path.is_file():
            raise HostRuntimeError(
                f"insts {insts_path} does not exist or is not a file."
            )

        xclbin = pyxrt.xclbin(str(xclbin_path))
        self._device.register_xclbin(xclbin)
        xclbin_uuid = xclbin.get_uuid()
        context = pyxrt.hw_context(self._device, xclbin_uuid)

        if kernel_name is None:
            kernels = xclbin.get_kernels()
            if not kernels:
                raise RuntimeError("No kernels found in xclbin")
            kernel_name = kernels[0].get_name()
        else:
            if not kernel_name in [k.get_name() for k in xclbin.get_kernels()]:
                raise HostRuntimeError(
                    f"Kernel {kernel_name} not found in xclbin (kernels found: {[k.get_name() for k in xclbin.get_kernels()]})"
                )

        insts = self.read_insts(insts_path)
        if hasattr(pyxrt, "module") and isinstance(insts, pyxrt.module):
            kernel = pyxrt.ext.kernel(context, insts, kernel_name)
        else:
            kernel = pyxrt.kernel(context, kernel_name)

        kernel_handle = XRTKernelHandle(kernel, xclbin, context, insts)
        return kernel_handle

    def run(
        self,
        kernel_handle: XRTKernelHandle,
        args,
        trace_config=None,
        fail_on_error: bool = True,
        **kwargs,
    ) -> XRTKernelResult:
        """
        Run a loaded XRT kernel.

        Args:
            kernel_handle (XRTKernelHandle): The handle to the loaded kernel.
            args: Arguments to pass to the kernel.
            trace_config (optional): Configuration for tracing. Defaults to None.
            fail_on_error (bool, optional): Whether to raise an exception on kernel failure. Defaults to True.
            **kwargs: Additional arguments.

        Returns:
            XRTKernelResult: The result of the kernel execution.

        Raises:
            HostRuntimeError: If arguments are invalid or kernel execution fails (and fail_on_error is True).
        """
        # Filter out callable functions and check arg types
        args = [a for a in args if not callable(a)]
        if not all([isinstance(a, self._tensor_class) for a in args]):
            raise HostRuntimeError(
                f"The {self.__class__.__name__} can only take {self._tensor_class.__name__} as arguments, but got: {args}"
            )
        [a.to("npu") for a in args]
        buffers = [a.buffer_object() for a in args]

        insts_bo = None
        insts_bytes = 0
        try:
            is_module = hasattr(pyxrt, "module") and isinstance(
                kernel_handle.insts, pyxrt.module
            )
            if not is_module:
                insts_bytes = kernel_handle.insts.nbytes
                if kernel_handle.insts_bo:
                    insts_bo = kernel_handle.insts_bo
                else:
                    insts_bo = self._tensor_class(
                        kernel_handle.insts,
                        flags=pyxrt.bo.cacheable,
                        group_id=kernel_handle.kernel.group_id(1),
                    ).buffer_object()

            start = time.time_ns()
            h = kernel_handle.kernel(3, insts_bo, insts_bytes, *buffers)
            r = h.wait()
            stop = time.time_ns()

            if fail_on_error and r != pyxrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED:
                raise HostRuntimeError(f"Kernel returned {str(r)}")
        finally:
            # delete insts buffer if it was created locally
            if insts_bo and not kernel_handle.insts_bo:
                del insts_bo

        return XRTKernelResult(r, stop - start)

    def device(self) -> "Device":
        """
        Get the device associated with this runtime.

        Returns:
            Device: The device object (NPU1 or NPU2).

        Raises:
            HostRuntimeError: If the device string is unknown.
        """
        from aie.iron.device import NPU1, NPU2

        devices = {
            "npu1": NPU1(),
            "npu2": NPU2(),
        }

        if self.npu_str in devices:
            return devices[self.npu_str]
        else:
            raise HostRuntimeError(
                f"Unknown device string: {self.npu_str}: expected npu1 or npu2"
            )


class CachedXRTKernelHandle(XRTKernelHandle):
    """
    A cached handle for a loaded XRT kernel.
    """

    def __init__(self, kernel, xclbin, context, insts, insts_bo=None):
        """
        Initialize the CachedXRTKernelHandle.

        Args:
            kernel: The XRT kernel object.
            xclbin: The XRT xclbin object.
            context: The XRT context object.
            insts: The instructions for the kernel.
            insts_bo (optional): The instruction buffer object. Defaults to None.
        """
        super().__init__(kernel, xclbin, context, insts, insts_bo)
        self._is_valid = True

    def invalidate(self):
        """
        Invalidate the handle and release resources.
        """
        self._is_valid = False
        if hasattr(self, "context"):
            del self.context
        if hasattr(self, "kernel"):
            del self.kernel
        if hasattr(self, "xclbin"):
            del self.xclbin
        if hasattr(self, "insts"):
            del self.insts
        if hasattr(self, "insts_bo"):
            del self.insts_bo


class CachedXRTRuntime(XRTHostRuntime):
    """
    A cached version of XRTHostRuntime that caches up to n contexts,
    depending on the type of NPU.
    It reuses contexts for the same xclbin (identified by path and mtime).
    """

    # I got these values through experimentation on two machines
    # These values are primarily determined by the hardware/driver, and could change
    # in the future. But currently, if you exceed these sizes, you will fail to be
    # able to create a new context. At the driver level, the cached contexts are
    # a system-wide constrained resource, so caching on systems with many concurrent
    # processes trying to create contexts (as in parallel CI jobs) can be flaky.
    # TODO: use some sort of file system artifact or figure out how to query the driver
    # for the state of the cache, and how to make loading operations atomic between processes.
    NPU_CONTEXT_CACHE_SIZE = {
        "npu1": 6,
        "npu2": 32,
    }

    def __init__(self):
        """
        Initialize the CachedXRTRuntime.
        """
        super().__init__()
        # We use OrderedDict so that we can use Fifo behavior for LRU eviction policies
        self._context_cache = OrderedDict()
        self._insts_cache = OrderedDict()

        # Set default from dict if present
        self._cache_size = None
        if self.npu_str in self.NPU_CONTEXT_CACHE_SIZE.keys():
            self._cache_size = self.NPU_CONTEXT_CACHE_SIZE[self.npu_str]

        # Environment variable always override default values
        # TODO: should probably emit warning if exceeds recorded max size.
        self._cache_size = os.environ.get("XRT_CONTEXT_CACHE_SIZE", self._cache_size)

        # Error if no default and no env var
        if self._cache_size is None:
            raise HostRuntimeError(f"No known cache size for {self.npu_str}")

        atexit.register(self.cleanup)

    def cleanup(self):
        """
        Clean up the cache by evicting all entries.
        """
        while self._context_cache:
            self._evict()
        while self._insts_cache:
            self._evict_insts()

    def _cleanup_entry(self, entry):
        context = entry["context"]
        handles = entry["handles"]

        # Invalidate all handles
        for ref in handles:
            handle = ref()
            if handle:
                handle.invalidate()

        # Explicitly delete context
        del context

    def _evict(self):
        # Pop the oldest item
        key, entry = self._context_cache.popitem(last=False)
        self._cleanup_entry(entry)

    def _cleanup_insts_entry(self, entry):
        insts_bo = entry["insts_bo"]
        del insts_bo

    def _evict_insts(self):
        key, entry = self._insts_cache.popitem(last=False)
        self._cleanup_insts_entry(entry)

    def run(
        self,
        kernel_handle: XRTKernelHandle,
        args,
        trace_config=None,
        fail_on_error: bool = True,
        only_if_loaded: bool = False,
        **kwargs,
    ) -> XRTKernelResult:
        """
        Run a loaded XRT kernel.

        Args:
            kernel_handle (XRTKernelHandle): The handle to the loaded kernel.
            args: Arguments to pass to the kernel.
            trace_config (optional): Configuration for tracing. Defaults to None.
            fail_on_error (bool, optional): Whether to raise an exception on kernel failure. Defaults to True.
            only_if_loaded (bool, optional): If True, only run if the kernel is currently loaded in the cache. Defaults to False.
            **kwargs: Additional arguments.

        Returns:
            XRTKernelResult: The result of the kernel execution.

        Raises:
            HostRuntimeError: If arguments are invalid, kernel execution fails, or kernel is not loaded (if only_if_loaded=True).
        """
        if only_if_loaded:
            if (
                isinstance(kernel_handle, CachedXRTKernelHandle)
                and not kernel_handle._is_valid
            ):
                raise HostRuntimeError("Kernel not loaded (evicted from cache)")

        return super().run(kernel_handle, args, trace_config, fail_on_error, **kwargs)

    def load(
        self,
        npu_kernel,
        retry: bool = True,
        **kwargs,
    ) -> XRTKernelHandle:
        """
        Load an NPU kernel into the cached XRT runtime.

        Args:
            npu_kernel: The NPU kernel to load.
            retry (bool, optional): Whether to retry loading if context creation fails due to resource limits. Defaults to True.
            **kwargs: Additional arguments for loading.

        Returns:
            XRTKernelHandle: A handle to the loaded kernel.

        Raises:
            HostRuntimeError: If xclbin or insts files do not exist, or if kernel is not found.
        """
        xclbin_path = Path(npu_kernel.xclbin_path).resolve()
        insts_path = Path(npu_kernel.insts_path).resolve()
        kernel_name = npu_kernel.kernel_name

        if not xclbin_path.exists() or not xclbin_path.is_file():
            raise HostRuntimeError(
                f"xclbin {xclbin_path} does not exist or is not a file."
            )
        if not insts_path.exists() or not insts_path.is_file():
            raise HostRuntimeError(
                f"insts {insts_path} does not exist or is not a file."
            )

        xclbin_mtime = xclbin_path.stat().st_mtime
        insts_mtime = insts_path.stat().st_mtime

        # Context Cache Lookup
        context_key = (str(xclbin_path), xclbin_mtime)

        try:
            if context_key in self._context_cache:
                entry = self._context_cache[context_key]
                self._context_cache.move_to_end(context_key)
                context = entry["context"]
                xclbin = entry["xclbin"]
                # Clean up dead handles
                entry["handles"] = [
                    ref for ref in entry["handles"] if ref() is not None
                ]
            else:
                xclbin = pyxrt.xclbin(str(xclbin_path))
                xclbin_uuid = xclbin.get_uuid()

                if len(self._context_cache) >= self._cache_size:
                    self._evict()

                self._device.register_xclbin(xclbin)

                # Try to create context, evicting if necessary
                context = None
                retries = 0
                max_retries = len(self._context_cache) if retry else 0
                while context is None:
                    try:
                        context = pyxrt.hw_context(self._device, xclbin_uuid)
                    except RuntimeError as e:
                        # If we hit a resource limit (err=-2 usually means EMFILE/ENFILE or similar resource exhaustion)
                        # and we have items in the cache, try evicting.
                        if (
                            "No such file or directory" in str(e)
                            and self._context_cache
                            and retries < max_retries
                        ):
                            self._evict()
                            retries += 1
                        else:
                            raise e

                entry = {
                    "context": context,
                    "xclbin": xclbin,
                    "handles": [],
                    "uuid": xclbin_uuid,
                }
                self._context_cache[context_key] = entry

            # Kernel Name Resolution
            if kernel_name is None:
                kernels = xclbin.get_kernels()
                if not kernels:
                    raise RuntimeError("No kernels found in xclbin")
                kernel_name = kernels[0].get_name()
            else:
                if not kernel_name in [k.get_name() for k in xclbin.get_kernels()]:
                    raise HostRuntimeError(
                        f"Kernel {kernel_name} not found in xclbin (kernels found: {[k.get_name() for k in xclbin.get_kernels()]})"
                    )

            insts = self.read_insts(insts_path)
            insts_bo = None
            if hasattr(pyxrt, "module") and isinstance(insts, pyxrt.module):
                kernel = pyxrt.ext.kernel(context, insts, kernel_name)
            else:
                kernel = pyxrt.kernel(context, kernel_name)

                # Magic number for RyzenAI group id that will be fixed in the future. See same code at XRT:
                # https://github.com/Xilinx/XRT/blob/56222ed5cfd119dff0d5bd920735b87024e8c829/src/runtime_src/core/common/api/xrt_module.cpp#L1621
                group_id = kernel.group_id(1)
                insts_key = (str(insts_path), insts_mtime, group_id)

                if insts_key in self._insts_cache:
                    insts_entry = self._insts_cache[insts_key]
                    self._insts_cache.move_to_end(insts_key)
                    insts_bo = insts_entry["insts_bo"]
                else:
                    if len(self._insts_cache) >= self._cache_size:
                        self._evict_insts()

                    insts_bo = self._tensor_class(
                        insts,
                        flags=pyxrt.bo.cacheable,
                        group_id=group_id,
                    ).buffer_object()

                    insts_entry = {
                        "insts_bo": insts_bo,
                    }
                    self._insts_cache[insts_key] = insts_entry

            kernel_handle = CachedXRTKernelHandle(
                kernel, xclbin, context, insts, insts_bo
            )
            entry["handles"].append(weakref.ref(kernel_handle))

            return kernel_handle

        except Exception:
            if context_key in self._context_cache:
                entry = self._context_cache[context_key]
                # Clean up dead handles
                entry["handles"] = [
                    ref for ref in entry["handles"] if ref() is not None
                ]
                if not entry["handles"]:
                    del self._context_cache[context_key]
                    self._cleanup_entry(entry)
            raise
