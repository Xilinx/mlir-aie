# SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
XRT-based implementation of the HostRuntime
"""
import atexit
import logging
import time
from collections import OrderedDict
from pathlib import Path
import numpy as np
import pyxrt

from ..hostruntime import HostRuntime, HostRuntimeError, KernelHandle, KernelResult
from ...device import Device, NPU1, NPU2
from .tensor import XRTTensor


class XRTKernelHandle(KernelHandle):
    def __init__(self, xclbin_path, kernel_name, insts_path, xclbin_mtime, insts_mtime):
        self.xclbin_path = xclbin_path
        self.kernel_name = kernel_name
        self.insts_path = insts_path
        self.xclbin_mtime = xclbin_mtime
        self.insts_mtime = insts_mtime

    def __eq__(self, other):
        if isinstance(other, XRTKernelHandle):
            return (
                self.xclbin_path == other.xclbin_path
                and self.kernel_name == other.kernel_name
                and self.insts_path == other.insts_path
                and self.xclbin_mtime == other.xclbin_mtime
                and self.insts_mtime == other.insts_mtime
            )
        return False

    def __hash__(self):
        return hash(
            (
                self.xclbin_path,
                self.kernel_name,
                self.insts_path,
                self.xclbin_mtime,
                self.insts_mtime,
            )
        )


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
    """Singleton manager for AIE XRT resources.
    There is a simple LRU cache of loaded XCLBINs."""

    _instance = None
    _tensor_class = XRTTensor

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        self._device = pyxrt.device(0)
        device_type_str = self._device.get_info(pyxrt.xrt_info_device.name)

        # Mostly for debugging, but also interesting to keep track of stats
        self._contexts_created = 0

        # Fetch the device type by matching strings for NPU2 or NPU1
        if any(
            keyword in device_type_str
            for keyword in [
                "NPU Strix",
                "NPU Strix Halo",
                "NPU Krackan",
                "RyzenAI-npu4",
                "RyzenAI-npu6",
            ]
        ):
            self._device_type = NPU2
        elif any(
            keyword in device_type_str
            for keyword in [
                "NPU",
                "NPU Phoenix",
                "RyzenAI-npu1",
            ]
        ):
            self._device_type = NPU1

        if self._device_type == NPU2:
            self._cache_size = 32
        else:
            self._cache_size = 1

        self._contexts = (
            OrderedDict()
        )  # (xclbin_path, xclbin_mtime) -> (context, xclbin)
        self._kernels = {}  # (xclbin_path, kernel_name) -> kernel
        atexit.register(self.cleanup)

    @classmethod
    def read_insts(cls, insts_path: Path):
        # Overload the function in the generic class so we can use xrt-specific handling of elf files.
        ext = insts_path.suffix.lower()
        if ext == ".elf":
            elf = pyxrt.elf(str(insts_path))
            return pyxrt.module(elf)
        else:
            return super().read_insts(insts_path)

    def _load_with_filemodtime_check(
        self, path: Path, load_func, expected_mtime: float, name: str
    ):
        mtime_before = path.stat().st_mtime
        result = load_func(path)
        mtime_after = path.stat().st_mtime
        if mtime_before != mtime_after:
            raise HostRuntimeError(f"{name} {path} modified during loading.")
        if mtime_after != expected_mtime:
            raise HostRuntimeError(
                f"{name} {path} modified during loading (mtime mismatch)."
            )
        return result

    def _evict_context_if_needed(self, fail_if_full):
        if len(self._contexts) >= self._cache_size:
            if fail_if_full:
                raise HostRuntimeError(
                    f"Cache is full ({self._cache_size} contexts) and fail_if_full is True."
                )
            # Evict oldest context
            (
                evicted_key,
                (evicted_context, evicted_xclbin),
            ) = self._contexts.popitem(last=False)

            # Remove associated kernels
            kernels_to_remove = [
                k
                for k in self._kernels
                if (k.xclbin_path, k.xclbin_mtime) == evicted_key
            ]
            for k in kernels_to_remove:
                kernel, insts = self._kernels.pop(k)
                try:
                    del kernel
                    del insts
                except Exception as e:
                    raise HostRuntimeError(str(e))

            try:
                del evicted_context
                del evicted_xclbin
            except Exception as e:
                raise HostRuntimeError(str(e))
            logging.debug(f"Evicted context for {evicted_key[0]}")

    def load(
        self,
        xclbin_path: Path,
        insts_path: Path,
        kernel_name: str | None = None,
        fail_if_full: bool = False,
    ) -> XRTKernelHandle:
        xclbin_path = xclbin_path.resolve()
        insts_path = insts_path.resolve()

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

        context_key = (xclbin_path, xclbin_mtime)

        if context_key not in self._contexts:
            self._evict_context_if_needed(fail_if_full)

            xclbin = self._load_with_filemodtime_check(
                xclbin_path,
                lambda p: pyxrt.xclbin(str(p)),
                xclbin_mtime,
                "xclbin",
            )

            self._device.register_xclbin(xclbin)
            xclbin_uuid = xclbin.get_uuid()
            try:
                context = pyxrt.hw_context(self._device, xclbin_uuid)
                self._contexts_created += 1
            except RuntimeError as e:
                if "DRM_IOCTL_AMDXDNA_CREATE_HWCTX" in str(e):
                    print(
                        f"Total contexts created: {self._contexts_created}, cached contexts: {len(self._contexts)}"
                    )
                    if not xclbin_path.exists():
                        raise RuntimeError(
                            f"Failed to create hw_context because xclbin file is missing: {xclbin_path}"
                        ) from e
                    raise RuntimeError(
                        f"Failed to create hw_context. xclbin file exists at {xclbin_path}. Original error: {e}"
                    ) from e
                raise e
            self._contexts[context_key] = (context, xclbin)
            logging.debug(f"Created new context for {Path(xclbin_path).name}")
        else:
            self._contexts.move_to_end(context_key)
            context, xclbin = self._contexts[context_key]
            logging.debug(f"Reusing context for {Path(xclbin_path).name}")

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

        kernel_handle = XRTKernelHandle(
            xclbin_path, kernel_name, insts_path, xclbin_mtime, insts_mtime
        )
        if kernel_handle in self._kernels:
            logging.debug(
                f"Reusing kernel: {kernel_name} from xclbin {Path(xclbin_path).name}"
            )
        else:
            insts = self._load_with_filemodtime_check(
                insts_path, self.read_insts, insts_mtime, "insts"
            )

            if isinstance(insts, pyxrt.module):
                kernel = pyxrt.ext.kernel(context, insts, kernel_name)
            else:
                kernel = pyxrt.kernel(context, kernel_name)

            self._kernels[kernel_handle] = (kernel, insts)
            logging.debug(
                f"Created new kernel {kernel_name} from xclbin {Path(xclbin_path).name}"
            )
        return kernel_handle

    def run(
        self,
        kernel_handle: XRTKernelHandle,
        args,
        only_if_loaded=False,
        fail_on_error=True,
    ) -> XRTKernelResult:
        args = [a for a in args if not callable(a)]  # Filter out callable functions
        if not all([isinstance(a, self._tensor_class) for a in args]):
            raise HostRuntimeError(
                f"The {self.__class__.__name__} can only take {self._tensor_class.__name__} as arguments, but got: {args}"
            )

        if only_if_loaded:
            context_key = (kernel_handle.xclbin_path, kernel_handle.xclbin_mtime)
            if context_key not in self._contexts:
                raise HostRuntimeError(
                    f"Context for kernel {kernel_handle.kernel_name} is not loaded and only_if_loaded=True."
                )

        # Ensure kernel is loaded
        self.load(
            kernel_handle.xclbin_path,
            kernel_handle.insts_path,
            kernel_handle.kernel_name,
        )

        kernel, insts = self._kernels[kernel_handle]
        buffers = [a.buffer_object() for a in args]

        insts_bo = None
        insts_bytes = 0
        if not isinstance(insts, pyxrt.module):
            insts_bytes = insts.nbytes
            insts_bo = pyxrt.bo(
                self._device,
                insts_bytes,
                pyxrt.bo.cacheable,
                kernel.group_id(1),
            )
            insts_bo.write(insts.view(np.uint8), 0)
            insts_bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

        start = time.time_ns()
        h = kernel(3, insts_bo, insts_bytes, *buffers)
        r = h.wait()
        stop = time.time_ns()

        # delete insts buffer
        if insts_bo:
            del insts_bo

        if fail_on_error and r != pyxrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED:
            raise HostRuntimeError(f"Kernel returned {str(r)}")

        return XRTKernelResult(r, stop - start, None)

    def device(self) -> Device:
        # return an instance of the device type
        return self._device_type()

    def cleanup(self):
        """Clean up all XRT resources"""
        while self._kernels:
            k, (kernel, insts) = self._kernels.popitem()
            del kernel
            del insts
            del k

        # Clear contexts
        contexts = list(self._contexts.values())
        self._contexts.clear()

        for context, xclbin in contexts:
            try:
                del context
                del xclbin
            except Exception as e:
                raise HostRuntimeError(str(e))

        # Clear device
        if self._device is not None:
            try:
                del self._device
            except Exception as e:
                HostRuntimeError(str(e))
            self._device = None

        logging.debug("Cleaned up AIE device manager")

    def reset(self):
        """Reset the device manager (for debugging)"""
        self.cleanup()
        self.__init__()
