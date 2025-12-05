# SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
XRT-based implementation of the HostRuntime
"""
import logging
from collections import OrderedDict
from pathlib import Path
import numpy as np
import pyxrt

from ..hostruntime import HostRuntime, IronRuntimeError, KernelHandle
from ...device import Device, NPU1, NPU2
from .tensor import XRTTensor


class XRTKernelHandle(KernelHandle):
    def __init__(self, xclbin_path, kernel_name, insts_path):
        self.xclbin_path = xclbin_path
        self.kernel_name = kernel_name
        self.insts_path = insts_path

    def __eq__(self, other):
        if isinstance(other, XRTKernelHandle):
            return (
                self.xclbin_path == other.xclbin_path
                and self.kernel_name == other.kernel_name
                and self.insts_path == other.insts_path
            )
        return False

    def __hash__(self):
        return hash((self.xclbin_path, self.kernel_name, self.insts_path))


class XRTHostRuntime(HostRuntime):
    """Singleton manager for AIE XRT resources.
    There is a simple LRU cache of loaded kernels."""

    _instance = None
    _tensor_class = XRTTensor
    MAX_LOADED_KERNELS = 16

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        self._device = pyxrt.device(0)
        device_type_str = self._device.get_info(pyxrt.xrt_info_device.name)

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

        self._contexts = {}  # xclbin_path -> (context, xclbin)
        self._kernels = OrderedDict()  # (xclbin_path, kernel_name) -> kernel

    def load(
        self,
        xclbin_path: Path,
        insts_path: Path,
        kernel_name: str | None = None,
        fail_if_full: bool = False,
    ) -> XRTKernelHandle:
        if not xclbin_path.exists() or not xclbin_path.is_file():
            raise IronRuntimeError(
                f"xclbin {xclbin_path} does not exist or is not a file."
            )
        if not insts_path.exists() or not insts_path.is_file():
            raise IronRuntimeError(
                f"insts {insts_path} does not exist or is not a file."
            )

        if xclbin_path not in self._contexts:
            xclbin = pyxrt.xclbin(str(xclbin_path))
            self._device.register_xclbin(xclbin)
            xclbin_uuid = xclbin.get_uuid()
            context = pyxrt.hw_context(self._device, xclbin_uuid)
            self._contexts[xclbin_path] = (context, xclbin)
            logging.debug(f"Created new context for {Path(xclbin_path).name}")
        else:
            context, xclbin = self._contexts[xclbin_path]
            logging.debug(f"Reusing context for {Path(xclbin_path).name}")

        if kernel_name is None:
            kernels = xclbin.get_kernels()
            if not kernels:
                raise RuntimeError("No kernels found in xclbin")
            kernel_name = kernels[0].get_name()
        else:
            if not kernel_name in [k.get_name() for k in xclbin.get_kernels()]:
                raise IronRuntimeError(
                    f"Kernel {kernel_name} not found in xclbin (kernels found: {[k.get_name() for k in xclbin.get_kernels()]})"
                )

        kernel_handle = XRTKernelHandle(xclbin_path, kernel_name, insts_path)
        if kernel_handle in self._kernels:
            self._kernels.move_to_end(kernel_handle)
            logging.debug(
                f"Reusing kernel: {kernel_name} from xclbin {Path(xclbin_path).name}"
            )
        else:
            if len(self._kernels) >= self.MAX_LOADED_KERNELS:
                if fail_if_full:
                    raise IronRuntimeError(
                        f"Cache is full ({self.MAX_LOADED_KERNELS} kernels) and fail_if_full is True."
                    )
                self._kernels.popitem(last=False)

            kernel = pyxrt.kernel(context, kernel_name)
            insts = self.read_insts(insts_path)
            self._kernels[kernel_handle] = (kernel, insts)
            logging.debug(
                f"Created new kernel {kernel_name} from xclbin {Path(xclbin_path).name}"
            )
        return kernel_handle

    def run(self, kernel_handle: XRTKernelHandle, args, only_if_loaded=False):
        args = [a for a in args if not callable(a)]  # Filter out callable functions
        if not all([isinstance(a, self._tensor_class) for a in args]):
            raise IronRuntimeError(
                f"The {self.__class__.__name__} can only take {self._tensor_class.__name__} as arguments, but got: {args}"
            )

        if only_if_loaded:
            if kernel_handle not in self._kernels:
                raise IronRuntimeError(
                    f"Kernel {kernel_handle.kernel_name} is not loaded and only_if_loaded=True."
                )
            self._kernels.move_to_end(kernel_handle)
        else:
            # Ensure kernel is loaded and MRU
            self.load(
                kernel_handle.xclbin_path,
                kernel_handle.insts_path,
                kernel_handle.kernel_name,
            )

        kernel, insts = self._kernels[kernel_handle]
        insts_bo = pyxrt.bo(
            self._device,
            insts.nbytes,
            pyxrt.bo.cacheable,
            kernel.group_id(1),
        )
        insts_bo.write(insts.view(np.uint8), 0)
        insts_bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
        buffers = [a.buffer_object() for a in args]
        h = kernel(3, insts_bo, insts.nbytes, *buffers)
        r = h.wait()
        if r != pyxrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED:
            raise IronRuntimeError(f"Kernel returned {str(r)}")
        # delete insts buffer
        del insts_bo

    def device(self) -> Device:
        # return an instance of the device type
        return self._device_type()

    def cleanup(self):
        """Clean up all XRT resources"""
        self._kernels.clear()

        # Clear contexts
        for context, _xclbin in self._contexts.values():
            try:
                del context
            except Exception as e:
                raise IronRuntimeError(str(e))
        self._contexts.clear()

        # Clear device
        if self._device is not None:
            try:
                del self._device
            except Exception as e:
                IronRuntimeError(str(e))
            self._device = None

        logging.debug("Cleaned up AIE device manager")

    def reset(self):
        """Reset the device manager (for debugging)"""
        self.cleanup()
        XRTHostRuntime._instance = None
