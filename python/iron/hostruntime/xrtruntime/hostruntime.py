# SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
XRT-based implementation of the HostRuntime
"""
import logging
from pathlib import Path
import numpy as np
import pyxrt

from ..hostruntime import HostRuntime, KernelHandle
from ...device import Device, NPU1, NPU2


class XRTKernelHandle(KernelHandle):
    def __init__(self, xclbin_path, kernel_name, instr_path=None):
        self.xclbin_path = xclbin_path
        self.kernel_name = kernel_name
        self.instr_path = instr_path


class XRTHostRuntime(HostRuntime):
    """Singleton manager for AIE XRT resources"""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        # TODO: what is there is more than one device?
        self._device = pyxrt.device(0)
        device_type_str = self._device.get_info(pyxrt.xrt_info_device.name)

        # Fetch the device type by matching strings for NPU2 or NPU1
        # TODO: how to use only a portion of the device rather than whole array?
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
        self._kernels = {}  # (xclbin_path, kernel_name) -> kernel

    def load(self, xclbin_path, kernel_name=None, instr_path=None) -> XRTKernelHandle:
        if xclbin_path not in self._contexts:
            xclbin = pyxrt.xclbin(xclbin_path)
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

        kernel_key = (xclbin_path, kernel_name)
        if kernel_key not in self._kernels:
            kernel = pyxrt.kernel(context, kernel_name)
            insts_bo = None
            insts_len = 0
            if instr_path:
                instr_binary_nbytes = Path(instr_path).stat().st_size
                insts_bo = pyxrt.bo(
                    self._device,
                    instr_binary_nbytes,
                    pyxrt.bo.cacheable,
                    kernel.group_id(1),
                )
                insts_len = instr_binary_nbytes // 4

            self._kernels[kernel_key] = (kernel, insts_bo, insts_len, instr_path)
            logging.debug(
                f"Created new kernel {kernel_name} from xclbin {Path(xclbin_path).name}"
            )
        else:
            logging.debug(
                f"Reusing kernel: {kernel_name} from xclbin {Path(xclbin_path).name}"
            )
        return XRTKernelHandle(xclbin_path, kernel_name, instr_path)

    def run(self, kernel_handle: XRTKernelHandle, *args, **kwargs):
        kernel, insts_bo, insts_len, instr_path = self._kernels[
            (kernel_handle.xclbin_path, kernel_handle.kernel_name)
        ]
        if insts_bo:
            with open(instr_path, "rb") as f:
                instr_binary = np.frombuffer(f.read(), dtype=np.uint32)
            insts_bo.write(instr_binary.view(np.uint8), 0)
            kernel(3, insts_bo, insts_len, *args)
        else:
            kernel(*args, **kwargs)

    def device(self) -> Device:
        return self._device_type()

    def cleanup(self):
        """Clean up all XRT resources"""
        for _kernel, insts_bo, _insts_len, _instr_path in self._kernels.values():
            if insts_bo:
                del insts_bo
        self._kernels.clear()

        # Clear contexts
        for context, _xclbin in self._contexts.values():
            try:
                del context
            except Exception as e:
                # TODO(erika): why might this throw exceptions?
                raise e
        self._contexts.clear()

        # Clear device
        if self._device is not None:
            try:
                del self._device
            except Exception as e:
                # TODO(erika): why might this throw exceptions?
                raise e
            self._device = None

        logging.debug("Cleaned up AIE device manager")

    def reset(self):
        """Reset the device manager (for debugging)"""
        self.cleanup()
        XRTHostRuntime._instance = None
