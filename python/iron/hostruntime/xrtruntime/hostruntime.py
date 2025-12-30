# SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
XRT-based implementation of the HostRuntime
"""
import logging
import time
from collections import OrderedDict
from pathlib import Path
import numpy as np
import pyxrt

from ..hostruntime import HostRuntime, HostRuntimeError, KernelHandle, KernelResult
from ...device import Device, NPU1, NPU2
from .tensor import XRTTensor


# XRTKernelHandle(kernel, xclbin, context, insts_path)
class XRTKernelHandle(KernelHandle):
    def __init__(self, kernel, xclbin, context, insts_path):
        self.kernel = kernel
        self.xclbin = xclbin
        self.context = context
        self.insts_path = insts_path


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

    _tensor_class = XRTTensor

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

    @classmethod
    def read_insts(cls, insts_path: Path):
        # Overload the function in the generic class so we can use xrt-specific handling of elf files.
        ext = insts_path.suffix.lower()
        if ext == ".elf":
            elf = pyxrt.elf(str(insts_path))
            return pyxrt.module(elf)
        else:
            return super().read_insts(insts_path)

    def load(
        self,
        xclbin_path: Path,
        insts_path: Path,
        kernel_name: str | None = None,
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

        if isinstance(insts, pyxrt.module):
            kernel = pyxrt.ext.kernel(context, insts, kernel_name)
        else:
            kernel = pyxrt.kernel(context, kernel_name)

        kernel_handle = XRTKernelHandle(kernel, xclbin, context, insts_path)
        return kernel_handle

    def run(
        self,
        kernel_handle: XRTKernelHandle,
        args,
        fail_on_error: bool = True,
    ) -> XRTKernelResult:
        # Filter out callable functions and check arg types
        args = [a for a in args if not callable(a)]
        if not all([isinstance(a, self._tensor_class) for a in args]):
            raise HostRuntimeError(
                f"The {self.__class__.__name__} can only take {self._tensor_class.__name__} as arguments, but got: {args}"
            )
        [a.to("npu") for a in args]
        buffers = [a.buffer_object() for a in args]

        # TODO: something about insts?
        insts = self.read_insts(kernel_handle.insts_path)
        insts_bytes = insts.nbytes
        insts_bo = pyxrt.bo(
            self._device,
            insts_bytes,
            pyxrt.bo.cacheable,
            kernel_handle.kernel.group_id(1),
        )
        insts_bo.write(insts.view(np.uint8), 0)
        insts_bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

        start = time.time_ns()
        h = kernel_handle.kernel(3, insts_bo, insts_bytes, *buffers)
        r = h.wait()
        stop = time.time_ns()

        if fail_on_error and r != pyxrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED:
            raise HostRuntimeError(f"Kernel returned {str(r)}")

        # delete insts buffer
        del insts_bo

        return XRTKernelResult(r, stop - start, None)

    def device(self) -> Device:
        # return an instance of the device type
        return self._device_type()
