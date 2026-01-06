# SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
XRT-based implementation of the HostRuntime
"""
import logging
import time
from collections import OrderedDict
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
    def __init__(self, kernel, xclbin, context, insts):
        self.kernel = kernel
        self.xclbin = xclbin
        self.context = context
        self.insts = insts


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
        self._device_type_str = self._device.get_info(pyxrt.xrt_info_device.name)

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
        npu_kernel,
    ) -> XRTKernelHandle:
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
        if isinstance(insts, pyxrt.module):
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
    ) -> XRTKernelResult:
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
        if not isinstance(kernel_handle.insts, pyxrt.module):
            insts_bytes = kernel_handle.insts.nbytes
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

        # delete insts buffer
        if insts_bo:
            del insts_bo

        return XRTKernelResult(r, stop - start)

    def device(self) -> "Device":
        from aie.iron.device import NPU1, NPU2

        # TODO: this is duplicated from the LIT helpers.
        # NPU Model mappings - centralized for easy updates
        # Maps generation name to list of model strings that may appear in xrt-smi
        NPU_MODELS = {
            "npu1": ["npu1", "Phoenix"],
            "npu2": ["npu4", "Strix", "npu5", "Strix Halo", "npu6", "Krackan"],
        }

        if any([model in self._device_type_str for model in NPU_MODELS["npu1"]]):
            return NPU2()
        elif any([model in self._device_type_str for model in NPU_MODELS["npu2"]]):
            return NPU1()
        else:
            raise RuntimeError(f"Unknown device type: {self._device_type_str}")
