# tensor.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 Advanced Micro Devices, Inc.

import numpy as np
import ctypes
import pyxrt as xrt

from ..tensor import Tensor
from ..config import NPU_DEVICE


class XRTTensor(Tensor):
    """
    Tensor object backed by NPU or CPU memory, fulfilled using PyXRT runtime operations.

    The class provides commom tensor operations such as creation,
    filling with values, and accessing data.

    """

    def __init__(self, shape_or_data, dtype=np.uint32, device=NPU_DEVICE):
        super().__init__(shape_or_data, dtype=dtype, device=device)
        device_index = 0
        self.xrt_device = xrt.device(device_index)

        # Ideally, we use xrt::ext::bo host-only BO but there are no bindings for that currenty.
        # Eventually, xrt:ext::bo uses the 0 magic number that shall be fixed in the future.
        # https://github.com/Xilinx/XRT/blob/9b114f18c4fcf4e3558291aa2d78f6d97c406365/src/runtime_src/core/common/api/xrt_bo.cpp#L1626
        group_id = 0
        self.bo = xrt.bo(
            self.xrt_device,
            self.len_bytes,
            xrt.bo.host_only,
            group_id,
        )
        ptr = self.bo.map()
        self.data = np.frombuffer(ptr, dtype=self.dtype).reshape(self.shape)

        if not isinstance(shape_or_data, tuple):
            np.copyto(self.data, shape_or_data)
        else:
            self.data.fill(0)

        if self.device == NPU_DEVICE:
            self._sync_to_device()

    def _sync_to_device(self):
        return self.bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

    def _sync_from_device(self):
        return self.bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)

    def __del__(self):
        """
        Destructor for Tensor.

        Releases associated device memory (e.g., XRT buffer object).
        """
        if hasattr(self, "bo"):
            del self.bo
            self.bo = None

    def buffer_object(self):
        """
        Returns the XRT buffer object associated with this tensor.

        Returns:
           xrt.bo: The XRT buffer object associated with this tensor.
        """
        return self.bo
