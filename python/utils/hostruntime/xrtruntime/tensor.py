# tensor.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 Advanced Micro Devices, Inc.

import numpy as np
import pyxrt as xrt

from ..tensor_class import Tensor
from aie.helpers.util import np_ndarray_type_get_shape


class XRTTensor(Tensor):
    """
    Tensor object backed by memory accessble from the 'npu' and 'cpu' devices, managed using PyXRT.

    The class provides common tensor operations such as creation,
    filling with values, and accessing data.

    """

    def __init__(
        self,
        shape_or_data,
        dtype=np.uint32,
        device="npu",
        flags=xrt.bo.host_only,
        group_id=0,
    ):
        """
        Initialize the XRTTensor.

        Args:
            shape_or_data (tuple or array-like):
                - If a tuple, creates a new tensor with the given shape and dtype.
                - If array-like, wraps the data into a tensor with optional dtype casting.
            dtype (np.dtype, optional): Data type of the tensor. Defaults to np.uint32.
            device (str, optional): Device string identifier. Defaults to 'npu'.
            flags (optional): XRT buffer object flags. Defaults to xrt.bo.host_only.
            group_id (int, optional): XRT buffer object group ID. Defaults to 0.
        """
        super().__init__(shape_or_data, dtype=dtype, device=device)
        device_index = 0
        self.xrt_device = xrt.device(device_index)

        # Extract the shape
        if isinstance(shape_or_data, tuple):
            # If this is a shape, check for it "ShapeLike"-ness using numpy ndarray types.
            np_type = np.ndarray[shape_or_data, np.dtype[dtype]]
            self._shape = np_ndarray_type_get_shape(np_type)
        elif hasattr(shape_or_data, "shape"):
            # If this is a shaped thing, we will trust it.
            self._shape = shape_or_data.shape
            np_data = shape_or_data
        else:
            # TODO(efficiency): Extra data copy here (when necessary)
            # so we can borrow verification of array-like things from numpy.
            np_data = np.array(shape_or_data, dtype=dtype, copy=False)
            self._shape = np_data.shape

        # Ideally, we use xrt::ext::bo host-only BO but there are no bindings for that currently.

        # Eventually, xrt:ext::bo uses the 0 magic number that shall be fixed in the future, so that is used as a default.
        # https://github.com/Xilinx/XRT/blob/9b114f18c4fcf4e3558291aa2d78f6d97c406365/src/runtime_src/core/common/api/xrt_bo.cpp#L1626
        self._bo = xrt.bo(
            self.xrt_device,
            int(np.prod(self._shape) * np.dtype(self.dtype).itemsize),
            flags,
            group_id,
        )

        ptr = self._bo.map()
        self._data = np.frombuffer(ptr, dtype=self.dtype).reshape(self._shape)

        if not isinstance(shape_or_data, tuple):
            np.copyto(self._data, np_data)
        else:
            self._data.fill(0)

        if self.device == "npu":
            self._sync_to_device()

    @property
    def data(self):
        """
        Get the underlying numpy array.

        Returns:
            np.ndarray: The underlying data.
        """
        return self._data

    @property
    def shape(self):
        """
        Get the shape of the tensor.

        Returns:
            tuple: The shape of the tensor.
        """
        return self._shape

    def _sync_to_device(self):
        """
        Syncs the tensor data from the host to the device memory.
        """
        return self._bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

    def _sync_from_device(self):
        """
        Syncs the tensor data from the device to the host memory.
        """
        return self._bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)

    def __del__(self):
        """
        Destructor for Tensor.

        Releases associated device memory (e.g., XRT buffer object).
        """
        if hasattr(self, "_bo"):
            del self._bo
            self._bo = None

    def buffer_object(self):
        """
        Returns the XRT buffer object associated with this tensor.

        Returns:
            buffer_object: The XRT buffer object associated with this tensor.
        """
        return self._bo
