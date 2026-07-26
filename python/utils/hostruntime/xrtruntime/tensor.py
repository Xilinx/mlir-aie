# tensor.py -*- Python -*-
#
# Copyright (C) 2025-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

import numpy as np
import pyxrt as xrt  # pyright: ignore[reportMissingImports]
from aie.helpers.util import np_ndarray_type_get_shape

from ..tensor_class import NpuTensor


class XRTTensor(NpuTensor):
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
        xrt_device=None,
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
            xrt_device (optional): Existing PyXRT device handle to use for BO allocation.
                When omitted, a new handle for device index 0 is opened for this tensor.
        """
        super().__init__(shape_or_data, dtype=dtype, device=device)
        self.xrt_device = xrt_device if xrt_device is not None else xrt.device(0)

        np_data = None
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
            # `np.asarray` is the NumPy-2.x-safe form of the old
            # `np.array(..., copy=False)`: avoid copy when possible, copy
            # when necessary, identical semantics on both 1.x and 2.x.
            np_data = np.asarray(shape_or_data, dtype=dtype)
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
            assert np_data is not None
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
        assert self._bo is not None
        return self._bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

    def _sync_from_device(self):
        """
        Syncs the tensor data from the device to the host memory.
        """
        assert self._bo is not None
        return self._bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)

    def __del__(self):
        """
        Destructor for NpuTensor.

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

    def _subview(self, offset_bytes, shape, dtype):
        nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
        view = type(self).__new__(type(self))
        # Set the NpuTensor contract fields without allocating a new buffer.
        NpuTensor.__init__(view, shape, dtype=dtype, device=self.device)
        view.xrt_device = self.xrt_device
        view._storage = self  # keep parent alive; shared storage
        view._shape = tuple(shape)
        # Derive from the buffer that owns the storage, at the offset accumulated
        # from the root, rather than from the immediate parent.
        #
        # A sub-buffer of a sub-buffer does not compose the way it appears to:
        # the host pointer is taken from the parent and so picks up the parent's
        # offset, but the device address is taken from the underlying allocation
        # and picks up only the innermost offset. Nesting that way leaves the
        # host reading one region while the device writes another, silently.
        # Deriving every view from the root keeps the two in agreement, whatever
        # depth the caller nests to.
        root = self.base or self
        absolute_offset = self.storage_offset + offset_bytes
        view._offset_bytes = absolute_offset
        view._bo = xrt.bo(root._bo, nbytes, absolute_offset)
        ptr = view._bo.map()
        view._data = np.frombuffer(ptr, dtype=dtype).reshape(view._shape)
        return view
