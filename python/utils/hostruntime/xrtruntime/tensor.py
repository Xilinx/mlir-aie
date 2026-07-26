# tensor.py -*- Python -*-
#
# Copyright (C) 2025-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

import numpy as np
import pyxrt as xrt  # pyright: ignore[reportMissingImports]
from aie.helpers.util import np_ndarray_type_get_shape

from ..tensor_class import NpuBuffer, NpuTensor


class XRTBuffer(NpuBuffer):
    """An XRT allocation, and the coherence of the memory in it.

    Sub-buffer handles are derived from this allocation and cached, so a view
    binds the same handle every dispatch and the whole chain stays alive for as
    long as the buffer does.

    Every handle is derived from the root rather than from another sub-buffer.
    A sub-buffer of a sub-buffer does not compose: XRT takes the host pointer
    from the parent, so it picks up the parent's offset, but the device address
    from the underlying allocation, so it picks up only the innermost one.
    Nesting that way leaves the host reading one region while the device writes
    another.
    """

    def __init__(self, xrt_device, nbytes, flags, group_id, device):
        super().__init__(nbytes, device)
        self.xrt_device = xrt_device
        self._bo = xrt.bo(xrt_device, nbytes, flags, group_id)
        self._host = np.frombuffer(self._bo.map(), dtype=np.uint8)
        self._handles = {}

    @property
    def host_bytes(self):
        return self._host

    def binding_handle(self, offset, nbytes):
        if offset == 0 and nbytes == self.nbytes:
            return self._bo
        key = (offset, nbytes)
        handle = self._handles.get(key)
        if handle is None:
            handle = xrt.bo(self._bo, nbytes, offset)
            self._handles[key] = handle
        return handle

    def sync_to_device(self, offset, nbytes):
        self.binding_handle(offset, nbytes).sync(
            xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE
        )

    def sync_from_device(self, offset, nbytes):
        self.binding_handle(offset, nbytes).sync(
            xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE
        )


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
        nbytes = int(np.prod(self._shape) * np.dtype(self.dtype).itemsize)
        self._buffer = XRTBuffer(
            self.xrt_device, nbytes, flags, group_id, self._initial_device
        )
        self._offset_bytes = 0
        self._bo = self._buffer.binding_handle(0, nbytes)
        self._data = self._buffer.host_bytes.view(self.dtype).reshape(self._shape)

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

        Writes through this array are not reconciled; use :meth:`mutate` for a
        write that is. Kept as the unmediated handle for callers that manage
        their own synchronization.

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
        start, end = self._extent
        return self._buffer.sync_to_device(start, end - start)

    def _sync_from_device(self):
        """
        Syncs the tensor data from the device to the host memory.
        """
        start, end = self._extent
        return self._buffer.sync_from_device(start, end - start)

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
        # Same allocation, further along: the buffer holds the bytes and their
        # coherence, so a view carries nothing but where it starts and what the
        # bytes mean there.
        view._buffer = self._buffer
        absolute_offset = self.storage_offset + offset_bytes
        view._offset_bytes = absolute_offset
        view._bo = self._buffer.binding_handle(absolute_offset, nbytes)
        view._data = (
            self._buffer.host_bytes[absolute_offset : absolute_offset + nbytes]
            .view(dtype)
            .reshape(view._shape)
        )
        return view
