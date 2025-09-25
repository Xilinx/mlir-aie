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


class Tensor:
    """
    Tensor object backed by NPU or CPU memory.

    The class provides commom tensor operations such as creation,
    filling with values, and accessing data.

    """

    def __repr__(self):
        """
        Return a string representation of the tensor.

        Note: For NPU tensors, this method causes implicit data synchronization from device to host
        to ensure the string representation reflects the current device state.
        """
        if self.device == "npu":
            self.__sync_from_device()
        array_str = np.array2string(self.data, separator=",")
        return f"tensor({array_str}, device='{self.device}')"

    def __init__(self, shape_or_data, dtype=np.uint32, device="npu"):
        """
        Initialize the tensor.

        Parameters:
            shape_or_data (tuple or array-like):
                - If a tuple, creates a new tensor with the given shape and dtype.
                - If array-like, wraps the data into a tensor with optional dtype casting.
            dtype (np.dtype, optional): Data type of the tensor. Defaults to np.uint32.
            device (str, optional): Device string identifier (e.g., 'npu', 'cpu'). Defaults to 'npu'.
        """
        if device not in ("npu", "cpu"):
            raise ValueError(f"Unsupported device: {device}")

        self.device = device

        if isinstance(shape_or_data, tuple):
            self.shape = shape_or_data
            self.dtype = dtype
            self.data = np.zeros(self.shape, dtype=self.dtype)
        else:
            np_data = np.array(shape_or_data, dtype=dtype)
            self.shape = np_data.shape
            self.dtype = np_data.dtype
            self.data = np_data.copy()

        self.len_bytes = np.prod(self.shape) * np.dtype(self.dtype).itemsize
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
            np.copyto(self.data, np_data)
            if self.device == "npu":
                self.__sync_to_device()

    def __array__(self, dtype=None):
        """
        NumPy protocol method to convert the tensor to a NumPy array.

        This allows the tensor to be used in NumPy functions or explicitly converted via np.array(tensor).

        Parameters:
            dtype (np.dtype, optional): Desired NumPy dtype for the resulting array.
                                         If None, returns with the tensor's current dtype.

        Returns:
            np.ndarray: A NumPy array containing the tensor's data.

        Note: For NPU tensors, this method causes implicit data synchronization from device to host
        to ensure the returned array reflects the current device state.
        """
        if self.device == "npu":
            self.__sync_from_device()
        if dtype:
            return self.data.astype(dtype)
        return self.data

    def __getitem__(self, index):
        """
        Retrieves the value at a specific index in the tensor.

        Parameters:
            index (int): The index of the value to retrieve.

        Returns:
            The value at the specified index.

        Note: For NPU tensors, this method causes implicit data synchronization from device to host
        to ensure the retrieved value reflects the current device state.
        """
        if self.device == "npu":
            self.__sync_from_device()
        return self.data[index]

    def __setitem__(self, index, value):
        """
        Sets the value at a specific index in the tensor.

        Parameters:
            index (int): The index of the value to set.
            value: The new value to assign.

        Note: For NPU tensors, this method causes implicit data synchronization from device to host
        before modification and back to device after modification to ensure
        data consistency across device and host memory.
        """
        if self.device == "npu":
            self.__sync_from_device()
        self.data[index] = value
        if self.device == "npu":
            self.__sync_to_device()

    def to(self, target_device: str):
        """
        Moves the tensor to a specified target device (either "npu" or "cpu").

        Parameters:
            target_device (str): The target device ("npu" or "cpu").

        Returns:
           The tensor object on the target device.
        """
        if target_device == "npu":
            self.__sync_to_device()
            self.device = "npu"
            return self
        elif target_device == "cpu":
            self.__sync_from_device()
            self.device = "cpu"
            return self
        else:
            raise ValueError(f"Unknown device '{target_device}'")

    def __sync_to_device(self):
        """
        Syncs the tensor data from the host to the device memory.
        """
        return self.bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

    def __sync_from_device(self):
        """
        Syncs the tensor data from the device to the host memory.
        """
        return self.bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)

    def buffer_object(self):
        """
        Returns the XRT buffer object associated with this tensor.

        Returns:
           xrt.bo: The XRT buffer object associated with this tensor.
        """
        return self.bo

    def numpy(self):
        """
        Returns a NumPy view of the tensor data on host memory.

        This method ensures that data is first synchronized from the device
        (e.g., NPU) to the host before returning the array.

        Returns:
            np.ndarray: The tensor's data as a NumPy array.

        Note: For NPU tensors, this method causes implicit data synchronization from device to host
        to ensure the returned array reflects the current device state.
        """
        if self.device == "npu":
            self.__sync_from_device()
        return self.data

    def fill_(self, value):
        """
        Fills the tensor with a scalar value (in-place operation).

        Parameters:
            value: The scalar value to fill the tensor with.

        Note: For NPU tensors, this method syncs the filled data to device after modification.
        """
        self.data.fill(value)
        if self.device == "npu":
            self.__sync_to_device()

    @staticmethod
    def _ctype_from_dtype(dtype):
        """
        Converts a NumPy data type to its corresponding ctypes type.
        Parameters:
            dtype (np.dtype): A NumPy data type (or a convertible type like np.float32).

        Returns:
            A ctypes type (e.g., ctypes.c_float).
        """
        if dtype == np.uint32:
            return ctypes.c_uint32
        elif dtype == np.int32:
            return ctypes.c_int32
        elif dtype == np.float32:
            return ctypes.c_float
        else:
            raise NotImplementedError(f"Unsupported dtype: {dtype}")

    def numel(self):
        """
        Calculates the number of elements in the tensor.

        Returns:
            int: The total number of elements in the tensor.
        """
        return int(np.prod(self.shape))

    @classmethod
    def ones(cls, *size, out=None, dtype=None, device=None, **kwargs):
        """
        Returns a tensor filled with ones, with shape defined by size.

        Parameters:
            *size (int...): Shape of the tensor, passed as separate ints or a single tuple/list.

        Keyword Arguments:
            out (Tensor, optional): Optional output tensor to write into.
            dtype (np.dtype, optional): Desired dtype. Defaults to np.float32.
            device (str, optional): Target device. Defaults to "npu".
            **kwargs: Additional keyword args.

        Returns:
            Tensor: A one-filled tensor.
        """
        if len(size) == 1 and isinstance(size[0], (tuple, list)):
            shape = tuple(size[0])
        else:
            shape = tuple(size)

        dtype = dtype or np.float32
        device = device or "npu"

        if out is not None:
            if out.shape != shape or out.dtype != dtype or out.device != device:
                raise ValueError(
                    "Provided `out` tensor must match shape, dtype, and device"
                )
            out.data.fill(1)
            if device == "npu":
                out.__sync_to_device()
            return out

        t = cls(shape, dtype=dtype, device=device, **kwargs)
        t.data.fill(1)
        if device == "npu":
            t.__sync_to_device()
        return t

    @classmethod
    def zeros(cls, *size, out=None, dtype=None, device=None, **kwargs):
        """
        Returns a tensor filled with zeros, with shape defined by size.

        Parameters:
            *size (int...): Shape of the tensor, passed as separate ints or a single tuple/list.

        Keyword Arguments:
            out (Tensor, optional): Optional output tensor to write into.
            dtype (np.dtype, optional): Desired dtype. Defaults to np.float32.
            device (str, optional): Target device. Defaults to "npu".
            **kwargs: Additional keyword args.

        Returns:
            Tensor: A zero-filled tensor.
        """
        # Normalize shape
        if len(size) == 1 and isinstance(size[0], (tuple, list)):
            shape = tuple(size[0])
        else:
            shape = tuple(size)

        dtype = dtype or np.float32
        device = device or "npu"

        if out is not None:
            if out.shape != shape or out.dtype != dtype or out.device != device:
                raise ValueError(
                    "Provided `out` tensor must match shape, dtype, and device"
                )
            out.data.fill(0)
            if device == "npu":
                out.__sync_to_device()
            return out

        t = cls(shape, dtype=dtype, device=device, **kwargs)
        t.data.fill(0)
        if device == "npu":
            t.__sync_to_device()
        return t

    @classmethod
    def randint(cls, low, high, size, *, out=None, dtype=None, device=None, **kwargs):
        """
        Returns a tensor filled with random integers uniformly sampled from [low, high).

        Parameters:
            low (int): Lowest integer to be drawn (inclusive).
            high (int): One above the highest integer to be drawn (exclusive).
            size (tuple): Shape of the returned tensor.

        Keyword Arguments:
            out (Tensor, optional): Optional tensor to write the result into.
            dtype (np.dtype, optional): Data type. Defaults to np.int64.
            device (str, optional): Target device. Defaults to "npu".
            **kwargs: Additional arguments passed to the constructor.

        Returns:
            Tensor: A tensor with random integers.
        """
        dtype = dtype or np.int64
        device = device or "npu"

        data = np.random.randint(low, high, size=size, dtype=dtype)

        if out is not None:
            if out.shape != size or out.dtype != dtype or out.device != device:
                raise ValueError(
                    "Provided `out` tensor must match shape, dtype, and device"
                )
            out.data[...] = data
            if device == "npu":
                out.__sync_to_device()
            return out

        t = cls(size, dtype=dtype, device=device, **kwargs)
        t.data[...] = data
        if device == "npu":
            t.__sync_to_device()
        return t

    @classmethod
    def rand(cls, *size, out=None, dtype=None, device=None, **kwargs):
        """
        Returns a tensor filled with random numbers from a uniform distribution on [0, 1).

        Parameters:
            *size (int...): Variable number of integers or a single tuple defining the shape.

        Keyword Arguments:
            out (Tensor, optional): Output tensor to write into.
            dtype (np.dtype, optional): Desired data type. Defaults to np.float32.
            device (str, optional): Target device. Defaults to "npu".
            **kwargs: Additional arguments passed to constructor.

        Returns:
            Tensor: A tensor with random values in [0, 1).
        """

        if len(size) == 1 and isinstance(size[0], (tuple, list)):
            shape = tuple(size[0])
        else:
            shape = tuple(size)

        dtype = dtype or np.float32
        device = device or "npu"

        data = np.random.uniform(0.0, 1.0, size=shape).astype(dtype)

        if out is not None:
            if out.shape != shape or out.dtype != dtype or out.device != device:
                raise ValueError(
                    "Provided `out` tensor must match shape, dtype, and device"
                )
            out.data[...] = data
            if device == "npu":
                out.__sync_to_device()
            return out

        t = cls(shape, dtype=dtype, device=device, **kwargs)
        t.data[...] = data
        if device == "npu":
            t.__sync_to_device()
        return t

    @classmethod
    def arange(
        cls, start=0, end=None, step=1, *, out=None, dtype=None, device=None, **kwargs
    ):
        """
        Returns a 1-D tensor with values from the interval [start, end) with spacing `step`.

        Parameters:
            start (number): Start of interval. Defaults to 0.
            end (number): End of interval (exclusive). Required if only one argument is given.
            step (number): Gap between elements. Defaults to 1.

        Keyword Arguments:
            dtype (np.dtype, optional): Desired output data type. Inferred if not provided.
            out (Tensor, optional): Optional tensor to write output to (must match shape and dtype).
            device (str, optional): Target device (e.g., "npu", "cpu"). Defaults to "npu".

        Returns:
            Tensor: 1-D tensor containing the sequence.
        """

        if end is None:
            start, end = 0, start

        if dtype is None:
            if any(isinstance(x, float) for x in (start, end, step)):
                dtype = np.float32
            else:
                dtype = np.int64

        device = device or "npu"

        data = np.arange(start, end, step, dtype=dtype)

        if out is not None:
            if out.shape != (data.size,) or out.dtype != dtype or out.device != device:
                raise ValueError(
                    "Provided `out` tensor must match shape, dtype, and device"
                )
            out.data[...] = data
            if device == "npu":
                out.__sync_to_device()
            return out

        t = cls((data.size,), dtype=dtype, device=device, **kwargs)
        t.data[...] = data
        if device == "npu":
            t.__sync_to_device()
        return t

    @classmethod
    def zeros_like(cls, other, dtype=None, device=None, **kwargs):
        """
        Creates a new tensor with the same shape as `other`, filled with zeros.

        Parameters:
            other (Tensor): The reference tensor to copy shape from.
            dtype (np.dtype, optional): Data type of the new tensor. Defaults to other's dtype.
            device (str, optional): Target device. Defaults to other's device.
            **kwargs: Additional keyword arguments forwarded to the constructor.

        Returns:
            Tensor: A new zero-filled tensor with the same shape.
        """
        dtype = dtype or other.dtype
        device = device or other.device
        t = cls(other.shape, dtype=dtype, device=device, **kwargs)
        t.data.fill(0)

        if device == "npu":
            t.__sync_to_device()

        return t

    def __del__(self):
        """
        Destructor for Tensor.

        Releases associated device memory (e.g., XRT buffer object).
        """
        if hasattr(self, "bo"):
            del self.bo
            self.bo = None


def tensor(data, dtype=np.float32, device="npu"):
    """
    Creates a Tensor from array-like input with the specified dtype and device.

    Parameters:
        data (array-like): Input data (list, tuple, or  NumPy array.).
        dtype (np.dtype, optional): Desired data type. Defaults to np.float32.
        device (str, optional): Target device (e.g., "npu", "cpu"). Defaults to "npu".

    Returns:
        Tensor: A new Tensor instance.
    """
    return Tensor(data, dtype=dtype, device=device)


ones = Tensor.ones
zeros = Tensor.zeros
randint = Tensor.randint
rand = Tensor.rand
arange = Tensor.arange
zeros_like = Tensor.zeros_like
