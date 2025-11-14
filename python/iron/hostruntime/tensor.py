# tensor.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 Advanced Micro Devices, Inc.
from abc import ABC, abstractmethod
import numpy as np

from .config import CPU_DEVICE, NPU_DEVICE


class Tensor(ABC):
    """
    Tensor object backed by NPU or CPU memory.

    The class provides commom tensor operations such as creation,
    filling with values, and accessing data.

    """

    DEVICES = [CPU_DEVICE, NPU_DEVICE]
    DEFAULT_DEVICE = NPU_DEVICE
    DEFAULT_INT_DTYPE = np.int64  # torch has default int64
    DEFAULT_FLOAT_DTYPE = np.float32  # torch has default float32

    def __init__(self, shape_or_data, dtype=np.uint32, device=NPU_DEVICE):
        """
        Initialize the tensor.

        Parameters:
            shape_or_data (tuple or array-like):
                - If a tuple, creates a new tensor with the given shape and dtype.
                - If array-like, wraps the data into a tensor with optional dtype casting.
            dtype (np.dtype, optional): Data type of the tensor. Defaults to np.uint32.
            device (str, optional): Device string identifier (e.g., 'npu', 'cpu'). Defaults to 'npu'.
        """
        if device not in self.__class__.DEVICES:
            raise ValueError(f"Unsupported device: {device}")

        self.device = device

        if isinstance(shape_or_data, tuple):
            self.shape = shape_or_data
            self.dtype = dtype
        else:
            np_data = np.array(shape_or_data, dtype=dtype)
            self.shape = np_data.shape
            self.dtype = np_data.dtype

        self.len_bytes = np.prod(self.shape) * np.dtype(self.dtype).itemsize

    def __repr__(self):
        """
        Return a string representation of the tensor.

        Note: This method may implicitly trigger data synchronization to devices.
        """
        if self.device == NPU_DEVICE:
            self._sync_from_device()
        array_str = np.array2string(self.data, separator=",")
        return f"{self.__class__.__name__}({array_str}, device='{self.device}')"

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
        if self.device == NPU_DEVICE:
            self._sync_from_device()
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
        if self.device == NPU_DEVICE:
            self._sync_from_device()
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
        if self.device == NPU_DEVICE:
            self._sync_from_device()
        self.data[index] = value
        if self.device == NPU_DEVICE:
            self._sync_to_device()

    def to(self, target_device: str):
        """
        Moves the tensor to a specified target device.

        Parameters:
            target_device (str): The target device.

        Returns:
           The tensor object on the target device.
        """
        if target_device == self.device:
            # nothing to do
            pass
        elif target_device == NPU_DEVICE:
            self._sync_to_device()
            self.device = NPU_DEVICE
            return self
        elif target_device == CPU_DEVICE:
            self._sync_from_device()
            self.device = CPU_DEVICE
            return self
        else:
            raise ValueError(f"Unknown device '{target_device}'")

    @abstractmethod
    def _sync_to_device(self):
        """
        Syncs the tensor data from the host to the device memory.
        """
        ...

    @abstractmethod
    def _sync_from_device(self):
        """
        Syncs the tensor data from the device to the host memory.
        """
        ...

    @classmethod
    def __check_or_create(cls, *size, out=None, dtype=None, device=None, **kwargs):
        # Normalize shape
        if len(size) == 1 and isinstance(size[0], (tuple, list)):
            shape = tuple(size[0])
        else:
            shape = tuple(size)

        dtype = dtype or np.float32
        device = device or cls.DEFAULT_DEVICE

        t = None
        if out is not None:
            if out.shape != shape or out.dtype != dtype or out.device != device:
                raise ValueError(
                    "Provided `out` tensor must match shape, dtype, and device"
                )
            t = out
        else:
            t = cls(shape, dtype=dtype, device=device, **kwargs)
        return t

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
        if self.device == NPU_DEVICE:
            self._sync_from_device()
        return self.data

    def fill_(self, value):
        """
        Fills the tensor with a scalar value (in-place operation).

        Parameters:
            value: The scalar value to fill the tensor with.

        Note: For NPU tensors, this method syncs the filled data to device after modification.
        """
        self.data.fill(value)
        if self.device == NPU_DEVICE:
            self._sync_to_device()

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
            device (str, optional): Target device. Defaults to iron.config.NPU_DEVICE.
            **kwargs: Additional keyword args.

        Returns:
            Tensor: A one-filled tensor.
        """
        t = cls.__check_or_create(*size, out=out, dtype=dtype, device=device, **kwargs)
        t.fill_(1)
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
            device (str, optional): Target device. Defaults to iron.config.NPU_DEVICE.
            **kwargs: Additional keyword args.

        Returns:
            Tensor: A zero-filled tensor.
        """
        t = cls.__check_or_create(*size, out=out, dtype=dtype, device=device, **kwargs)
        t.fill_(0)
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
            device (str, optional): Target device. Defaults to iron.config.NPU_DEVICE.
            **kwargs: Additional arguments passed to the constructor.

        Returns:
            Tensor: A tensor with random integers.
        """
        dtype = dtype or np.int64
        device = device or cls.DEFAULT_DEVICE

        t = cls.__check_or_create(*size, out=out, dtype=dtype, device=device, **kwargs)
        t.data[:] = np.random.randint(low, high, size=size, dtype=dtype)
        if device == NPU_DEVICE:
            t._sync_to_device()
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
            device (str, optional): Target device. Defaults to iron.config.NPU_DEVICE.
            **kwargs: Additional arguments passed to constructor.

        Returns:
            Tensor: A tensor with random values in [0, 1).
        """
        dtype = dtype or np.float32
        device = device or cls.DEFAULT_DEVICE

        t = cls.__check_or_create(*size, out=out, dtype=dtype, device=device, **kwargs)
        t.data[:] = np.random.uniform(0.0, 1.0, size=t.shape).astype(dtype)
        if device == NPU_DEVICE:
            t._sync_to_device()
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
            device (str, optional): Target device. Defaults to iron.config.NPU_DEVICE.

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

        device = device or cls.DEFAULT_DEVICE

        data = np.arange(start, end, step, dtype=dtype)

        if out is not None:
            if out.shape != (data.size,) or out.dtype != dtype or out.device != device:
                raise ValueError(
                    "Provided `out` tensor must match shape, dtype, and device"
                )
            out.data[...] = data
            if device == NPU_DEVICE:
                out._sync_to_device()
            return out

        t = cls((data.size,), dtype=dtype, device=device, **kwargs)
        t.data[...] = data
        if device == NPU_DEVICE:
            t._sync_to_device()
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

        if device == NPU_DEVICE:
            t._sync_to_device()

        return t


class CPUOnlyTensor(Tensor):
    """
    This class exists primarily for testing purposes, to test tensor operations without assuming
    access to a host runtime (e.g., xrt).
    """

    DEVICES = [CPU_DEVICE]
    DEFAULT_DEVICE = CPU_DEVICE

    def __init__(self, shape_or_data, dtype=np.uint32, device=CPU_DEVICE):
        super().__init__(shape_or_data, dtype=dtype, device=device)
        if not isinstance(shape_or_data, tuple):
            self.data = np.copy(shape_or_data)
        else:
            self.data = np.zeros(shape_or_data, dtype=dtype)

    def _sync_to_device(self):
        # Nothing to do for CPU only
        pass

    def _sync_from_device(self):
        # Nothing to do for CPU only
        pass


# Set default tensor class
try:
    from .xrtruntime.tensor import XRTTensor

    DEFAULT_IRON_TENSOR_CLASS = XRTTensor
except ImportError:
    DEFAULT_IRON_TENSOR_CLASS = CPUOnlyTensor


def tensor(*args, **kwargs):
    return DEFAULT_IRON_TENSOR_CLASS(*args, **kwargs)


def ones(*args, **kwargs):
    return DEFAULT_IRON_TENSOR_CLASS.ones(*args, **kwargs)


def zeros(*args, **kwargs):
    return DEFAULT_IRON_TENSOR_CLASS.zeros(*args, **kwargs)


def randint(*args, **kwargs):
    return DEFAULT_IRON_TENSOR_CLASS.randint(*args, **kwargs)


def rand(*args, **kwargs):
    return DEFAULT_IRON_TENSOR_CLASS.rand(*args, **kwargs)


def arange(*args, **kwargs):
    return DEFAULT_IRON_TENSOR_CLASS.arange(*args, **kwargs)


def zeros_like(*args, **kwargs):
    return DEFAULT_IRON_TENSOR_CLASS.zeros_like(*args, **kwargs)


def set_iron_tensor_class(cls):
    if not issubclass(cls, Tensor):
        raise ValueError(
            f"IRON Tensors must inherit from the Tensor class but {cls} does not."
        )
    global DEFAULT_IRON_TENSOR_CLASS
    DEFAULT_IRON_TENSOR_CLASS = cls
