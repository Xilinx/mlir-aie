# tensor.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 Advanced Micro Devices, Inc.
from abc import ABC, abstractmethod
import numpy as np
import ctypes

from .config import CPU_DEVICE, NPU_DEVICE


class Tensor(ABC):
    """
    Tensor object backed by NPU or CPU memory.

    The class provides commom tensor operations such as creation,
    filling with values, and accessing data.

    """

    DEVICES = [CPU_DEVICE, NPU_DEVICE]
    DEFAULT_DEVICE = NPU_DEVICE
    DEFAULT_INT_DTYPE = np.int64
    DEFAULT_FLOAT_DTYPE = np.float64

    def __init__(self, data, dtype=None, device=None, copy=True):
        """
        Initialize the tensor.

        Parameters:
            data (array-like): data to populate the tensor with.
            dtype (np.dtype, optional): Data type of the tensor. Defaults to np.uint32.
            device (str, optional): Device string identifier (e.g., 'npu', 'cpu'). Defaults to 'npu'.
        """
        device = device or self.DEFAULT_DEVICE
        dtype = dtype or self.DEFAULT_INT_DTYPE
        if device not in self.__class__.DEVICES:
            raise ValueError(f"Unsupported device: {device}")

        self.device = device
        self.data = np.array(data, copy=copy, dtype=dtype)

    def __repr__(self):
        """
        Return a string representation of the tensor.

        Note: This method may implicitly trigger data synchronization to devices.
        """
        if self.device == NPU_DEVICE:
            self._sync_from_device()
        array_str = np.array2string(self.data, separator=",")
        return f"{self.__class__.__name__}({array_str}, device='{self.device}')"

    def __array__(self, dtype=None, copy=None):
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
        if dtype and dtype != self.data.dtype:
            return self.data.astype(dtype, copy=copy)
        if copy:
            np.copy(self.data)
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

    """
    # TODO(erika): the constructor should take data and not shape, that is the problem with this I think
    @property
    def __array_interface__(self):
        if self.device == NPU_DEVICE:
            self._sync_from_device()
        return {
            "shape": self.data.shape,
            "typestr": np.dtype(self.data.dtype).str,
            "data": (
                self.data.__array_interface__["data"][0],
                False,
            ),  # address and writable flag
            "version": 3,
        }
    """

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
    def __check(cls, size, out=None, dtype=None, device=None, **kwargs):
        if out is not None:
            if len(size) == 1 and isinstance(size[0], (tuple, list)):
                shape = tuple(size[0])
            else:
                shape = tuple(size)

            if out.shape != shape or out.dtype != dtype or out.device != device:
                raise ValueError(
                    "Provided `out` tensor must match shape, dtype, and device"
                )
            return np.asarray(out, dtype=dtype)
        return None

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
        return self.__array__()

    def fill(self, value):
        """
        Fills the tensor with a scalar value (in-place operation).

        Parameters:
            value: The scalar value to fill the tensor with.

        Note: For NPU tensors, this method syncs the filled data to device after modification.
        """
        self.data.fill(value)
        if self.device == NPU_DEVICE:
            self._sync_to_device()

    @property
    def size(self):
        """
        Calculates the number of elements in the tensor.

        Returns:
            int: The total number of elements in the tensor.
        """
        return self.data.size

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def shape(self):
        return self.data.shape

    @property
    def nbytes(self):
        return self.data.nbytes

    @classmethod
    def ones(cls, size, dtype=None, device=None, **kwargs):
        """
        Returns a tensor filled with ones, with shape defined by size.

        Parameters:
            size (int | tuple[int]): Shape of the tensor, given as an int or tuple of ints.

        Keyword Arguments:
            out (Tensor, optional): Optional output tensor to write into.
            dtype (np.dtype, optional): Desired dtype.
            device (str, optional): Target device. Defaults to iron.config.NPU_DEVCE.
            **kwargs: Additional keyword args.

        Returns:
            Tensor: A one-filled tensor.
        """
        dtype = dtype or cls.DEFAULT_FLOAT_DTYPE
        data = cls.__check(size, dtype=dtype, device=device, **kwargs)
        if data is None:
            data = np.ones(size, dtype=dtype)
        return cls(data, dtype=dtype, device=device)

    @classmethod
    def zeros(cls, size, dtype=None, device=None, **kwargs):
        """
        Returns a tensor filled with zeros, with shape defined by size.

        Parameters:
            size (int | tuple[int]): Shape of the tensor, given as an int or tuple of ints.

        Keyword Arguments:
            out (Tensor, optional): Optional output tensor to write into.
            dtype (np.dtype, optional): Desired dtype.
            device (str, optional): Target device. Defaults to iron.config.NPU_DEVCE.
            **kwargs: Additional keyword args.

        Returns:
            Tensor: A zero-filled tensor.
        """
        dtype = dtype or cls.DEFAULT_FLOAT_DTYPE
        data = cls.__check(size, dtype=dtype, device=device, **kwargs)
        if data is None:
            data = np.zeros(size, dtype=dtype)
        return cls(data, dtype=dtype, device=device)

    @classmethod
    def randint(cls, low, high, size, dtype=None, device=None, **kwargs):
        """
        Returns a tensor filled with random integers uniformly sampled from [low, high).

        Parameters:
            low (int): Lowest integer to be drawn (inclusive).
            high (int): One above the highest integer to be drawn (exclusive).
            size (tuple): Shape of the returned tensor.

        Keyword Arguments:
            out (Tensor, optional): Optional tensor to write the result into.
            dtype (np.dtype, optional): Data type. Defaults to np.int64.
            device (str, optional): Target device. Defaults to iron.config.NPU_DEVCE.
            **kwargs: Additional arguments passed to the constructor.

        Returns:
            Tensor: A tensor with random integers.
        """
        data = cls.__check(size, dtype=dtype, device=device, **kwargs)
        if data is None:
            data = np.random.randint(low, high, size=size, dtype=dtype)
        return cls(data, dtype=dtype, device=device, copy=False)

    @classmethod
    def rand(cls, size, out=None, dtype=None, device=None, **kwargs):
        """
        Returns a tensor filled with random numbers from a uniform distribution on [0, 1).

        Parameters:
            size (int | tuple[int]): Shape of the tensor, given as an int or tuple of ints.

        Keyword Arguments:
            out (Tensor, optional): Output tensor to write into.
            dtype (np.dtype, optional): Desired data type. Defaults to np.float32.
            device (str, optional): Target device. Defaults to iron.config.NPU_DEVCE.
            **kwargs: Additional arguments passed to constructor.

        Returns:
            Tensor: A tensor with random values in [0, 1).
        """
        dtype = dtype or cls.DEFAULT_FLOAT_DTYPE
        data = cls.__check(size, dtype=dtype, device=device, **kwargs)
        if data is None:
            data = np.random.uniform(0.0, 1.0, size=size).astype(dtype)
        return cls(data, dtype=dtype, device=device, copy=False)

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
            device (str, optional): Target device. Defaults to iron.config.NPU_DEVCE.

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
        t = cls(np.zeros(other.shape), dtype=dtype, device=device, **kwargs)
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

    def __init__(self, data, dtype=None, device=None, copy=True):
        device = device or self.DEFAULT_DEVICE
        super().__init__(data, dtype=dtype, device=device)

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
