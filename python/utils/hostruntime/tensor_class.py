# tensor.py -*- Python -*-
#
# Copyright (C) 2025-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
import os
from abc import ABC, abstractmethod
from functools import cached_property

import numpy as np
import numpy.typing as npt


# Smallest unit of host/device cache coherence, in bytes.
#
# Host<->device synchronization is not byte-granular: on a non-cache-coherent
# part the shim maintains whole cache lines (it walks
# ``sysconf(_SC_LEVEL1_DCACHE_LINESIZE)``-sized steps and the CPU flushes the
# entire line containing an address), so a sync for one region unavoidably acts
# on every byte sharing its first and last line.  Sub-region views must
# therefore not share a line, or one view's sync can write a neighbor's stale
# host copy over data the device just produced.
#
# Detected once, floored at 64 so a bogus or missing report cannot weaken the
# check.  This mirrors what the runtime actually does rather than the page size:
# pages are the unit of address translation and protection, lines are the unit
# of coherence.
def _detect_coherence_granule(default=64):
    try:
        with open(
            "/sys/devices/system/cpu/cpu0/cache/index0/coherency_line_size"
        ) as fp:
            return max(int(fp.read().strip()), default)
    except (OSError, ValueError):
        pass
    try:
        return max(os.sysconf("SC_LEVEL1_DCACHE_LINESIZE"), default)
    except (ValueError, OSError, AttributeError):
        return default


COHERENCE_GRANULE = _detect_coherence_granule()

# Mapping from ml_dtypes (non-native numpy) types to their torch equivalents.
# Native numpy dtypes (float32, int32, …) are handled directly by torch.from_numpy
# and do not need an entry here.
# Populated lazily at first use to avoid importing torch/ml_dtypes at module load.
_ML_DTYPE_TO_TORCH: dict | None = None


def _ml_dtype_to_torch_map():
    global _ML_DTYPE_TO_TORCH
    if _ML_DTYPE_TO_TORCH is None:
        import ml_dtypes
        import torch  # pyright: ignore[reportMissingImports]

        _candidates = {
            ml_dtypes.bfloat16: torch.bfloat16,
        }
        for attr in (
            "float8_e4m3fn",
            "float8_e5m2",
            "float8_e4m3fnuz",
            "float8_e5m2fnuz",
        ):
            ml_dt = getattr(ml_dtypes, attr, None)
            torch_dt = getattr(torch, attr, None)
            if ml_dt is not None and torch_dt is not None:
                _candidates[ml_dt] = torch_dt
        _ML_DTYPE_TO_TORCH = {
            np.dtype(ml_dt): torch_dt for ml_dt, torch_dt in _candidates.items()
        }
    return _ML_DTYPE_TO_TORCH


# Same-width unsigned integer dtype for the ND reinterpret-view trick.
_UINT_VIEW_DTYPE = {
    1: np.uint8,
    2: np.uint16,
    4: np.uint32,
    8: np.uint64,
}


def _array_to_torch(array: np.ndarray):
    """
    Convert a numpy array to a torch tensor, zero-copy.

    For native numpy dtypes (float32, float16, int32, …) torch.from_numpy is used directly
    (fastest path for these types).

    For ml_dtypes types (bfloat16, float8_*) that torch cannot consume via from_numpy:
    reinterpret as a same-width unsigned integer numpy view, wrap with from_numpy,
    then view as the target torch dtype.  This is guaranteed zero-copy for all ranks.

    Raises:
        ImportError: If torch is not installed.
    """
    # _ml_dtype_to_torch_map() imports torch (raising ImportError with a helpful message
    # if absent) and returns the ml_dtype -> torch dtype mapping.
    torch_dtype = _ml_dtype_to_torch_map().get(array.dtype)
    import torch  # pyright: ignore[reportMissingImports]  # already imported by _ml_dtype_to_torch_map(); cached by Python

    if torch_dtype is None:
        # Native numpy dtype: torch.from_numpy handles it directly and fastest.
        return torch.from_numpy(array)

    # ml_dtype: reinterpret memory as a same-width uint, then view as the torch dtype.
    uint_dtype = _UINT_VIEW_DTYPE[array.dtype.itemsize]
    return torch.from_numpy(array.view(uint_dtype)).view(torch_dtype)


class NpuTensor(ABC):
    """
    A host-mapped, device-resident buffer of fixed shape and dtype.

    This is a buffer with a residency state machine, not a general array. Its
    invariant is host/device coherence: the host and the device each hold a view
    of the same storage, and the two are reconciled only at the points this class
    defines. Everything else it offers (indexing, filling, the numpy and torch
    bridges, :meth:`subview`) exists to keep that reconciliation correct while
    still letting callers treat the buffer as data.

    The invariant in full:

    * Writes through the declared paths (:meth:`__setitem__`, :meth:`fill_`, and
      the factories) resynchronize. A write through the raw ``data`` array does
      not, and is the one way to leave host and device disagreeing.
    * :meth:`to` moves residency and is a no-op when the buffer is already on the
      target device, so a caller that has written through a declared path never
      pays for a redundant transfer, and a caller that has bypassed one gets no
      transfer at all.
    * Reconciliation is not byte-granular. It acts on whole cache lines, which is
      why :meth:`subview` requires its regions to be granule-aligned.

    Subclasses supply the storage and the two transfer primitives; the invariant
    itself lives here so every backend states it the same way.

    Named for the role rather than the mechanism: ``XRTTensor`` and ``HRXTensor``
    name what implements the buffer, ``NpuTensor`` names what it is. ``Tensor``
    remains as an alias for existing callers.
    """

    DEVICES = ["cpu", "npu"]
    DEFAULT_DEVICE = "npu"
    DEFAULT_INT_DTYPE = np.int64  # torch has default int64
    DEFAULT_FLOAT_DTYPE = np.float32  # torch has default float32

    # Alignment :meth:`subview` requires of a sub-region, in bytes. A backend
    # whose host/device reconciliation has a different granularity may override
    # it; see :data:`COHERENCE_GRANULE`.
    _coherence_granule = COHERENCE_GRANULE

    def __init__(self, shape_or_data, dtype: npt.DTypeLike = np.uint32, device="npu"):
        """
        Initialize the tensor.

        Args:
            shape_or_data (tuple or array-like):
                - If a tuple, creates a new tensor with the given shape and dtype.
                - If array-like, wraps the data into a tensor with optional dtype casting.
            dtype (np.dtype, optional): Data type of the tensor. Defaults to np.uint32.
            device (str, optional): Device string identifier (e.g., 'npu', 'cpu'). Defaults to 'npu'.
        """
        if device not in self.__class__.DEVICES:
            raise ValueError(f"Unsupported device: {device}")
        self.device = device
        self.dtype = dtype

    @property
    @abstractmethod
    def data(self) -> np.ndarray:
        """
        Subclasses must implement a data property.

        Returns:
            np.ndarray: The underlying data of the tensor.
        """
        pass

    @property
    @abstractmethod
    def shape(self) -> tuple[int, ...]:
        """
        Subclasses must implement a shape property.

        Returns:
            tuple: The shape of the tensor.
        """
        pass

    def __repr__(self):
        """
        Return a string representation of the tensor.

        Note: This method may implicitly trigger data synchronization to devices.
        """
        if self.device == "npu":
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
        if self.device == "npu":
            self._sync_from_device()
        if dtype:
            return self.data.astype(dtype)
        return self.data

    def __getitem__(self, index):
        """
        Retrieves the value at a specific index in the tensor.

        Args:
            index (int): The index of the value to retrieve.

        Returns:
            The value at the specified index.

        Note: For NPU tensors, this method causes implicit data synchronization from device to host
        to ensure the retrieved value reflects the current device state.
        """
        if self.device == "npu":
            self._sync_from_device()
        return self.data[index]

    def __setitem__(self, index, value):
        """
        Sets the value at a specific index in the tensor.

        Args:
            index (int): The index of the value to set.
            value: The new value to assign.

        Note: For NPU tensors, this method causes implicit data synchronization from device to host
        before modification and back to device after modification to ensure
        data consistency across device and host memory.
        """
        if self.device == "npu":
            self._sync_from_device()
        self.data[index] = value
        if self.device == "npu":
            self._sync_to_device()

    def __len__(self):
        """
        Return the length of the tensor.

        Returns:
            int: The length of the tensor (size of the first dimension).

        Raises:
            TypeError: If the tensor is 0-dimensional.
        """
        if self.data.ndim == 0:
            raise TypeError("len() of a 0-d tensor")
        return self.shape[0]

    @cached_property
    def nbytes(self) -> int:
        """
        Number of bytes consumed by elements in the tensor
        """
        return self.numel() * self.element_size

    @cached_property
    def element_size(self) -> int:
        """
        Number of bytes per element
        """
        return np.dtype(self.dtype).itemsize

    def to(self, target_device: str):
        """
        Moves the tensor to a specified target device.

        Args:
            target_device (str): The target device.

        Returns:
           The tensor object on the target device.
        """
        if target_device == self.device:
            # nothing to do
            pass
        elif target_device == "npu":
            self._sync_to_device()
            self.device = "npu"
        elif target_device == "cpu":
            self._sync_from_device()
            self.device = "cpu"
        else:
            raise ValueError(f"Unknown device '{target_device}'")
        return self

    def subview(self, offset, shape, dtype=None):
        """
        Return a tensor viewing a sub-region of this tensor's underlying storage.

        The returned tensor shares this tensor's buffer (no new allocation, no
        copy), holds a reference to this tensor so the storage outlives the view,
        and synchronizes its own slice. It is a plain tensor of the same backend
        class, not a distinct type.

        The region must be aligned to :data:`COHERENCE_GRANULE`, because host and
        device are reconciled a cache line at a time, not a byte at a time. Two
        views sharing a line are not independent: synchronizing one acts on the
        other's bytes in that line, so a view whose host copy is stale can be
        written back over data the device just produced in its neighbor. The
        check makes that unrepresentable instead of leaving it to callers to
        remember. It is enforced for every backend, including the CPU-only one
        that has no coherence concern of its own, so a layout validated against
        the test backend stays valid on a device.

        A view may end anywhere if it ends where this tensor ends: its last line
        is shared with no sibling, so it is no worse than synchronizing the whole
        buffer. This also keeps a whole-buffer view (``offset=0``) legal for a
        tensor whose own size is not a multiple of the granule.

        Note that alignment bounds the damage but does not make a sync exactly
        slice-scoped: some driver paths (an imported buffer, or one with no
        kernel mapping) maintain the whole buffer regardless of the requested
        range. Correctness must not depend on a sync being narrow, only on views
        not sharing a coherence granule.

        Args:
            offset: Start of the region, in elements of this tensor's dtype.
            shape: Logical shape of the view.
            dtype (np.dtype, optional): dtype to interpret the region as.
                Defaults to this tensor's dtype.

        Returns:
            NpuTensor: A view sharing this tensor's storage.

        Raises:
            ValueError: If the region falls outside this tensor's buffer, or is
                not aligned to :data:`COHERENCE_GRANULE`.
        """
        view_dtype = np.dtype(dtype) if dtype is not None else np.dtype(self.dtype)
        offset_bytes = int(offset) * np.dtype(self.dtype).itemsize
        nbytes = int(np.prod(shape)) * view_dtype.itemsize
        if offset_bytes < 0 or offset_bytes + nbytes > self.nbytes:
            raise ValueError(
                f"subview(offset={offset}, shape={tuple(shape)}, dtype={view_dtype}) "
                f"is out of bounds for a buffer of {self.nbytes} bytes"
            )
        granule = self._coherence_granule
        ends_at_parent_end = offset_bytes + nbytes == self.nbytes
        if offset_bytes % granule or (nbytes % granule and not ends_at_parent_end):
            raise ValueError(
                f"subview(offset={offset}, shape={tuple(shape)}, dtype={view_dtype}) "
                f"spans bytes [{offset_bytes}, {offset_bytes + nbytes}) of this "
                f"buffer, which is not aligned to the {granule}-byte coherence "
                f"granule. Host and device are reconciled a cache line at a time, "
                f"so a view sharing a line with a neighbor cannot be synchronized "
                f"independently of it. Pad the region layout so each view starts "
                f"at a multiple of {granule} bytes and (unless it ends where this "
                f"buffer ends) is a multiple of {granule} bytes long."
            )
        return self._subview(offset_bytes, tuple(shape), view_dtype)

    @abstractmethod
    def _sync_to_device(self):
        """
        Syncs the tensor data from the host to the device memory.

        This method should be implemented by subclasses to handle device-specific synchronization.
        """
        ...

    @abstractmethod
    def _sync_from_device(self):
        """
        Syncs the tensor data from the device to the host memory.

        This method should be implemented by subclasses to handle device-specific synchronization.
        """
        ...

    def _subview(self, offset_bytes, shape, dtype):
        """
        Backend hook for :meth:`subview`.

        Build and return a tensor of the same backend class that shares this
        tensor's underlying storage starting at ``offset_bytes`` with the given
        ``shape`` and ``dtype``. Implementations must not allocate or copy, must
        record this tensor as the storage owner so it outlives the view, and
        must leave residency per-view (the returned view syncs its own slice).

        Kept concrete (not abstract) so a backend that does not implement
        sub-region views is not forced to; the default raises
        ``NotImplementedError``.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support subview()")

    @classmethod
    def __check_or_create(cls, *size, out=None, dtype=None, device=None, **kwargs):
        """
        Internal helper to check an output tensor or create a new one.

        Args:
            *size: Shape of the tensor.
            out (NpuTensor, optional): Output tensor to check.
            dtype (np.dtype, optional): Data type.
            device (str, optional): Device.
            **kwargs: Additional arguments for tensor creation.

        Returns:
            NpuTensor: The checked or created tensor.

        Raises:
            ValueError: If `out` tensor does not match shape, dtype, or device.
        """
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
        if self.device == "npu":
            self._sync_from_device()
        return self.data

    def to_torch(self):
        """
        Returns a torch tensor sharing the data in this tensor if possible.

        Syncs from device first if the tensor is on the NPU.

        Returns:
            torch.Tensor: A torch tensor containing the data.

        Raises:
            ImportError: If torch is not installed.
        """
        return _array_to_torch(self.numpy())

    def torch_view(self):
        """
        Returns a torch tensor sharing this buffer's host memory without syncing from device.

        Unlike to_torch(), this does NOT sync from the NPU first. Marks the buffer as
        CPU-resident so that a subsequent .to("npu") call (or the NPU operator's implicit
        sync) will push the written data to device. Use this on write paths where the
        caller is about to overwrite the buffer contents.

        Returns:
            torch.Tensor: A zero-copy torch tensor view of the host-side buffer.

        Raises:
            ImportError: If torch is not installed.
        """
        self.device = "cpu"  # mark dirty so next to("npu") will actually sync
        return _array_to_torch(self.data)

    @classmethod
    def from_torch(cls, torch_tensor, device=None, **kwargs):
        """
        Returns a tensor with a copy of the data in the torch_tensor.

        Args:
            torch_tensor (torch.Tensor): The source torch tensor.
            device (str, optional): The target device. Defaults to None.
            **kwargs: Additional arguments for tensor creation.

        Returns:
            NpuTensor: A new tensor containing the data from the torch tensor.

        Raises:
            ImportError: If torch is not installed.
        """
        import torch  # pyright: ignore[reportMissingImports]
        from ml_dtypes import bfloat16

        # Detach (to drop grad) and ensure on CPU
        t = torch_tensor.detach()
        if t.device.type != "cpu":
            t = t.cpu()
        # Ensure contiguous for safe view operations
        if not t.is_contiguous():
            t = t.contiguous()

        if t.dtype == torch.bfloat16:
            # View the same memory as int16, then as NumPy bfloat16
            # This avoids numeric conversion and extra passes over memory.
            u16_np = t.view(torch.uint16).numpy()  # shares memory
            np_array = u16_np.view(bfloat16)  # reinterpret
        else:
            np_array = t.numpy()

        return cls(
            np_array,
            dtype=np_array.dtype,
            device=device or cls.DEFAULT_DEVICE,
            **kwargs,
        )

    def fill_(self, value):
        """
        Fills the tensor with a scalar value (in-place operation).

        Args:
            value: The scalar value to fill the tensor with.

        Note: For NPU tensors, this method syncs the filled data to device after modification.
        """
        self.data.fill(value)
        if self.device == "npu":
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

        Args:
            *size (int...): Shape of the tensor, passed as separate ints or a single tuple/list.
            out (NpuTensor, optional): Optional output tensor to write into.
            dtype (np.dtype, optional): Desired dtype. Defaults to np.float32.
            device (str, optional): Target device. Defaults to 'npu'.
            **kwargs: Additional keyword args.

        Returns:
            NpuTensor: A one-filled tensor.
        """
        t = cls.__check_or_create(*size, out=out, dtype=dtype, device=device, **kwargs)
        t.fill_(1)
        return t

    @classmethod
    def zeros(cls, *size, out=None, dtype=None, device=None, **kwargs):
        """
        Returns a tensor filled with zeros, with shape defined by size.

        Args:
            *size (int...): Shape of the tensor, passed as separate ints or a single tuple/list.
            out (NpuTensor, optional): Optional output tensor to write into.
            dtype (np.dtype, optional): Desired dtype. Defaults to np.float32.
            device (str, optional): Target device. Defaults to 'npu'.
            **kwargs: Additional keyword args.

        Returns:
            NpuTensor: A zero-filled tensor.
        """
        t = cls.__check_or_create(*size, out=out, dtype=dtype, device=device, **kwargs)
        t.fill_(0)
        return t

    @classmethod
    def full(cls, size, fill_value, *, out=None, dtype=None, device=None, **kwargs):
        """
        Returns a tensor of shape `size` filled with `fill_value`.

        Args:
            size (int or tuple/list of int): Shape of the returned tensor.
            fill_value (scalar): Value to fill the tensor with.
            out (NpuTensor, optional): Optional output tensor to write into.
            dtype (np.dtype, optional): Desired dtype. Defaults to np.float32.
            device (str, optional): Target device. Defaults to 'npu'.
            **kwargs: Additional keyword args.

        Returns:
            NpuTensor: A tensor filled with `fill_value`.
        """
        t = cls.__check_or_create(size, out=out, dtype=dtype, device=device, **kwargs)
        t.fill_(fill_value)
        return t

    @classmethod
    def randint(
        cls,
        low,
        high,
        size,
        *,
        out=None,
        dtype=None,
        device=None,
        generator=None,
        **kwargs,
    ):
        """
        Returns a tensor filled with random integers uniformly sampled from [low, high).

        Args:
            low (int): Lowest integer to be drawn (inclusive).
            high (int): One above the highest integer to be drawn (exclusive).
            size (tuple): Shape of the returned tensor.
            out (NpuTensor, optional): Optional tensor to write the result into.
            dtype (np.dtype, optional): Data type. Defaults to np.int64.
            device (str, optional): Target device. Defaults to 'npu'.
            generator (np.random.Generator, optional): Source RNG for reproducibility.
                If None, uses np.random module-level state.
            **kwargs: Additional arguments passed to the constructor.

        Returns:
            NpuTensor: A tensor with random integers.
        """
        dtype = dtype or np.int64
        device = device or cls.DEFAULT_DEVICE

        t = cls.__check_or_create(size, out=out, dtype=dtype, device=device, **kwargs)
        if generator is not None:
            random_val = generator.integers(low, high, size=size, dtype=dtype)
        else:
            random_val = np.random.randint(low, high, size=size, dtype=dtype)
        if size == ():
            t.data.fill(random_val)
        else:
            t.data[:] = random_val
        if device == "npu":
            t._sync_to_device()
        return t

    @classmethod
    def rand(cls, *size, out=None, dtype=None, device=None, generator=None, **kwargs):
        """
        Returns a tensor filled with random numbers from a uniform distribution on [0, 1).

        Args:
            *size (int...): Variable number of integers or a single tuple defining the shape.
            out (NpuTensor, optional): Output tensor to write into.
            dtype (np.dtype, optional): Desired data type. Defaults to np.float32.
            device (str, optional): Target device. Defaults to 'npu'.
            generator (np.random.Generator, optional): Source RNG for reproducibility.
                If None, uses np.random module-level state.
            **kwargs: Additional arguments passed to constructor.

        Returns:
            NpuTensor: A tensor with random values in [0, 1).
        """
        if not size:
            raise ValueError("rand() received no arguments")
        dtype = dtype or np.float32
        device = device or cls.DEFAULT_DEVICE

        t = cls.__check_or_create(*size, out=out, dtype=dtype, device=device, **kwargs)
        if generator is not None:
            random_val = generator.uniform(0.0, 1.0, size=t.shape).astype(dtype)
        else:
            random_val = np.random.uniform(0.0, 1.0, size=t.shape).astype(dtype)
        # Ensure values are < 1.0 for low-precision types
        is_bfloat16 = False
        try:
            from ml_dtypes import bfloat16

            if dtype == bfloat16:
                is_bfloat16 = True
        except ImportError:
            pass

        if np.issubdtype(dtype, np.floating) or is_bfloat16:
            max_val = np.nextafter(dtype(1.0), dtype(0.0))
            random_val = np.clip(random_val, 0.0, max_val)

        if t.shape == ():
            t.data.fill(random_val)
        else:
            t.data[:] = random_val
        if device == "npu":
            t._sync_to_device()
        return t

    @classmethod
    def arange(
        cls,
        start=0,
        end=None,
        step=1,
        *,
        shape=None,
        out=None,
        dtype=None,
        device=None,
        **kwargs,
    ):
        """
        Returns a tensor with values from the interval [start, end) with spacing `step`.

        Args:
            start (number): Start of interval. Defaults to 0.
            end (number): End of interval (exclusive). Required if only one argument is given.
            step (number): Gap between elements. Defaults to 1.
            shape (tuple, optional): If given, reshape the 1-D sequence to this shape.
                `prod(shape)` must equal the length of the generated range.
            dtype (np.dtype, optional): Desired output data type. Inferred if not provided.
            out (NpuTensor, optional): Optional tensor to write output to (must match shape and dtype).
            device (str, optional): Target device. Defaults to 'npu'.

        Returns:
            NpuTensor: A tensor containing the sequence (1-D by default, or `shape` if given).
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

        if shape is not None:
            shape = tuple(shape)
            if int(np.prod(shape)) != data.size:
                raise ValueError(
                    f"iron.arange: shape={shape} (prod={int(np.prod(shape))}) does "
                    f"not match generated range size {data.size}"
                )
            data = data.reshape(shape)
        else:
            shape = (data.size,)

        if out is not None:
            if out.shape != shape or out.dtype != dtype or out.device != device:
                raise ValueError(
                    "Provided `out` tensor must match shape, dtype, and device"
                )
            out.data[...] = data
            if device == "npu":
                out._sync_to_device()
            return out

        t = cls(shape, dtype=dtype, device=device, **kwargs)
        t.data[...] = data
        if device == "npu":
            t._sync_to_device()
        return t

    @classmethod
    def zeros_like(cls, other, dtype=None, device=None, **kwargs):
        """
        Creates a new tensor with the same shape as `other`, filled with zeros.

        Args:
            other (NpuTensor): The reference tensor to copy shape from.
            dtype (np.dtype, optional): Data type of the new tensor. Defaults to other's dtype.
            device (str, optional): Target device. Defaults to other's device.
            **kwargs: Additional keyword arguments forwarded to the constructor.

        Returns:
            NpuTensor: A new zero-filled tensor with the same shape.
        """
        dtype = dtype or other.dtype
        device = device or other.device
        t = cls(other.shape, dtype=dtype, device=device, **kwargs)
        t.data.fill(0)

        if device == "npu":
            t._sync_to_device()

        return t


class CPUOnlyTensor(NpuTensor):
    """
    This class exists primarily for testing purposes, to test tensor operations without assuming
    access to a host runtime (e.g., xrt).
    """

    DEVICES = ["cpu"]
    DEFAULT_DEVICE = "cpu"

    def __init__(self, shape_or_data, dtype: npt.DTypeLike = np.uint32, device="cpu"):
        """
        Initialize the CPUOnlyTensor.

        Args:
            shape_or_data (tuple or array-like):
                - If a tuple, creates a new tensor with the given shape and dtype.
                - If array-like, wraps the data into a tensor with optional dtype casting.
            dtype (np.dtype, optional): Data type of the tensor. Defaults to np.uint32.
            device (str, optional): Device string identifier. Defaults to 'cpu'.
        """
        super().__init__(shape_or_data, dtype=dtype, device=device)
        if not isinstance(shape_or_data, tuple):
            self._data = np.array(shape_or_data, dtype=dtype)
        else:
            self._data = np.zeros(shape_or_data, dtype=dtype)
        self._shape = self._data.shape

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
        For CPUOnlyTensor, this is a no-op.
        """
        # Nothing to do for CPU only
        pass

    def _sync_from_device(self):
        """
        Syncs the tensor data from the device to the host memory.
        For CPUOnlyTensor, this is a no-op.
        """
        # Nothing to do for CPU only
        pass

    def _subview(self, offset_bytes, shape, dtype):
        nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
        view = CPUOnlyTensor.__new__(CPUOnlyTensor)
        # Set the NpuTensor contract fields without allocating a new array.
        NpuTensor.__init__(view, shape, dtype=dtype, device=self.device)
        view._storage = self  # keep parent alive; shared storage
        view._shape = tuple(shape)
        # A numpy view over the same bytes (zero-copy) so writes are shared.
        flat = self._data.reshape(-1).view(np.uint8)
        view._data = (
            flat[offset_bytes : offset_bytes + nbytes].view(dtype).reshape(view._shape)
        )
        return view


# The former name of NpuTensor, kept so existing callers and subclasses outside
# this repository keep working. New code should use NpuTensor.
Tensor = NpuTensor
