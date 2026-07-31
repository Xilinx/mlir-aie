# tensor.py -*- Python -*-
#
# Copyright (C) 2025-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""The host tensor: a shaped, typed view over a coherence-managed allocation.

The allocation itself, its residency bookkeeping, and the torch bridge live in
:mod:`buffer`, :mod:`coherence` and :mod:`torch_interop`. Each is re-exported
here, so ``from .tensor_class import Storage`` and friends keep resolving.
"""

import math
import operator
from abc import ABC, abstractmethod
from functools import cached_property

import numpy as np
import numpy.typing as npt

from .buffer import HostOnlyTransport as HostOnlyTransport
from .buffer import Storage as Storage
from .buffer import Transport as Transport
from .buffer import WholeExtentTransport as WholeExtentTransport
from .coherence import COHERENCE_GRANULE as COHERENCE_GRANULE
from .coherence import _CoherenceMap as _CoherenceMap
from .coherence import _detect_coherence_granule as _detect_coherence_granule
from .coherence import _read_sysfs_line_size as _read_sysfs_line_size
from .torch_interop import _array_to_torch as _array_to_torch
from .torch_interop import _ml_dtype_to_torch_map as _ml_dtype_to_torch_map


def _as_shape(shape):
    """Normalize and validate a shape, rejecting what would silently misbehave.

    A bare integer is accepted as a length, matching the factories in this
    module. Every extent must be a true non-negative integer: a negative one
    would pass the byte-count checks (it makes the region look smaller than it
    is) and then be read as a Python negative index by the backends, producing a
    view of the wrong size at the wrong place.
    """
    if isinstance(shape, (str, bytes)):
        raise TypeError(f"shape must be an integer or a sequence, got {shape!r}")
    try:
        extents = (operator.index(shape),)
    except TypeError:
        try:
            extents = tuple(operator.index(extent) for extent in shape)
        except TypeError:
            raise TypeError(
                f"shape must be an integer or a sequence of integers, got {shape!r}"
            ) from None
    for extent in extents:
        if extent < 0:
            raise ValueError(f"shape extents must not be negative, got {extents}")
    return extents


class _WriteBorrow:
    """The array a write scope hands out, and the record it leaves behind.

    A class rather than a generator: this is entered once per region written, so
    a buffer filled window by window pays for it as many times as it has
    windows, and the generator machinery costs more than the work it wraps.
    """

    __slots__ = ("_tensor", "_array", "_reconcile")

    # Bound on entry rather than in the constructor, so leaving the scope
    # without having entered it raises instead of quietly recording a write
    # that never happened.
    _array: np.ndarray

    def __init__(self, tensor, reconcile):
        self._tensor = tensor
        self._reconcile = reconcile

    def __enter__(self):
        tensor = self._tensor
        if self._reconcile:
            # Pull without claiming. to("cpu") would do both, and the claim
            # outlives a body that raises: a failed write would leave residency
            # moved, so the next transfer flushes a region nobody wrote and a
            # failed mutate disagrees with a failed overwrite about what it
            # left behind. Recording the write is __exit__'s job, and it only
            # does it when the body completed.
            tensor._reconcile_for_read()
        array = tensor.data[...]
        self._array = array
        return array

    def __exit__(self, *exc_info):
        # Retire the borrow either way: the scope is over, and a reference kept
        # past it must not stay writable whether or not the body succeeded.
        self._array.flags.writeable = False
        if exc_info[0] is not None:
            # The body raised, so the write it promised did not happen, and
            # recording it would be a claim about bytes the host may never have
            # held: an overwrite() scope skips the reconcile on entry, so on this
            # path the device's copy was never pulled, and marking the region
            # host-written would serve it to the next read and push it over the
            # device's at the next transfer. Leaving the state alone loses a
            # partial write, which is what a failed write should look like.
            return False
        tensor = self._tensor
        start, end = tensor._extent
        tensor._coherence().set(start, end, "cpu")
        return False


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
      the factories) are reconciled: the factories transfer as they construct,
      and the in-place writes record the region so the next :meth:`to` sends it.
      A write through the raw ``data`` array does neither, and is the one way to
      leave host and device disagreeing.
    * A method that spans regions in different states reconciles per region.
      Asking a whole tensor where it lives collapses a mixed extent to one
      answer, which is the right answer for :meth:`to` and the wrong one to
      transfer by.
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

    # Alignment :meth:`subview` requires of a sub-region, in bytes. Left unset
    # so the module-level default is resolved per call rather than frozen into
    # the class at import; a backend whose host/device reconciliation has a
    # different granularity sets its own. See :data:`COHERENCE_GRANULE`.
    _coherence_granule: int | None = None

    # Set on views by the backend hook. Declared here rather than conjured onto
    # the instance so that every tensor has them, they are part of the contract
    # a backend implements against, and the type checker can see them.
    _parent: "NpuTensor | None" = None
    _offset_bytes: int = 0
    # Same, for the two the class fills in lazily rather than the backend. Held
    # as declared attributes because writing through ``__dict__`` to cache them
    # is an index into a read-only mapping as far as a type checker is
    # concerned, and the attribute is what was meant in any case.
    _storage: "Storage | None" = None
    _extent_cache: "tuple[int, int] | None" = None
    # Bound on first use. Held directly rather than reached through the buffer
    # because the residency query is on the path a dispatch takes for every
    # argument, and an attribute read is the whole fast path.
    _coherence_ref: "_CoherenceMap | None" = None

    @classmethod
    def _resolve_coherence_granule(cls):
        """Alignment :meth:`subview` enforces for this backend, in bytes."""
        return (
            COHERENCE_GRANULE
            if cls._coherence_granule is None
            else cls._coherence_granule
        )

    @property
    def _owner(self):
        """The buffer that owns the storage (this one, if it owns it)."""
        return self.base or self

    @property
    def storage(self) -> "Storage":
        """The allocation this tensor is a view of.

        A backend that still owns its bytes directly is wrapped on first use in
        a storage whose transport reconciles whole-extent, so every tensor has
        one whether or not its backend has adopted the split.
        """
        existing = self._storage
        if existing is None:
            existing = Storage(
                WholeExtentTransport(self),
                self.nbytes,
                self._initial_device,
                type(self)._resolve_coherence_granule(),
            )
            self._storage = existing
        return existing

    def _coherence(self) -> "_CoherenceMap":
        coherence = self._coherence_ref
        if coherence is None:
            coherence = self.storage.coherence
            self._coherence_ref = coherence
        return coherence

    @property
    def _extent(self):
        """Byte range this tensor covers in its buffer.

        Cached: shape, dtype and offset are fixed once the tensor exists, and
        this is read on every residency query, which a dispatch does per
        argument.
        """
        cached = self._extent_cache
        if cached is not None:
            return cached
        start = self.storage_offset
        size = math.prod(self.shape) * np.dtype(self.dtype).itemsize
        extent = (start, start + size)
        self._extent_cache = extent
        return extent

    @property
    def device(self):
        """Which agent holds the authoritative copy of *this tensor's bytes*.

        Read from the allocation, so two tensors over the same bytes can never
        disagree. A tensor spanning regions in different states reports ``cpu``
        while any part of it still needs flushing, since that is the answer that
        keeps a caller's reconcile from being skipped.
        """
        coherence = self._coherence_ref or self._coherence()
        uniform = coherence.uniform
        if uniform is not None:
            return uniform
        start, end = self._extent
        mixed = coherence.get(start, end)
        if mixed is not None:
            return mixed
        return "cpu" if coherence.any_host_written(start, end) else "npu"

    @device.setter
    def device(self, value):
        if value not in self.__class__.DEVICES:
            raise ValueError(f"Unsupported device: {value}")
        start, end = self._extent
        self._coherence().set(start, end, value)

    @property
    def base(self):
        """The buffer that owns this one's storage, or None if it owns it itself.

        Mirrors :attr:`numpy.ndarray.base`, including collapsing a chain of
        views: the base of a view of a view is the buffer that actually owns the
        storage, not the intermediate view. The intermediate is still referenced
        internally, so the whole chain stays alive for as long as any view of it
        does.
        """
        owner = self._parent
        while owner is not None:
            parent = owner._parent
            if parent is None:
                return owner
            owner = parent
        return None

    @property
    def is_view(self):
        """Whether this buffer shares another buffer's storage."""
        return self.base is not None

    @property
    def storage_offset(self):
        """Where this buffer starts within :attr:`base`'s storage, in bytes.

        Zero for a buffer that owns its storage. Accumulated through nesting, so
        it is always measured from the owner rather than from the view this one
        was carved out of. Compare :meth:`torch.Tensor.storage_offset`, which is
        in elements; this is in bytes because a view may reinterpret the dtype.
        """
        return self._offset_bytes

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
        self._initial_device = device
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
        self._reconcile_for_read()
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
        self._reconcile_for_read()
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
        self._reconcile_for_read()
        return self.data[index]

    def __setitem__(self, index, value):
        """
        Sets the value at a specific index in the tensor.

        Args:
            index (int): The index of the value to set.
            value: The new value to assign.

        Note: this reconciles before the write so the untouched elements keep
        their current contents, and records the write rather than flushing it,
        so a run of assignments costs one transfer at the next :meth:`to`.
        """
        with self.mutate() as array:
            array[index] = value

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

    def _reconcile_for_read(self):
        """Make the host's copy of this tensor's extent current.

        The pull half of ``to("cpu")`` without the claim that follows it: a read
        makes the host copy current, it does not make the host the author.
        Recording it as host-written would cost the next transfer a flush of
        bytes nobody wrote.

        Range-aware, because an extent covering regions in different states has
        no single answer: the regions the device holds are the ones to pull, and
        asking the tensor as a whole where it lives collapses them to one.
        """
        coherence = self._coherence_ref or self._coherence()
        if coherence.uniform == _CoherenceMap.HOST:
            return
        start, end = self._extent
        if coherence.get(start, end) == _CoherenceMap.HOST:
            return
        for lo, hi in coherence.ranges(start, end, _CoherenceMap.DEVICE):
            self.storage.sync_from_device(lo, hi - lo)

    def to(self, target_device: str):
        """
        Moves the tensor to a specified target device.

        Args:
            target_device (str): The target device.

        Returns:
           The tensor object on the target device.
        """
        coherence = self._coherence_ref or self._coherence()
        if coherence.uniform == target_device:
            return self
        start, end = self._extent
        if coherence.get(start, end) == target_device:
            # Already wholly where it is wanted: nothing to transfer and nothing
            # to record. A dispatch takes this path for every argument it does
            # not have to move, so it stays off the map entirely.
            return self
        if target_device == "npu":
            # Send the dirty ranges, not the whole extent. Ranges are coalesced,
            # so a buffer written end to end costs one transfer and a buffer
            # with a few dirty windows costs those windows.
            for lo, hi in coherence.ranges(start, end, _CoherenceMap.HOST):
                self.storage.sync_to_device(lo, hi - lo)
            coherence.set(start, end, "npu")
        elif target_device == "cpu":
            for lo, hi in coherence.ranges(start, end, _CoherenceMap.DEVICE):
                self.storage.sync_from_device(lo, hi - lo)
            coherence.set(start, end, "cpu")
        else:
            raise ValueError(f"Unknown device '{target_device}'")
        return self

    def mutate(self):
        """Borrow this buffer's bytes for a host write.

        Reconciles on entry so partial updates see current contents, records the
        write on exit, and retires the borrowed array so a reference kept past
        the block fails loudly instead of writing bytes nobody will flush.

        The flush itself is deferred: the region is marked host-written and the
        next transfer to the device sends every dirty range in one pass, so a
        caller filling many windows of one buffer pays for one flush, not one
        per window.

            with tensor.mutate() as buf:
                buf[:] = values
        """
        return _WriteBorrow(self, reconcile=True)

    def overwrite(self):
        """Borrow for a write that replaces every byte of this tensor.

        The same as :meth:`mutate` without the reconcile on entry, which nothing
        can observe if all of it is about to be replaced. Filling a buffer this
        way costs one transfer rather than two.

        The caller is promising to write the whole region. Bytes left unwritten
        keep whatever the host last had there and are sent to the device with
        the rest, so use :meth:`mutate` for a partial update. This is the one
        promise here that cannot be checked; it is still narrower than reaching
        for ``data``, which makes the same promise and does not record the write.
        """
        return _WriteBorrow(self, reconcile=False)

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

        Three ways this departs from numpy and torch, none of them accidental:

        Out-of-range and negative arguments are rejected, where numpy slicing
        clamps and wraps them. Clamping a region silently hands back a smaller
        one than asked for, and a DMA region that is quietly the wrong size is
        worse than an error. The method-with-arguments spelling is far enough
        from ``[start:stop]`` that it should not invite the slicing intuition.

        The alignment requirement has no analogue in either library, because
        neither reconciles anything between two agents.

        ``offset`` is in bytes, where numpy counts offsets in elements of the
        array's own dtype and torch reports ``storage_offset`` in elements.
        Bytes because that is the unit the thing being carved is measured in: a
        buffer has no dtype, the alignment rule below is in bytes, and
        :attr:`storage_offset` reports bytes, so the argument going in and the
        value reported back are the same number. ``shape`` stays in elements of
        the view's dtype, since it describes the tensor rather than the region.

        Args:
            offset: Start of the region, in bytes from the start of this
                tensor's own region.
            shape: Logical shape of the view.
            dtype (np.dtype, optional): dtype to interpret the region as.
                Defaults to this tensor's dtype.

        Returns:
            NpuTensor: A view sharing this tensor's storage.

        Raises:
            ValueError: If the region falls outside this tensor's buffer, or is
                not aligned to :data:`COHERENCE_GRANULE`.
        """
        if type(self)._subview is NpuTensor._subview:
            # Answer the capability question before complaining about a region
            # this backend could not have produced whatever its shape.
            raise NotImplementedError(
                f"{type(self).__name__} does not support subview()"
            )
        view_dtype = np.dtype(dtype) if dtype is not None else np.dtype(self.dtype)
        shape = _as_shape(shape)
        try:
            offset = operator.index(offset)
        except TypeError:
            raise TypeError(
                f"subview() offset must be an integer number of bytes, got "
                f"{offset!r}."
            ) from None
        offset_bytes = offset
        # math.prod over validated ints: np.prod would accumulate in int64 and
        # wrap silently, which defeats the bounds check below rather than
        # tripping it.
        nbytes = math.prod(shape) * view_dtype.itemsize
        if offset_bytes < 0 or offset_bytes + nbytes > self.nbytes:
            raise ValueError(
                f"subview(offset={offset}, shape={shape}, dtype={view_dtype}) "
                f"is out of bounds for a buffer of {self.nbytes} bytes"
            )
        granule = self._resolve_coherence_granule()
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

        The in-tree implementations build the view with ``__new__`` and set its
        fields directly, replaying only ``NpuTensor.__init__``, because running
        the backend's own ``__init__`` would allocate the buffer this view
        exists to avoid. The constraint that follows is on backend authors: a
        subclass that establishes state in its own ``__init__`` must set that
        state here too, or its views will be missing it. Nothing in tree does,
        which is why this is a note rather than a hook.

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
        self._reconcile_for_read()
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

        Note: this replaces every byte, so it skips the reconcile a partial
        write would need, and records the write rather than flushing it. The
        next transfer to the device sends it.
        """
        with self.overwrite() as array:
            array.fill(value)

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
        # Re-home the bytes in a buffer so views share one allocation and one
        # coherence map, then keep a typed view of the whole of it.
        source = self._data
        self._storage = Storage(
            HostOnlyTransport(source.nbytes),
            source.nbytes,
            self._initial_device,
            self._resolve_coherence_granule(),
        )
        self._offset_bytes = 0
        self._data = self._storage.host_bytes.view(source.dtype).reshape(self._shape)
        np.copyto(self._data, source)

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
        nbytes = math.prod(shape) * dtype.itemsize
        view = type(self).__new__(type(self))
        # Set the NpuTensor contract fields without allocating a new array.
        NpuTensor.__init__(view, shape, dtype=dtype, device=self.device)
        view._parent = self  # keep parent alive; shared storage
        view._shape = tuple(shape)
        view._storage = self._storage
        absolute = self.storage_offset + offset_bytes
        view._offset_bytes = absolute
        # A numpy view over the same bytes (zero-copy) so writes are shared.
        view._data = (
            self.storage.host_bytes[absolute : absolute + nbytes]
            .view(dtype)
            .reshape(view._shape)
        )
        return view


# The former name of NpuTensor, kept so existing callers and subclasses outside
# this repository keep working. New code should use NpuTensor.
Tensor = NpuTensor
