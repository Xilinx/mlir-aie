# tensor.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""HRX-backed Tensor: a device-visible, host-coherent buffer mapped once and
kept mapped (persistent), with explicit flush/invalidate for coherence.

The buffer is an HRX persistent mapping: the engine reads/writes the mapping
directly, so there is no host staging copy. Coherence around device work is
maintained with cheap cache ops:
  * ``_sync_to_device``  -> ``hrx_buffer_flush_range``     (host writes -> device)
  * ``_sync_from_device``-> ``hrx_buffer_invalidate_range``(device writes -> host)
"""

import ctypes
import numpy as np

from ..tensor_class import Tensor
from aie.helpers.util import np_ndarray_type_get_shape
from . import HRXContext


class HRXTensor(Tensor):
    """Tensor backed by an HRX persistent-mapped device buffer.

    Each tensor allocates its buffer through the process-wide
    :class:`~.context.HRXContext`. Buffers are therefore isolated per process:
    separate processes (including different users) never share buffer handles,
    and the amdxdna driver isolates each process's device memory. See
    :class:`~.context.HRXContext` for the full concurrency / multi-tenancy model
    (process isolation, the finite system-wide hardware-context pool, and the
    single-threaded-dispatch expectation within a process).
    """

    def __init__(self, shape_or_data, dtype=np.uint32, device="npu", **kwargs):
        """Allocate an HRX persistent-mapped buffer and wrap it as a tensor.

        Args:
            shape_or_data: Either a shape ``tuple`` to allocate a zero-filled
                buffer, or an array-like (anything with a ``shape``, or something
                ``numpy.asarray`` accepts) whose contents are copied in.
            dtype (numpy.dtype, optional): Element type used when ``shape_or_data``
                is a shape or a plain sequence. Defaults to ``numpy.uint32``.
            device (str, optional): Initial residency, ``"npu"`` or ``"cpu"``.
                ``"npu"`` flushes the initial host contents to the device after
                allocation. Defaults to ``"npu"``.
            **kwargs: Accepted for API compatibility with other tensor backends;
                ignored by HRX.
        """
        super().__init__(shape_or_data, dtype=dtype, device=device)
        self._ctx = HRXContext.get()

        np_data = None
        if isinstance(shape_or_data, tuple):
            np_type = np.ndarray[shape_or_data, np.dtype[dtype]]
            self._shape = np_ndarray_type_get_shape(np_type)
        elif hasattr(shape_or_data, "shape"):
            self._shape = shape_or_data.shape
            np_data = shape_or_data
        else:
            np_data = np.asarray(shape_or_data, dtype=dtype)
            self._shape = np_data.shape

        nbytes = int(np.prod(self._shape) * np.dtype(self.dtype).itemsize)
        # HRX rejects zero-size allocations; keep a 1-element floor like XRT does
        # implicitly via group_id buffers (designs never use 0-size IO here).
        self._alloc_size = max(nbytes, 1)
        self._buf, host_ptr = self._ctx.allocate_persistent(self._alloc_size)
        assert host_ptr is not None

        # Wrap the persistent host pointer as a numpy array (zero-copy view).
        buf_type = ctypes.c_char * self._alloc_size
        self._cbuf = buf_type.from_address(host_ptr)
        self._data = np.frombuffer(
            self._cbuf, dtype=self.dtype, count=int(np.prod(self._shape))
        ).reshape(self._shape)

        if np_data is not None:
            np.copyto(self._data, np_data)
        else:
            self._data.fill(0)

        if self.device == "npu":
            self._sync_to_device()

    @property
    def data(self):
        assert self._data is not None
        return self._data

    @property
    def shape(self):
        return self._shape

    def _sync_to_device(self):
        """Flush host writes out to the device (cheap clflush, no copy)."""
        if self._buf:
            self._ctx.flush_range(self._buf, 0, self._alloc_size)

    def _sync_from_device(self):
        """Invalidate the host cache so reads observe device writes."""
        if self._buf:
            self._ctx.invalidate_range(self._buf, 0, self._alloc_size)

    def __del__(self):
        # Drop the numpy view before releasing the underlying mapping.
        try:
            self._data = None
            self._cbuf = None
        except Exception:
            pass
        buf = getattr(self, "_buf", None)
        if buf:
            try:
                self._ctx.release_buffer(buf)
            except Exception:
                pass
            self._buf = None

    def buffer_object(self):
        """Return the underlying HRX buffer handle.

        Returns:
            The opaque ``hrx_buffer_t`` handle backing this tensor (used as the
            ``buffer`` field of a dispatch binding).
        """
        return self._buf

    def nbytes_alloc(self) -> int:
        """Return the allocated buffer size in bytes.

        Returns:
            int: The number of bytes allocated on the device (the element count
            times the item size, with a 1-byte floor since HRX rejects zero-size
            allocations).
        """
        return self._alloc_size
