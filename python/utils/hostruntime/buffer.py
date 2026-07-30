# tensor.py -*- Python -*-
#
# Copyright (C) 2025-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Allocations, and how each backend reconciles one between host and device."""

from abc import ABC, abstractmethod

import numpy as np

from .coherence import _CoherenceMap


class NpuBuffer(ABC):
    """One allocation, and the coherence of the memory in it.

    A buffer owns bytes and knows how to reconcile them between host and device.
    It has no shape and no dtype: those are interpretations, and they belong to
    the :class:`NpuTensor` views layered over it. Many tensors may share one
    buffer, which is why the coherence state lives here -- it describes the
    memory, so storing it per tensor would let two names for the same bytes
    disagree.

    Compare :class:`torch.Storage`, which draws the same line for the same
    reason, and is where ``storage_offset`` comes from.
    """

    def __init__(self, nbytes, device, granule=None):
        self.nbytes = nbytes
        self.coherence = _CoherenceMap(nbytes, device, granule)

    @property
    @abstractmethod
    def host_bytes(self) -> np.ndarray:
        """The allocation as a flat ``uint8`` array the host can address.

        Every tensor over this buffer takes its data from here, so it is part of
        the contract rather than a convention each backend happens to follow.
        """

    @abstractmethod
    def sync_to_device(self, offset, nbytes):
        """Make the host's writes to ``[offset, offset+nbytes)`` visible to the device."""

    @abstractmethod
    def sync_from_device(self, offset, nbytes):
        """Make the device's writes to ``[offset, offset+nbytes)`` visible to the host."""

    def binding_handle(self, offset, nbytes):
        """A handle a runtime can bind for this region, if the backend has one.

        Returning None is a legitimate answer, not a stub: a design where the
        host writes at offsets into a whole allocation and the kernel addresses
        the layout itself never needs a per-region handle at all.
        """
        return None


class _TensorBackedBuffer(NpuBuffer):
    """Compatibility buffer for a backend that still owns its own storage.

    Delegates reconciliation to the tensor's transfer methods, whole-extent,
    which is what those backends did before buffers existed. It lets a backend
    adopt the split when it is ready rather than being restructured by this
    change, without opting out of per-region coherence.
    """

    def __init__(self, tensor):
        super().__init__(
            tensor.nbytes,
            tensor._initial_device,
            type(tensor)._resolve_coherence_granule(),
        )
        self._tensor = tensor

    def sync_to_device(self, offset, nbytes):
        self._tensor._sync_to_device()

    def sync_from_device(self, offset, nbytes):
        self._tensor._sync_from_device()

    def binding_handle(self, offset, nbytes):
        return getattr(self._tensor, "_bo", None)

    @property
    def host_bytes(self):
        return self._tensor._data.reshape(-1).view(np.uint8)


class CPUBuffer(NpuBuffer):
    """A host allocation. Reconciliation is a no-op: there is only one agent."""

    def __init__(self, nbytes, device, granule=None):
        super().__init__(nbytes, device, granule)
        self._host = np.zeros(nbytes, dtype=np.uint8)

    @property
    def host_bytes(self):
        return self._host

    def sync_to_device(self, offset, nbytes):
        pass

    def sync_from_device(self, offset, nbytes):
        pass
