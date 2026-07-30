# buffer.py -*- Python -*-
#
# Copyright (C) 2025-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""An allocation, and how a backend reaches and reconciles the bytes in it.

The allocation is one class. What differs between backends is not the allocation
but the two things layered under it: where the host's bytes come from, and what
reconciling a range actually does. Those are the :class:`Transport`.
"""

from abc import ABC, abstractmethod

import numpy as np

from .coherence import _CoherenceMap


class Transport(ABC):
    """How one allocation's bytes are reached, and reconciled between agents.

    Everything that varies between backends lives here, and it is only ever
    these three things: where the host's bytes come from, what moving a range in
    either direction does, and whether a region has a handle a runtime can bind.
    A backend is a transport, not a kind of allocation.

    Ranges are part of the contract rather than a hint. A transport that cannot
    honour them says so in its name (see :class:`WholeExtentTransport`) instead
    of taking the arguments and ignoring them, which is a promise the caller has
    no way to check, and is the reason this is a strategy rather than a subclass
    of the allocation.
    """

    @property
    @abstractmethod
    def host_bytes(self) -> np.ndarray:
        """The allocation as a flat ``uint8`` array the host can address."""

    @abstractmethod
    def to_device(self, offset, nbytes):
        """Make the host's writes to ``[offset, offset+nbytes)`` visible to the device."""

    @abstractmethod
    def from_device(self, offset, nbytes):
        """Make the device's writes to ``[offset, offset+nbytes)`` visible to the host."""

    def handle(self, offset, nbytes):
        """A handle a runtime can bind for this region, if the backend has one.

        Returning None is a legitimate answer, not a stub: a design where the
        host writes at offsets into a whole allocation and the kernel addresses
        the layout itself never needs a per-region handle at all.
        """
        return None


class HostOnlyTransport(Transport):
    """Bytes the host allocates and no other agent touches.

    Reconciliation is a no-op because there is only one agent, which is the
    degenerate case of this contract rather than a different one.
    """

    def __init__(self, nbytes):
        self._host = np.zeros(nbytes, dtype=np.uint8)

    @property
    def host_bytes(self):
        return self._host

    def to_device(self, offset, nbytes):
        pass

    def from_device(self, offset, nbytes):
        pass


class WholeExtentTransport(Transport):
    """For a backend whose transfer methods do not take a range.

    Reconciles the whole allocation whatever range it is handed, which is what
    those backends did before storage was split out of the tensor. The name is
    the point: over-reconciling is safe, so the cost is wider cache maintenance
    than was asked for, and per-region coherence is still tracked above it, but
    a reader can see which it is without opening the method.

    A backend that can reconcile by range wants its own transport instead. HRX
    is the standing example, since ``hrx_buffer_flush_range`` already takes an
    offset and a size and it is the tensor layer above that discards them.
    """

    def __init__(self, tensor):
        self._tensor = tensor

    @property
    def host_bytes(self):
        return self._tensor._data.reshape(-1).view(np.uint8)

    def to_device(self, offset, nbytes):
        self._tensor._sync_to_device()

    def from_device(self, offset, nbytes):
        self._tensor._sync_from_device()

    def handle(self, offset, nbytes):
        return getattr(self._tensor, "_bo", None)


class Storage:
    """One allocation, and the coherence of the memory in it.

    Storage owns bytes and the record of which agent holds each range of them.
    It has no shape and no dtype: those are interpretations, and interpretations
    are what :class:`NpuTensor` is for. Many tensors may name one storage, which
    is why the coherence state lives here. Kept per tensor, two names for the
    same bytes could disagree and nothing would reconcile them.

    This is the split torch draws between ``UntypedStorage`` and ``Tensor``, and
    is where ``storage_offset`` comes from.

    Not subclassed. A backend supplies a :class:`Transport`, so the reconcile
    mechanism is data this class holds rather than an identity a subclass
    carries. That is the same argument made one level up about coherence
    belonging to the memory rather than to whichever tensor names it, and it is
    what stops a backend quietly redefining what a range means.
    """

    def __init__(self, transport: Transport, nbytes, device, granule=None):
        self.nbytes = nbytes
        self.coherence = _CoherenceMap(nbytes, device, granule)
        self._transport = transport

    @property
    def host_bytes(self) -> np.ndarray:
        """The allocation as a flat ``uint8`` array the host can address."""
        return self._transport.host_bytes

    def sync_to_device(self, offset, nbytes):
        """Make the host's writes to ``[offset, offset+nbytes)`` visible to the device."""
        self._transport.to_device(offset, nbytes)

    def sync_from_device(self, offset, nbytes):
        """Make the device's writes to ``[offset, offset+nbytes)`` visible to the host."""
        self._transport.from_device(offset, nbytes)

    def binding_handle(self, offset, nbytes):
        """A handle a runtime can bind for this region, or None."""
        return self._transport.handle(offset, nbytes)
