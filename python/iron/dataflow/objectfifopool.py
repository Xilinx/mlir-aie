# objectfifopool.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""A pool of buffers and locks that a core reaches through acquire and release.

An [`ObjectFifo`][aie.iron.ObjectFifo] states both ends of a transfer and generates
the buffers, the locks and the DMA programs at each end. A
[`ObjectFifoPool`][aie.iron.ObjectFifoPool] states one end only. The other end belongs to a
DMA program the author writes, so a design keeps an ordered multi-BD
[`TileDma`][aie.iron.TileDma] and still lets its core read like any other fifo
consumer.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from ... import ir  # pyright: ignore[reportMissingImports, reportAttributeAccessIssue]
from ...dialects._aie_enum_gen import (  # pyright: ignore[reportMissingImports]
    ObjectFifoRole,
)
from ...dialects._aie_ops_gen import (  # pyright: ignore[reportMissingImports]
    objectfifo_acquire,
    objectfifo_core_endpoint,
    objectfifo_pool,
    objectfifo_release,
    objectfifo_segment,
)
from ...helpers.npdtypes import np_ndarray_type_get_shape
from ...helpers.sourceloc import SourceSite
from ...helpers.util import np_ndarray_type_to_memref_type
from ..device import Tile
from ..resolvable import NotResolvedError, PerDeviceConfigurationResolvable
from .endpoint import ObjectFifoEndpoint


def _named(thing, what: str) -> str:
    """Return ``thing``'s symbol name, which a pool refers to by name."""
    name = getattr(thing, "name", None)
    if not name:
        raise ValueError(
            f"ObjectFifoPool names its {what} in the IR, so {thing!r} needs a "
            "name. A DeviceConfiguration names one it reaches through a Worker; "
            "otherwise pass name= when you construct it"
        )
    return name


class PoolSegment:
    """One slice of an [`ObjectFifoPool`][aie.iron.ObjectFifoPool]'s object, and its locks.

    Segments partition the object in increasing offset order and together
    cover it exactly. Each carries a produce and a consume lock, so several
    actors can hold disjoint slices at once.

    ``manual_fill`` and ``manual_drain`` hand one end to a DMA program the
    author writes. That end names no handle, and the author owes it the lock
    protocol the other end expects: a filler acquires ``produce_lock`` and
    releases ``consume_lock`` once per object, a drainer the reverse.
    """

    def __init__(
        self,
        size: int,
        *,
        offset: int = 0,
        name: str | None = None,
        produce_lock=None,
        consume_lock=None,
        manual_fill: bool = False,
        manual_drain: bool = False,
    ):
        """Construct a PoolSegment.

        Args:
            size (int): Elements of the object this segment spans.
            offset (int, optional): Its first element. Defaults to 0.
            name (str | None, optional): Its symbol. The pool names it when
                this is None. Defaults to None.
            produce_lock (Lock | None, optional): The lock a filler takes. The
                compiler makes one when this is None, which a hand-written DMA
                program cannot then name. Defaults to None.
            consume_lock (Lock | None, optional): The lock a drainer takes,
                under the same rule. Defaults to None.
            manual_fill (bool, optional): A DMA program the author writes fills
                this segment. Defaults to False.
            manual_drain (bool, optional): The same at the draining end.
                Defaults to False.
        """
        self.size = size
        self.offset = offset
        self.name = name
        self.produce_lock = produce_lock
        self.consume_lock = consume_lock
        self.manual_fill = manual_fill
        self.manual_drain = manual_drain

    def _resolve(self, index: int, base: str) -> None:
        """Emit this segment, inside its pool's body."""
        locks = {}
        if self.produce_lock is not None:
            locks["produce_lock"] = _named(self.produce_lock, "produce lock")
        if self.consume_lock is not None:
            locks["consume_lock"] = _named(self.consume_lock, "consume lock")
        objectfifo_segment(
            self.name or f"{base}_s{index}",
            self.offset,
            self.size,
            manual_fill=self.manual_fill or None,
            manual_drain=self.manual_drain or None,
            **locks,
        )

    def __repr__(self) -> str:
        return (
            f"PoolSegment(size={self.size}, offset={self.offset}, "
            f"name={self.name!r})"
        )


class ObjectFifoPool(PerDeviceConfigurationResolvable):
    """A rotating set of buffers on one tile, and the locks that guard them.

    Example:
        ```python
        prod, cons = Lock(tile=t, init=1), Lock(tile=t, init=0)
        pool = ObjectFifoPool(
            np.ndarray[(256,), np.dtype[np.int32]],
            tile=t,
            buffers=[Buffer(type=ty, tile=t)],
            segments=[PoolSegment(256, produce_lock=prod, consume_lock=cons, manual_fill=True)],
        )
        # The Worker takes pool.cons() and calls acquire(1) / release(1) on it.
        # A hand-written Bd acquires `prod` and releases `cons`.
        ```
    """

    def __init__(
        self,
        obj_type: type[np.ndarray],
        *,
        tile: Tile,
        depth: int = 1,
        name: str | None = None,
        buffers: Sequence | None = None,
        segments: Sequence[PoolSegment] | None = None,
        produce_lock=None,
        consume_lock=None,
        manual_fill: bool = False,
        manual_drain: bool = False,
    ):
        """Construct an ObjectFifoPool.

        Args:
            obj_type (type[np.ndarray]): The type of one object.
            tile (Tile): The tile holding the buffers and the locks.
            depth (int, optional): Objects in the rotation. Defaults to 1.
            name (str | None, optional): The pool's symbol. Defaults to None.
            buffers (Sequence[Buffer] | None, optional): ``depth`` buffers to
                use. The compiler makes them when this is None. Defaults to
                None.
            segments (Sequence[PoolSegment] | None, optional): How the object
                divides. Defaults to one segment spanning it.
            produce_lock (Lock | None, optional): The lone segment's produce
                lock. Ignored when ``segments`` is given. Defaults to None.
            consume_lock (Lock | None, optional): Its consume lock, under the
                same rule. Defaults to None.
            manual_fill (bool, optional): Set ``manual_fill`` on that lone
                segment. Ignored when ``segments`` is given. Defaults to False.
            manual_drain (bool, optional): The same at the draining end.
                Defaults to False.

        Raises:
            ValueError: Arguments are validated.
        """
        if depth < 1:
            raise ValueError(f"ObjectFifoPool depth must be > 0, but got {depth}")
        if buffers is not None and len(buffers) != depth:
            raise ValueError(
                f"ObjectFifoPool of depth {depth} takes {depth} buffers, "
                f"got {len(buffers)}"
            )
        self._site = SourceSite.capture()
        self._obj_type = obj_type
        self._tile = tile
        self._depth = depth
        self.name = name
        self._buffers = list(buffers) if buffers is not None else None
        if segments is None:
            elems = math.prod(np_ndarray_type_get_shape(obj_type))
            segments = [
                PoolSegment(
                    elems,
                    produce_lock=produce_lock,
                    consume_lock=consume_lock,
                    manual_fill=manual_fill,
                    manual_drain=manual_drain,
                )
            ]
        self._segments = list(segments)
        self._check_segments()
        self._op = None
        self._handles: list[ObjectFifoPoolHandle] = []

    def _check_segments(self) -> None:
        """Check that the segments cover the object in increasing offset order."""
        elems = math.prod(np_ndarray_type_get_shape(self._obj_type))
        at = 0
        for segment in self._segments:
            if segment.offset != at:
                raise ValueError(
                    f"ObjectFifoPool segment at offset {segment.offset} leaves a "
                    f"gap or overlaps the one before it, which ends at {at}"
                )
            at += segment.size
        if at != elems:
            raise ValueError(
                f"ObjectFifoPool segments cover {at} of the object's {elems} elements"
            )

    @property
    def obj_type(self) -> type[np.ndarray]:
        """The type of one object."""
        return self._obj_type

    @property
    def tile(self) -> Tile:
        """The tile holding the buffers and the locks."""
        return self._tile

    @property
    def buffers(self) -> list | None:
        """The buffers given at construction, which a hand-written BD names."""
        return list(self._buffers) if self._buffers is not None else None

    @property
    def segments(self) -> list[PoolSegment]:
        """The segments, in increasing offset order."""
        return list(self._segments)

    @property
    def handles(self) -> list["ObjectFifoPoolHandle"]:
        """The handles taken on this pool."""
        return list(self._handles)

    def resources(self) -> list:
        """Return the buffers and locks this pool refers to by symbol."""
        locks = [
            lock
            for segment in self._segments
            for lock in (segment.produce_lock, segment.consume_lock)
            if lock is not None
        ]
        return [*(self._buffers or []), *locks]

    def name_symbols(self, claim) -> None:
        """Name every symbol this pool refers to that the design left unnamed.

        Args:
            claim: Takes a preferred name and returns one free of collisions.
        """
        for index, buffer in enumerate(self._buffers or []):
            if buffer.name is None:
                buffer._name = claim(f"{self.name}__buf{index}")
        for index, segment in enumerate(self._segments):
            for lock, role in (
                (segment.produce_lock, "prod"),
                (segment.consume_lock, "cons"),
            ):
                if lock is not None and lock.name is None:
                    lock._name = claim(f"{self.name}__{role}lock{index}")
        for handle in self._handles:
            if handle.name is None:
                peers = [h for h in self._handles if h.verb == handle.verb]
                index = peers.index(handle)
                suffix = "" if index == 0 else str(index)
                handle._name = claim(f"{self.name}__{handle.verb}{suffix}")

    @property
    def op(self):
        if self._op is None:
            raise NotResolvedError()
        return self._op

    def tiles(self) -> list[Tile]:
        return [self._tile]

    def _segment_names(self, segments: Sequence[PoolSegment] | None) -> list[str]:
        """Return the symbols of ``segments``, or every one of the pool's."""
        chosen = self._segments if segments is None else list(segments)
        for segment in chosen:
            if segment not in self._segments:
                raise ValueError(f"{segment!r} is not a segment of this pool")
        base = self.name or "pool"
        return [
            segment.name or f"{base}_s{self._segments.index(segment)}"
            for segment in chosen
        ]

    def cons(
        self, segments: Sequence[PoolSegment] | None = None, name: str | None = None
    ) -> "ObjectFifoPoolHandle":
        """Return a handle a Worker drains the pool through."""
        return self._handle(ObjectFifoRole.Drain, segments, name)

    def prod(
        self, segments: Sequence[PoolSegment] | None = None, name: str | None = None
    ) -> "ObjectFifoPoolHandle":
        """Return a handle a Worker fills the pool through."""
        return self._handle(ObjectFifoRole.Fill, segments, name)

    def _handle(self, role, segments, name) -> "ObjectFifoPoolHandle":
        handle = ObjectFifoPoolHandle(self, role, segments, name)
        self._handles.append(handle)
        return handle

    def resolve(
        self,
        loc: ir.Location | None = None,
        ip: ir.InsertionPoint | None = None,
    ) -> None:
        if self._op is not None:
            return
        if self._tile is None:
            raise ValueError("Cannot resolve ObjectFifoPool until it has been placed.")
        base = self.name or "pool"
        buffers = (
            [_named(buffer, "buffer") for buffer in self._buffers]
            if self._buffers is not None
            else None
        )
        self._op = objectfifo_pool(
            base,
            self._tile.op,
            self._depth,
            ir.TypeAttr.get(np_ndarray_type_to_memref_type(self._obj_type)),
            buffers=buffers,
            loc=loc or self._site.location(base),
            ip=ip,
        )
        block = self._op.regions[0].blocks.append()
        with ir.InsertionPoint(block):
            for index, segment in enumerate(self._segments):
                segment._resolve(index, base)

    def __str__(self) -> str:
        return (
            f"ObjectFifoPool({self._obj_type}, tile={self._tile}, "
            f"depth={self._depth}, name={self.name!r}, "
            f"segments={len(self._segments)})"
        )


class ObjectFifoPoolHandle(PerDeviceConfigurationResolvable):
    """What a Worker acquires and releases an [`ObjectFifoPool`][aie.iron.ObjectFifoPool]'s objects through.

    Obtain one from
    [`ObjectFifoPool.cons()`][aie.iron.dataflow.objectfifopool.ObjectFifoPool.cons]
    or
    [`ObjectFifoPool.prod()`][aie.iron.dataflow.objectfifopool.ObjectFifoPool.prod]
    and pass it to a [`Worker`][aie.iron.Worker] as a function argument, as for an
    [`ObjectFifoHandle`][aie.iron.dataflow.objectfifo.ObjectFifoHandle]. A core
    body takes either and cannot tell them apart.
    """

    def __init__(self, pool: ObjectFifoPool, role, segments, name: str | None):
        self._site = SourceSite.capture()
        self._pool = pool
        self._role = role
        self._segments = segments
        self._name = name
        self._op = None
        self._endpoint = None

    @property
    def name(self) -> str | None:
        """The endpoint's symbol."""
        return self._name

    @property
    def verb(self) -> str:
        """What this handle does to the pool, as its symbol spells it."""
        return "cons" if self._role == ObjectFifoRole.Drain else "prod"

    @property
    def pool(self) -> ObjectFifoPool:
        """The pool this handle reaches."""
        return self._pool

    @property
    def endpoint(self) -> ObjectFifoEndpoint | None:
        """The endpoint of this ObjectFifoPoolHandle."""
        return self._endpoint

    @endpoint.setter
    def endpoint(self, endpoint: ObjectFifoEndpoint) -> None:
        if self._endpoint and self._endpoint != endpoint:
            raise ValueError(
                f"Endpoint already set for ObjectFifoPoolHandle {self._name}: "
                f"Set to {self._endpoint}, trying to set to {endpoint}"
            )
        self._endpoint = endpoint

    @property
    def op(self):
        if self._op is None:
            raise NotResolvedError()
        return self._op

    def tiles(self) -> list[Tile]:
        return self._pool.tiles()

    def _memref_type(self):
        """Return the type an acquire hands back.

        Mirrors ``ObjectFifoCoreEndpointOp::getAccessType()``: the object's own
        type for the whole object, and the rank-reduced subview of the run this
        handle's segments span for anything less. The offset has to survive,
        since the core reads the objects in place.
        """
        elem = np_ndarray_type_to_memref_type(self._pool.obj_type)
        chosen = self._pool.segments if self._segments is None else list(self._segments)
        offset = chosen[0].offset
        size = chosen[-1].offset + chosen[-1].size - offset
        if offset == 0 and size == math.prod(elem.shape):
            return elem
        return ir.MemRefType.get(
            [size], elem.element_type, ir.StridedLayoutAttr.get(offset, [1])
        )

    def acquire(self, num_elem: int = 1):
        """Acquire ``num_elem`` objects, as an ObjectFifo handle does.

        Returns the object itself for one, and the list for more. The buffers
        a hand-written DMA program fills reach the core this way.
        """
        if num_elem < 1:
            raise ValueError("Must acquire at least one object")
        types = [self._memref_type()] * num_elem
        result = objectfifo_acquire(types, self.op.sym_name.value)
        if num_elem == 1:
            return result
        return list(result)

    def release(self, num_elem: int = 1) -> None:
        """Release ``num_elem`` objects back to the other end."""
        if num_elem < 1:
            raise ValueError("Must release at least one object")
        objectfifo_release(self.op.sym_name.value, num_elem)

    def resolve(
        self,
        loc: ir.Location | None = None,
        ip: ir.InsertionPoint | None = None,
    ) -> None:
        if self._op is not None:
            return
        self._pool.resolve(loc=loc, ip=ip)
        base = self._pool.name or "pool"
        if self._name is None:
            # Several handles may take the same role on one pool, so the symbol
            # carries the handle's index among them. A DeviceConfiguration names
            # them before this, and only a pool resolved outside one lands here.
            peers = [h for h in self._pool.handles if h.verb == self.verb]
            index = peers.index(self)
            suffix = "" if index == 0 else str(index)
            self._name = f"{base}__{self.verb}{suffix}"
        segments = self._pool._segment_names(self._segments)
        # The core that took this handle is the one that reaches the objects,
        # so the endpoint sits on its tile; the verifier rejects an endpoint
        # reached from anywhere else. An unplaced endpoint falls back to the
        # pool's own tile.
        tile = self._endpoint.tile if self._endpoint is not None else None
        tile = tile or self._pool.tile
        self._op = objectfifo_core_endpoint(
            self._name,
            tile.op,
            self._role,
            base,
            segments=segments,
            loc=loc or self._site.location(self._name),
            ip=ip,
        )

    def __str__(self) -> str:
        return (
            f"ObjectFifoPoolHandle({self._pool}, role={self._role}, "
            f"name={self._name!r})"
        )
