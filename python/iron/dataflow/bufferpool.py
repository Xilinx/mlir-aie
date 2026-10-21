# bufferpool.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""A pool of buffers and locks that a core reaches through acquire and release.

An [`ObjectFifo`][iron.ObjectFifo] states both ends of a transfer and generates
the buffers, the locks and the DMA programs at each end. A
[`BufferPool`][iron.BufferPool] states one end only. The other end belongs to a
DMA program the author writes, so a design keeps an ordered multi-BD
[`TileDma`][iron.TileDma] and still lets its core read like any other fifo
consumer.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ... import ir  # pyright: ignore[reportMissingImports]
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


def _named(thing, what: str) -> str:
    """Return ``thing``'s symbol name, which a pool refers to by name."""
    name = getattr(thing, "name", None)
    if not name:
        raise ValueError(
            f"BufferPool names its {what} in the IR, so give {thing!r} a name= "
            "when you construct it"
        )
    return name


class PoolSegment:
    """One slice of a [`BufferPool`][iron.BufferPool]'s object, and its locks.

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


class BufferPool(PerDeviceConfigurationResolvable):
    """A rotating set of buffers on one tile, and the locks that guard them.

    Example:
        ```python
        prod, cons = Lock(tile=t, init=1, name="x_prod"), Lock(tile=t, init=0, name="x_cons")
        pool = BufferPool(
            np.ndarray[(256,), np.dtype[np.int32]],
            tile=t,
            buffers=[Buffer(type=ty, name="x", tile=t)],
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
        """Construct a BufferPool.

        Args:
            obj_type (type[np.ndarray]): The type of one object.
            tile (Tile): The tile holding the buffers and the locks.
            depth (int, optional): Objects in the rotation. Defaults to 1.
            name (str | None, optional): The pool's symbol. Defaults to None.
            buffers (Sequence[Buffer] | None, optional): ``depth`` buffers to
                use, each named. The compiler makes them when this is None.
                Defaults to None.
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
            raise ValueError(f"BufferPool depth must be > 0, but got {depth}")
        if buffers is not None and len(buffers) != depth:
            raise ValueError(
                f"BufferPool of depth {depth} takes {depth} buffers, "
                f"got {len(buffers)}"
            )
        self._site = SourceSite.capture()
        self._obj_type = obj_type
        self._tile = tile
        self._depth = depth
        self.name = name
        self._buffers = list(buffers) if buffers is not None else None
        if segments is None:
            (elems,) = np_ndarray_type_get_shape(obj_type)
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
        self._handles: list[BufferPoolHandle] = []

    def _check_segments(self) -> None:
        """Check that the segments cover the object in increasing offset order."""
        (elems,) = np_ndarray_type_get_shape(self._obj_type)
        at = 0
        for segment in self._segments:
            if segment.offset != at:
                raise ValueError(
                    f"BufferPool segment at offset {segment.offset} leaves a "
                    f"gap or overlaps the one before it, which ends at {at}"
                )
            at += segment.size
        if at != elems:
            raise ValueError(
                f"BufferPool segments cover {at} of the object's {elems} elements"
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
    def op(self):
        if self._op is None:
            raise NotResolvedError()
        return self._op

    def tiles(self) -> list[Tile]:
        return [self._tile]

    def _segment_names(self, segments: Sequence[PoolSegment] | None) -> list[str]:
        """The symbols of ``segments``, defaulting to every one of the pool's."""
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
    ) -> "BufferPoolHandle":
        """Return a handle a Worker drains the pool through."""
        return self._handle(ObjectFifoRole.Drain, segments, name)

    def prod(
        self, segments: Sequence[PoolSegment] | None = None, name: str | None = None
    ) -> "BufferPoolHandle":
        """Return a handle a Worker fills the pool through."""
        return self._handle(ObjectFifoRole.Fill, segments, name)

    def _handle(self, role, segments, name) -> "BufferPoolHandle":
        handle = BufferPoolHandle(self, role, segments, name)
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
            raise ValueError("Cannot resolve BufferPool until it has been placed.")
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
            f"BufferPool({self._obj_type}, tile={self._tile}, "
            f"depth={self._depth}, name={self.name!r}, "
            f"segments={len(self._segments)})"
        )


class BufferPoolHandle(PerDeviceConfigurationResolvable):
    """What a Worker acquires and releases a [`BufferPool`][iron.BufferPool]'s objects through.

    Obtain one from [`BufferPool.cons()`][iron.dataflow.bufferpool.BufferPool.cons]
    or [`BufferPool.prod()`][iron.dataflow.bufferpool.BufferPool.prod] and pass
    it to a [`Worker`][iron.Worker] as a function argument, as for an
    [`ObjectFifoHandle`][iron.dataflow.objectfifo.ObjectFifoHandle].
    """

    def __init__(self, pool: BufferPool, role, segments, name: str | None):
        self._site = SourceSite.capture()
        self._pool = pool
        self._role = role
        self._segments = segments
        self._name = name
        self._op = None

    @property
    def name(self) -> str | None:
        """The endpoint's symbol."""
        return self._name

    @property
    def pool(self) -> BufferPool:
        """The pool this handle reaches."""
        return self._pool

    @property
    def op(self):
        if self._op is None:
            raise NotResolvedError()
        return self._op

    def tiles(self) -> list[Tile]:
        return self._pool.tiles()

    def _memref_type(self):
        """The type an acquire hands back: the run this handle's segments span."""
        chosen = self._pool.segments if self._segments is None else list(self._segments)
        elems = sum(segment.size for segment in chosen)
        dtype = np_ndarray_type_to_memref_type(self._pool.obj_type).element_type
        return ir.MemRefType.get([elems], dtype)

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
        verb = "cons" if self._role == ObjectFifoRole.Drain else "prod"
        self._name = self._name or f"{base}_{verb}"
        segments = self._pool._segment_names(self._segments)
        self._op = objectfifo_core_endpoint(
            self._name,
            self._pool.tile.op,
            self._role,
            base,
            segments=segments,
            loc=loc or self._site.location(self._name),
            ip=ip,
        )

    def __str__(self) -> str:
        return f"BufferPoolHandle({self._pool}, role={self._role}, name={self._name!r})"
