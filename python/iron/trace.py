# trace.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""What each core's body does to its objectFIFOs, recorded as a design is built.

Inside ``with record() as recorder``, building a design notes, per
[`Worker`][aie.iron.Worker], every acquire and release its body makes on an
``ObjectFifo``, in program order, nested in the control flow it sits in
(``range_``, ``forever_``, ``if_``/``else_``), and every ``ObjectFifoLink``
the design draws. A Python ``for`` or ``if`` runs while the body is traced,
so its accesses appear unrolled. Nothing is recorded outside a ``record()``
block, and recording changes nothing that is built.

Every control-flow construct a core body can use records its own block; a
new one must too, or what it contains is credited to the block around it.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Iterator, Literal

if TYPE_CHECKING:
    from .dataflow.objectfifo import ObjectFifo


@dataclass(frozen=True)
class Access:
    """One acquire or release a core's body makes.

    Attributes:
        fifo: The ``ObjectFifo`` it names.
        producer: It is made through the fifo's producer end.
        acquire: It acquires (False: it releases).
        count: The objects it names.
    """

    fifo: ObjectFifo
    producer: bool
    acquire: bool
    count: int


@dataclass(frozen=True)
class Link:
    """An ``ObjectFifoLink``: objects moved through a memtile from some fifos into others.

    Attributes:
        srcs: The fifos it reads.
        dsts: The fifos it fills.
        src_offsets: Where each source lands in a destination object, joining.
        dst_offsets: Where each destination's part starts in a source object, splitting.
        mode: ``"space"`` splits or joins every object at the offsets;
            ``"time"`` has the many side take turns with whole objects.
    """

    srcs: tuple[ObjectFifo, ...]
    dsts: tuple[ObjectFifo, ...]
    src_offsets: tuple[int, ...]
    dst_offsets: tuple[int, ...]
    mode: Literal["space", "time"]


@dataclass
class Block:
    """A stretch of a core's body: the whole body, or one control-flow construct in it.

    Attributes:
        kind: ``"core"`` (a Worker's body), ``"for"`` (``range_``),
            ``"forever"``, ``"if"`` or ``"else"``.
        owner: The ``Worker`` for a core, ``range_``'s arguments for a
            ``for``, the condition for an ``if``; otherwise None.
        body: What it does, in program order.
    """

    kind: Literal["core", "for", "forever", "if", "else"]
    owner: Any = None
    body: list[Access | Block] = field(default_factory=list)


class Recorder:
    """What one ``record()`` block saw.

    Attributes:
        cores: One ``"core"`` block per Worker body built, in build order.
        links: Every ``ObjectFifoLink`` made, in order.
    """

    def __init__(self) -> None:
        self.cores: list[Block] = []
        self.links: list[Link] = []
        self.open: list[Block] = []


_recorder: ContextVar[Recorder | None] = ContextVar("iron_trace_recorder", default=None)


@contextmanager
def record() -> Iterator[Recorder]:
    """Record what the designs built inside the block do to their objectFIFOs.

    Yields:
        The ``Recorder`` that fills as they are built.
    """
    recorder = Recorder()
    token = _recorder.set(recorder)
    try:
        yield recorder
    finally:
        _recorder.reset(token)


@contextmanager
def block(
    kind: Literal["core", "for", "forever", "if", "else"], owner: Any = None
) -> Iterator[None]:
    """Note that what runs inside is a core body or one control-flow construct in one.

    A control-flow block outside any core body (a runtime sequence's loop, a
    Worker's own loop around its body) is not recorded.
    """
    recorder = _recorder.get()
    if recorder is None or (kind != "core" and not recorder.open):
        yield
        return
    noted = Block(kind, owner)
    if kind == "core":
        recorder.cores.append(noted)
    else:
        recorder.open[-1].body.append(noted)
    recorder.open.append(noted)
    try:
        yield
    finally:
        recorder.open.pop()


def access(fifo: ObjectFifo, producer: bool, acquire: bool, count: int) -> None:
    """Note an acquire or a release in the core body being built, if one is recorded."""
    recorder = _recorder.get()
    if recorder is not None and recorder.open:
        recorder.open[-1].body.append(Access(fifo, producer, acquire, count))


def link(made: Link) -> None:
    """Note an ``ObjectFifoLink`` the design draws, if one is recorded."""
    recorder = _recorder.get()
    if recorder is not None:
        recorder.links.append(made)
