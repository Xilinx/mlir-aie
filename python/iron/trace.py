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

Every control-flow construct a core body can use opens a
``Recorder.scope``; a new one must too, or what it contains is credited to
the scope around it.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, Iterator, Literal

if TYPE_CHECKING:
    from ..dialects._aie_enum_gen import ObjectFifoPort
    from .dataflow.objectfifo import ObjectFifo, ObjectFifoLink


@dataclass(frozen=True)
class Access:
    """One acquire or release a core's body makes.

    Attributes:
        fifo: The ``ObjectFifo`` it names.
        port: The end it is made through.
        kind: Whether it acquires or releases.
        count: The objects it names.
    """

    fifo: ObjectFifo
    port: ObjectFifoPort
    kind: Literal["acquire", "release"]
    count: int


@dataclass
class Scope:
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
    body: list[Access | Scope] = field(default_factory=list)


class Recorder:
    """What one ``record()`` block sees as designs are built.

    Attributes:
        cores: One ``"core"`` scope per Worker body built, in build order.
        links: Every ``ObjectFifoLink`` made, in order.
    """

    _active: ClassVar[ContextVar[Recorder | None]] = ContextVar(
        "iron_trace_recorder", default=None
    )

    def __init__(self) -> None:
        self.cores: list[Scope] = []
        self.links: list[ObjectFifoLink] = []
        self._open: list[Scope] = []

    @classmethod
    def active(cls) -> Recorder | None:
        """Return the recorder a ``record()`` block opened, or None outside one."""
        return cls._active.get()

    @classmethod
    @contextmanager
    def scope(
        cls,
        kind: Literal["core", "for", "forever", "if", "else"],
        owner: Any = None,
    ) -> Iterator[None]:
        """Note that what runs inside is a core body or one control-flow construct in one.

        Records nothing outside a ``record()`` block, nor for control flow
        outside any core body (a runtime sequence's loop, a Worker's own
        loop around its body).
        """
        recorder = cls.active()
        if recorder is None or (kind != "core" and not recorder._open):
            yield
            return
        opened = Scope(kind, owner)
        if kind == "core":
            recorder.cores.append(opened)
        else:
            recorder._open[-1].body.append(opened)
        recorder._open.append(opened)
        try:
            yield
        finally:
            recorder._open.pop()

    def acquired(self, fifo: ObjectFifo, port: ObjectFifoPort, count: int) -> None:
        """Note an acquire in the core body being built."""
        if self._open:
            self._open[-1].body.append(Access(fifo, port, "acquire", count))

    def released(self, fifo: ObjectFifo, port: ObjectFifoPort, count: int) -> None:
        """Note a release in the core body being built."""
        if self._open:
            self._open[-1].body.append(Access(fifo, port, "release", count))

    def linked(self, link: ObjectFifoLink) -> None:
        """Note an ``ObjectFifoLink`` the design draws."""
        self.links.append(link)


@contextmanager
def record() -> Iterator[Recorder]:
    """Record what the designs built inside the block do to their objectFIFOs.

    Yields:
        The ``Recorder`` that fills as they are built.
    """
    recorder = Recorder()
    token = Recorder._active.set(recorder)
    try:
        yield recorder
    finally:
        Recorder._active.reset(token)
