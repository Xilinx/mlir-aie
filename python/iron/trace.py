# trace.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""What each core's body does to its objectFIFOs, recorded as a design is built.

Inside ``with record() as recording``, building a design notes, per
[`Worker`][aie.iron.Worker], every acquire and release its body makes on an
``ObjectFifo``, in program order, nested in the control flow it sits in
(``range_``, ``forever_``, ``if_``/``else_``), and every ``ObjectFifoLink``
the design draws, and every fill and drain its runtime sequence issues.
Each access and scope carries the user statement that
made it (a ``SourceSite``), the same one its emitted op is located at. A
Python ``for`` or ``if`` runs while the body is traced, so its accesses
appear unrolled. Nothing is recorded outside a ``record()`` block, and
recording changes nothing that is built.

The scopes are the frames the body is emitted in: every control-flow
construct a core body can use pushes one, and a new one must too, or what
it contains is credited to the frame around it.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Iterator, Literal

from ..helpers.sourceloc import SourceSite
from .runtime._context import active_recording_scope, frames

if TYPE_CHECKING:
    from ..dialects._aie_enum_gen import (  # pyright: ignore[reportMissingImports]
        ObjectFifoPort,
    )
    from ..helpers.taplib import TensorAccessPattern
    from .dataflow.objectfifo import ObjectFifo, ObjectFifoHandle, ObjectFifoLink


@dataclass(frozen=True)
class Access:
    """One acquire or release a core's body makes.

    Attributes:
        fifo: The ``ObjectFifo`` it names.
        port: The end it is made through.
        kind: Whether it acquires or releases.
        count: The objects it names.
        site: The user statement that made it.
    """

    fifo: ObjectFifo
    port: ObjectFifoPort
    kind: Literal["acquire", "release"]
    count: int
    site: SourceSite | None = field(default=None, compare=False)


@dataclass(frozen=True)
class Transfer:
    """One fill or drain the runtime sequence issues on an ``ObjectFifo``.

    Attributes:
        fifo: The ``ObjectFifo`` it moves objects into or out of.
        kind: Whether the host fills the fifo or drains it.
        elements: The elements it moves, or None where a per-call length or
            repeat, or a walk only the call knows, sizes it.
        site: The user statement that issued it.
    """

    fifo: ObjectFifo
    kind: Literal["fill", "drain"]
    elements: int | None
    site: SourceSite | None = field(default=None, compare=False)


@dataclass
class Scope:
    """A stretch of a core's body: the whole body, or one control-flow construct in it.

    Attributes:
        kind: ``"core"`` (a Worker's body), ``"for"`` (``range_``),
            ``"forever"``, ``"if"`` or ``"else"``.
        owner: The ``Worker`` for a core; for a ``for``, the normalized
            ``(start, stop, step)`` triple ``range_`` calls it with (defaults
            and keyword arguments applied, as ``_for`` itself would bind
            them), each entry an ``int`` or None where it is a value only the
            call knows; otherwise None. No MLIR object outlives the build.
        site: The user statement that opened it; for a core, the one that
            declared its Worker.
        body: What it does, in program order.
    """

    kind: Literal["core", "for", "forever", "if", "else"]
    owner: Any = None
    site: SourceSite | None = field(default=None, compare=False)
    body: list[Access | Scope] = field(default_factory=list)


class Recording:
    """What one ``record()`` block sees as designs are built.

    Attributes:
        cores: One ``"core"`` scope per Worker body built, in build order.
        links: Every ``ObjectFifoLink`` made, in order.
        transfers: Every fill and drain the runtime sequences issue, in order.
    """

    def __init__(self) -> None:
        self.cores: list[Scope] = []
        self.links: list[ObjectFifoLink] = []
        self.transfers: list[Transfer] = []

    def opened(
        self,
        kind: Literal["core", "for", "forever", "if", "else"],
        owner: Any,
        parent: Scope | None,
    ) -> Scope | None:
        """Return the scope a frame being pushed is noted as, None for control flow outside any core body."""
        if kind == "for":
            owner = tuple(a if isinstance(a, int) else None for a in owner)
        elif kind != "core":
            owner = None
        site = owner._site if kind == "core" else SourceSite.capture()
        opened = Scope(kind, owner, site)
        if kind == "core":
            self.cores.append(opened)
        elif parent is not None:
            parent.body.append(opened)
        else:
            return None
        return opened

    def acquired(self, fifo: ObjectFifo, port: ObjectFifoPort, count: int) -> None:
        """Note an acquire in the core body being emitted."""
        self._note(Access(fifo, port, "acquire", count, SourceSite.capture()))

    def released(self, fifo: ObjectFifo, port: ObjectFifoPort, count: int) -> None:
        """Note a release in the core body being emitted."""
        self._note(Access(fifo, port, "release", count, SourceSite.capture()))

    def transferred(
        self,
        handle: ObjectFifoHandle,
        tap: TensorAccessPattern | None,
        sized_per_call: bool,
    ) -> None:
        """Note a fill or drain of ``handle``'s fifo, ``tap`` the walk it moves.

        Args:
            handle: The end the transfer is issued on: a producer end is
                filled, a consumer end drained.
            tap: The walk the transfer moves, if one is given.
            sized_per_call: A per-call length or repeat sizes it.
        """
        elements = (
            tap.numel
            if tap is not None and isinstance(tap.numel, int) and not sized_per_call
            else None
        )
        kind = "fill" if handle.handle_type == "prod" else "drain"
        self.transfers.append(
            Transfer(handle.object_fifo, kind, elements, SourceSite.capture())
        )

    def linked(self, link: ObjectFifoLink) -> None:
        """Note an ``ObjectFifoLink`` the design draws."""
        self.links.append(link)

    def _note(self, access: Access) -> None:
        emitting = frames()
        if emitting and emitting[-1].scope is not None:
            emitting[-1].scope.body.append(access)


@contextmanager
def record() -> Iterator[Recording]:
    """Record what the designs built inside the block do to their objectFIFOs.

    Yields:
        The ``Recording`` that fills as they are built.
    """
    with active_recording_scope(Recording()) as recording:
        yield recording
