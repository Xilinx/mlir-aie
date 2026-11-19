# _context.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Active-sequence, configuration and recording context, and the frames of the body being emitted.

The runtime sequence body runs eagerly inside ``@runtime_sequence`` at resolve
time (mirroring how ``Worker.core_fn`` runs inside ``@core``). The body's
data-movement verbs live on ObjectFifo handles (``inA.prod().fill(...)``), so
they need a way to reach the in-flight sequence without an ``rt`` reference being
threaded through the body signature. This module holds that link as a ContextVar,
so it is well-defined even across nested calls and is always cleared on exit.

It also holds the frames of the body being emitted, one per core body and
control-flow construct (``range_``, ``forever_``, ``if_``, ``else_``),
innermost last, and the recording (``aie.iron.trace.record``) a tool opened,
which notes each frame as a scope while it is active.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterator, Literal

if TYPE_CHECKING:
    from ..configuration import DeviceConfiguration
    from ..trace import Recording, Scope
    from .runtime import ActiveSequence

_active_sequence: ContextVar["ActiveSequence | None"] = ContextVar(
    "iron_active_sequence", default=None
)
_active_configuration: ContextVar["DeviceConfiguration | None"] = ContextVar(
    "iron_active_configuration", default=None
)


def active_sequence() -> "ActiveSequence":
    """Return the sequence currently being emitted.

    Raises:
        RuntimeError: If called outside a runtime sequence body (e.g. a
            ``fill``/``drain`` verb invoked outside the function passed to
            ``Runtime(seq_fn, fn_args)``).
    """
    seq = _active_sequence.get()
    if seq is None:
        raise RuntimeError(
            "No active runtime sequence: fill()/drain(), DmaEndpoint.task() and "
            "TaskGroup() must be called from within the function passed to "
            "Runtime(seq_fn, fn_args)."
        )
    return seq


@contextmanager
def active_sequence_scope(seq: "ActiveSequence") -> Iterator["ActiveSequence"]:
    """Bind ``seq`` as the active sequence for the duration of the body."""
    token = _active_sequence.set(seq)
    try:
        yield seq
    finally:
        _active_sequence.reset(token)


def active_configuration() -> "DeviceConfiguration | None":
    return _active_configuration.get()


@contextmanager
def active_configuration_scope(
    configuration: "DeviceConfiguration",
) -> Iterator["DeviceConfiguration"]:
    token = _active_configuration.set(configuration)
    try:
        yield configuration
    finally:
        _active_configuration.reset(token)


_active_recording: ContextVar["Recording | None"] = ContextVar(
    "iron_active_recording", default=None
)


def active_recording() -> "Recording | None":
    """Return the recording a tool opened, or None when nothing records."""
    return _active_recording.get()


@contextmanager
def active_recording_scope(recording: "Recording") -> Iterator["Recording"]:
    """Bind ``recording`` as the active recording for the duration of the body."""
    token = _active_recording.set(recording)
    try:
        yield recording
    finally:
        _active_recording.reset(token)


@dataclass
class Frame:
    """One core body or control-flow construct being emitted.

    Attributes:
        kind: ``"core"``, ``"for"``, ``"forever"``, ``"if"`` or ``"else"``.
        owner: The Worker for a core, the normalized ``(start, stop, step)``
            bounds for a loop, the condition for an ``if``; otherwise None.
        body: A loop's body block, which ``yield_`` terminates.
        specs: A loop's waited flags per TaskGroup it carries.
        values: What a loop's terminating ``yield_`` passed, if any.
        scope: The scope the active recording notes it as, if any.
    """

    kind: Literal["core", "for", "forever", "if", "else"]
    owner: Any = None
    body: Any = None
    specs: Any = None
    values: list | None = None
    scope: "Scope | None" = None


# A ContextVar (like the active runtime sequence) so concurrent threads or
# async tasks generating designs each see only their own frames.
_frames: ContextVar[tuple[Frame, ...]] = ContextVar("iron_frames", default=())


def frames() -> tuple[Frame, ...]:
    """Return the frames being emitted, innermost last."""
    return _frames.get()


@contextmanager
def frame(
    kind: Literal["core", "for", "forever", "if", "else"],
    owner: Any = None,
    body: Any = None,
    specs: Any = None,
) -> Iterator[Frame]:
    """Push a frame for the duration of the body, noted as a scope while a recording is active."""
    pushed = Frame(kind, owner, body, specs)
    outer = _frames.get()
    recording = _active_recording.get()
    if recording is not None:
        parent = outer[-1].scope if outer else None
        pushed.scope = recording.opened(kind, owner, parent)
    _frames.set(outer + (pushed,))
    try:
        yield pushed
    finally:
        _frames.set(outer)
