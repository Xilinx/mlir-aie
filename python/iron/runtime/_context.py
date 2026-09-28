# _context.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Active-sequence context.

The runtime sequence body runs eagerly inside ``@runtime_sequence`` at resolve
time (mirroring how ``Worker.core_fn`` runs inside ``@core``). The body's
data-movement verbs live on ObjectFifo handles (``inA.prod().fill(...)``), so
they need a way to reach the in-flight sequence without an ``rt`` reference being
threaded through the body signature. This module holds that link as a ContextVar,
so it is well-defined even across nested calls and is always cleared on exit.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING, Iterator

if TYPE_CHECKING:
    from .runtime import ActiveSequence

_active_sequence: ContextVar["ActiveSequence | None"] = ContextVar(
    "iron_active_sequence", default=None
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
            "No active runtime sequence: fill()/drain() and TaskGroup() must be "
            "called from within the function passed to Runtime(seq_fn, fn_args)."
        )
    return seq


def require_sequence(what: str, hint: str | None = None) -> "ActiveSequence":
    """Return the active sequence, or reject ``what`` as used outside one.

    The one check behind every IRON verb that emits into the runtime sequence
    (``Lock.set``, the tile DMA tasks), so they all fail the same way when
    called from a Worker body or after the program was resolved.

    Args:
        what: the verb being checked, e.g. ``"Lock.set on prod"``.
        hint: what to use instead outside the sequence, if there is something.

    Raises:
        RuntimeError: If no runtime sequence body is being emitted.
    """
    seq = _active_sequence.get()
    if seq is None:
        raise RuntimeError(
            f"{what} must be called from within the function passed to "
            "Runtime(seq_fn, fn_args)" + (f"; {hint}." if hint else ".")
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
