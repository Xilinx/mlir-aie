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
    from ..configuration import DeviceConfiguration
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
