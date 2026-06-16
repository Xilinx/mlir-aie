# _context.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc.
"""Active-sequence context for the eager runtime-sequence runner.

In the eager model the body of a runtime sequence runs *inside* the
``@runtime_sequence`` builder. Data-movement verbs live on the objects they
act on (``ObjectFifoHandle.fill/drain``, ``Buffer.write``) and ``TaskGroup``
is constructed free-standing. All of them need to reach the in-flight sequence
to emit ops, allocate BD ids, and register themselves for end-of-sequence
finalization. They find it here rather than by holding a reference to the
``Runtime``, which keeps the call sites free of plumbing and avoids importing
``Runtime`` into the dataflow / buffer modules (circular).
"""

from contextlib import contextmanager
from contextvars import ContextVar

_ACTIVE_SEQUENCE: ContextVar = ContextVar("active_runtime_sequence", default=None)


def active_sequence():
    """The ActiveSequence currently being built, or raise if called outside one."""
    seq = _ACTIVE_SEQUENCE.get()
    if seq is None:
        raise RuntimeError(
            "Runtime data-movement (fill/drain/write/TaskGroup) may only be used "
            "inside the function passed to Runtime.sequence(...)."
        )
    return seq


@contextmanager
def active_sequence_scope(seq):
    token = _ACTIVE_SEQUENCE.set(seq)
    try:
        yield seq
    finally:
        _ACTIVE_SEQUENCE.reset(token)
