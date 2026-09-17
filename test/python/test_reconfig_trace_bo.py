# test_reconfig_trace_bo.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# RUN: %pytest %s
"""Unit test for the reconfiguration trace-buffer BO provisioning -- no NPU.

A folded design that enables hardware trace declares
``aie.trace.host_config {buffer_size = N}`` in its runtime_sequence; the fold
appends a dedicated N-byte trace-buffer arg at the tail of the design's own
tensor args (before any control-packet buffer). ``Reconfiguration.compile()``
surfaces N as ``FullElf.trace_buffer_bytes`` so a runlist host allocates + binds
that BO at the design's arg tail, keeping the control BO at the following index
(see the ``FullElf`` docstring).

These exercise the pure text->size extraction directly, without running aiecc:
no trace -> None, one trace design -> its byte count, and the >1-design guard
(multi-design trace folding is unsupported and must fail loud, not mis-bind).
"""

import pytest

from aie.utils.compile.reconfiguration import Reconfiguration


def _design(text, name="d"):
    """A staged-design tuple as Reconfiguration.add() builds:
    (name, mlir_text, external_kernels, compilable). Only index 1 is read."""
    return (name, text, [], None)


_TRACE_SEQ = """
module {
  aie.device(npu1_1col) {
    aiex.runtime_sequence(%arg0: memref<16xi32>) {
      aie.trace.host_config {buffer_size = 16384 : i32}
    }
  }
}
"""

_NO_TRACE_SEQ = """
module {
  aie.device(npu1_1col) {
    aiex.runtime_sequence(%arg0: memref<16xi32>) {
    }
  }
}
"""


def test_no_trace_returns_none():
    """No design declares aie.trace.host_config -> no trace BO to provision."""
    assert Reconfiguration._trace_buffer_bytes([_design(_NO_TRACE_SEQ)]) is None
    assert Reconfiguration._trace_buffer_bytes([]) is None


def test_single_trace_extracts_buffer_size():
    """The declared buffer_size is surfaced verbatim as the BO byte count."""
    assert Reconfiguration._trace_buffer_bytes([_design(_TRACE_SEQ)]) == 16384


def test_single_trace_among_non_trace_designs():
    """One trace design folded with plain designs still surfaces its size."""
    designs = [
        _design(_NO_TRACE_SEQ, "a"),
        _design(_TRACE_SEQ, "b"),
        _design(_NO_TRACE_SEQ, "c"),
    ]
    assert Reconfiguration._trace_buffer_bytes(designs) == 16384


def test_buffer_size_not_first_attr():
    """buffer_size is extracted even when other attrs precede it in the braces."""
    text = _TRACE_SEQ.replace(
        "{buffer_size = 16384 : i32}",
        "{sym = @t, buffer_size = 8192 : i32}",
    )
    assert Reconfiguration._trace_buffer_bytes([_design(text)]) == 8192


def test_multi_trace_designs_raise():
    """Multi-design trace folding is unsupported -- fail loud, not mis-bind."""
    designs = [_design(_TRACE_SEQ, "a"), _design(_TRACE_SEQ, "b")]
    with pytest.raises(RuntimeError, match="single"):
        Reconfiguration._trace_buffer_bytes(designs)
