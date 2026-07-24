# test_hostruntime_repin.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# RUN: %pytest %s
"""Unit tests for HostRuntime._repin_outputs -- no NPU required.

Tensor.data marks a tensor host-dirty on every access (not just writes, since
numpy gives no separate write hook), so a resident argument re-authored in
place between dispatches is never missed. But that also fires on a plain read
(e.g. peeking a dispatch's output via .numpy()), which would otherwise make
the *next* dispatch's pre-run `.to("npu")` wrongly push the stale host copy
back over fresh device-computed state. _repin_outputs is the compensating
step: reassert "npu" residency on the Out/InOut argument positions right
after a successful dispatch.
"""

from aie.utils.hostruntime.hostruntime import HostRuntime


class _FakeTensor:
    """Minimal stand-in exposing only the `.device` attribute _repin_outputs touches."""

    def __init__(self, device):
        self.device = device


def test_repin_outputs_reasserts_npu_on_output_positions():
    t = _FakeTensor("cpu")  # e.g. left dirty by a prior .numpy() read
    HostRuntime._repin_outputs([t], [True])
    assert t.device == "npu"


def test_repin_outputs_leaves_non_output_positions_alone():
    t = _FakeTensor("cpu")
    HostRuntime._repin_outputs([t], [False])
    assert t.device == "cpu"


def test_repin_outputs_noop_when_output_flags_none():
    """No role info (e.g. a static .mlir file with no In/Out annotations)."""
    t = _FakeTensor("cpu")
    HostRuntime._repin_outputs([t], None)
    assert t.device == "cpu"


def test_repin_outputs_noop_when_output_flags_empty():
    t = _FakeTensor("cpu")
    HostRuntime._repin_outputs([t], [])
    assert t.device == "cpu"


def test_repin_outputs_stops_at_shorter_output_flags():
    """A trailing arg beyond output_flags (e.g. an appended trace buffer) is
    left untouched rather than raising."""
    t_out = _FakeTensor("cpu")
    t_trace = _FakeTensor("cpu")
    HostRuntime._repin_outputs([t_out, t_trace], [True])
    assert t_out.device == "npu"
    assert t_trace.device == "cpu"


def test_repin_outputs_mixed_roles():
    t_in = _FakeTensor("cpu")
    t_out = _FakeTensor("cpu")
    HostRuntime._repin_outputs([t_in, t_out], [False, True])
    assert t_in.device == "cpu"
    assert t_out.device == "npu"
