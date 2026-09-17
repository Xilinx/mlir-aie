# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# RUN: %pytest %s
"""Only declared whole-call trace regions may become benchmark cycle metrics."""

from types import SimpleNamespace

import pytest
from aie.iron.algorithms import kernel_design as kd


def test_unprofiled_contract_does_not_run_a_trace(tmp_path):
    fn = SimpleNamespace(contract=SimpleNamespace(trace_cycles=False))
    assert (
        kd.cycles_per_call(None, [], 1, int, fn=fn, trace_size=1024, workdir=tmp_path)
        == []
    )


@pytest.mark.parametrize(
    "intervals,valid",
    [([17, 19], True), ([], False), ([17], False), ([1, 17, 1, 19], False)],
)
def test_cycle_protocol_requires_one_complete_pair_per_call(
    monkeypatch, tmp_path, intervals, valid
):
    fn = SimpleNamespace(name="profiled", contract=SimpleNamespace(trace_cycles=True))
    cfg = SimpleNamespace(
        physical_mlir_path="physical.mlir", trace_to_json=lambda *args: None
    )
    monkeypatch.setattr(kd, "TraceConfig", lambda **kwargs: cfg)
    monkeypatch.setattr(kd, "upload", lambda *args, **kwargs: ([], object()))
    monkeypatch.setattr(kd, "get_cycles_summary", lambda path: [(0, *intervals)])

    def run():
        return kd.cycles_per_call(
            lambda *args, **kwargs: None,
            [],
            1,
            int,
            fn=fn,
            trace_size=1024,
            workdir=tmp_path,
            calls=2,
        )

    if valid:
        assert run() == intervals
    else:
        with pytest.raises(RuntimeError, match="whole-call trace intervals"):
            run()
