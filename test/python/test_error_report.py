# test_error_report.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# RUN: %pytest %s
"""cases.error_report on every kernel case, fed its own reference (no NPU).

``--report-error`` runs it on device output; here the "device output" is the
contract's reference laid out as the device would store it, so every case's
decoding, trimming and broadcasting is exercised without hardware.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
from aie.iron.device import from_name
from aie.utils.hostruntime import set_current_device
from ml_dtypes import bfloat16

sys.path.insert(0, str(Path(__file__).resolve().parent / "npu"))
import cases  # noqa: E402
from kernel_cases import CASES  # noqa: E402


def _stored(fn, ref, calls):
    """What the device would leave in each output buffer for ``ref``.

    A kernel with no streamed input computes one tile, repeated every call.
    """
    c = fn.contract
    multiple = len(c.out_indices) > 1
    stored = []
    for i, r in zip(c.out_indices, ref if multiple else (ref,)):
        layout = c.layouts[i] if c.layouts else None
        tiles = np.asarray(r).reshape(-1, *layout.shape) if layout else r
        if layout and len(tiles) == 1 < calls:
            tiles = np.repeat(tiles, calls, axis=0)
        stored.append(layout.encode(tiles).ravel() if layout else tiles)
    return tuple(stored) if multiple else stored[0]


@pytest.mark.parametrize("device", ["npu1", "npu2"])
def test_every_case_reports_its_own_reference(device):
    set_current_device(from_name(device, n_cols=1))
    try:
        reported = 0
        for case in CASES:
            if case.devices and device not in case.devices:
                continue
            with cases.device_for(case.devices):
                fn = case.fn()
                inputs = cases.inputs_for(case, "random", np.random.default_rng(0))
                ref = fn.expected(inputs, scalars=case.scalars)
                entries = cases.error_report(
                    fn,
                    _stored(fn, ref, case.calls),
                    inputs,
                    calls=case.calls,
                    scalars=case.scalars,
                )
            for e in entries:
                # The contract's own answer is exactly as far from the
                # reference measured against as the report says it is.
                assert e["max_ulp"] == e["reference_max_ulp"], case.name
                assert e["reference"] in ("float64", "float32", "bfloat16"), case.name
                reported += 1
        assert reported > 100
    finally:
        set_current_device(None)


def test_an_error_is_found_where_it_is():
    set_current_device(from_name("npu2", n_cols=1))
    try:
        case = next(c for c in CASES if c.factory == "gelu")
        fn = case.fn()
        inputs = cases.inputs_for(case, "random", np.random.default_rng(0))
        got = fn.expected(inputs, scalars=case.scalars).copy()
        assert got.dtype == bfloat16
        flat = got.reshape(-1).view(np.uint16)
        flat[7] += 1000  # 1000 bf16 ulps away, in the same binade or the next
        e = cases.error_report(fn, got, inputs, calls=case.calls)[0]
        assert e["reference"] == "float64" and e["n"] == got.size
        assert e["worst_index"] == 7 and e["max_ulp"] >= 999
        assert e["not_correctly_rounded"] >= 1
    finally:
        set_current_device(None)
