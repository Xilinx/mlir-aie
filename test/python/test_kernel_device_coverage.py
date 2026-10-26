# test_kernel_device_coverage.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# RUN: %pytest %s
"""Every kernel build a device can make runs on that device in CI (no NPU).

The device tests run each case in ``npu/kernel_cases.py`` on the NPUs its
``devices`` allows. A case limited to one NPU whose factory also builds for
the other leaves that build untested there, unless another case already
runs the same build (``recipe_key``) on it. A factory refuses a device it has
no kernel for with ``NotImplementedError`` or ``ValueError``.

Factories the generic harness does not judge name the test that does in
``test_kernel_contracts.NOT_JUDGED``; that test must run on every NPU the
factory builds for.
"""

import re
import sys
from pathlib import Path

import pytest
from aie.iron import kernels
from aie.iron.device import from_name
from aie.utils.hostruntime import set_current_device
from ml_dtypes import bfloat16

sys.path.insert(0, str(Path(__file__).parent / "npu"))
from kernel_cases import CASES  # noqa: E402
from test_kernel_contracts import NOT_JUDGED  # noqa: E402

DEVICES = ("npu1", "npu2")


def _build(device: str, factory: str, kwargs: dict):
    """The kernel ``factory`` builds for ``device``, or None if it has none there."""
    with set_current_device(from_name(device)):
        try:
            return getattr(kernels, factory)(**kwargs)
        except (NotImplementedError, ValueError):
            return None


@pytest.mark.parametrize("device", DEVICES)
def test_every_build_a_device_makes_runs_on_it(device):
    runs = {
        _build(device, c.factory, c.kwargs).recipe_key
        for c in CASES
        if c.supported_on(device)
    }
    untested = [
        c.name
        for c in CASES
        if not c.supported_on(device)
        and (ef := _build(device, c.factory, c.kwargs)) is not None
        and ef.recipe_key not in runs
    ]
    assert not untested, (
        f"cases {device} can build but never runs (drop their devices= "
        f"restriction, or make the factory refuse {device}): {untested}"
    )


def test_mm_refuses_a_tile_its_kernel_cannot_step():
    # mm_aie2.h's bf16 loops step 16 rows; mm_aie2p.h's step 8.
    tile = dict(
        dim_m=8, dim_k=64, dim_n=16, input_dtype=bfloat16, output_dtype=bfloat16
    )
    assert _build("npu1", "mm", tile) is None
    assert _build("npu2", "mm", tile) is not None
    assert _build("npu1", "mm", tile | dict(dim_m=16)) is not None


@pytest.mark.parametrize("device", DEVICES)
def test_not_judged_tests_run_where_their_kernels_build(device):
    # A RUN line of the test itself (%s), or of a lit file that runs it by name.
    runs = [
        (path, line)
        for path in Path(__file__).parent.rglob("*")
        if path.suffix in (".py", ".test")
        for line in path.read_text().splitlines()
        # [:] keeps lit from reading this line as a directive of this file.
        if re.match(r"# RUN[:]", line) and f"%run_on_{device}_" in line
    ]
    missing = []
    for name, reason in NOT_JUDGED.items():
        if _build(device, name, {}) is None:
            continue
        for test in re.findall(r"\b(test_\w+\.py)\b", reason):
            if not any(
                (path.name == test and "%s" in line) or f"/{test}" in line
                for path, line in runs
            ):
                missing.append(f"{name}: {test}")
    assert not missing, f"no {device} RUN line in the test that judges: {missing}"
