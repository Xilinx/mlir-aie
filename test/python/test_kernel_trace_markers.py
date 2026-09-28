# test_kernel_trace_markers.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# RUN: %pytest %s
# REQUIRES: peano
"""Every library kernel's trace markers match its contract's ``trace`` (no NPU).

A benchmark cycle count is one ``event0()`` -> ``event1()`` interval per
call, so a kernel whose markers sit in a sibling function, around an inner
loop, or behind an early return charts the wrong number or none. Each build
is compiled to -O2 IR with the library's own flags and its entry symbol
classified by ``remarks.trace_markers``; the result must equal the declared
``Trace``. The builds are every factory at its defaults and ``.dtypes``
entries, the initializers they declare, and each device case whose entry
symbol those do not already cover. A case that changes only the code inside
an already-checked symbol (a tile-size define) is not compiled again.
"""

import concurrent.futures
import os
import sys
from pathlib import Path

import pytest
from aie.iron.device import NPU1Col1, NPU2Col1
from aie.iron.kernels import Trace
from aie.utils import config
from aie.utils.compile.remarks import entry_symbol, kernel_builds, trace_shape
from aie.utils.hostruntime import set_current_device

sys.path.insert(0, str(Path(__file__).parent / "npu"))
from kernel_cases import CASES  # noqa: E402

# The only builds whose calls cannot be timed; each contract says why.
UNTIMED = {"set_rounding"}


def _builds(device, generation):
    """``[(name, ExternalFunction)]``: every factory build, then the new symbols."""
    set_current_device(device)
    out, seen = [], set()

    def add(name, ef, only_new):
        key = (ef.source_file or ef.source_string, entry_symbol(ef))
        if only_new and key in seen:
            return
        seen.add(key)
        out.append((name, ef))
        for i, init in getattr(ef.contract, "initializers", ()):
            add(f"{name}/initializer[{i}]", init(ef), only_new=True)

    for name, ef in kernel_builds():
        add(name, ef, only_new=False)
    for case in CASES:
        if not case.devices or generation in case.devices:
            add(case.name, case.fn(), only_new=True)
    return out


@pytest.fixture(
    scope="module",
    params=[("aie2", "npu1"), ("aie2p", "npu2")],
    ids=["aie2", "aie2p"],
)
def classified(request, tmp_path_factory):
    target, generation = request.param
    if not (Path(config.aie_runtime_lib_dir()) / target.upper()).is_dir():
        pytest.skip(f"this build has no aie_runtime_lib/{target.upper()}")
    device = NPU1Col1() if target == "aie2" else NPU2Col1()
    try:
        builds = _builds(device, generation)
    finally:
        set_current_device(None)
    workdir = tmp_path_factory.mktemp(target)

    def one(indexed):
        i, (_, ef) = indexed
        cell = workdir / f"build{i}"
        cell.mkdir()
        return trace_shape(ef, target, cell)

    with concurrent.futures.ThreadPoolExecutor(os.cpu_count() or 1) as pool:
        shapes = list(pool.map(one, enumerate(builds)))
    return target, [(name, ef, shape) for (name, ef), shape in zip(builds, shapes)]


def _found(ef, found) -> str:
    where = f"{entry_symbol(ef)} in {ef.source_file or '<inline source>'}"
    if found == "whole_call":
        return f"{where} brackets the whole call"
    if found == "none":
        return f"{where} reaches no event0()/event1()"
    return f"{where}: {found}"


def test_markers_match_the_declared_trace(classified):
    target, results = classified
    wrong = []
    for name, ef, found in results:
        # A contract-less build cannot declare an exemption, so it must be timed.
        trace = ef.contract.trace if ef.contract else Trace.whole_call()
        if trace is None:
            wrong.append(f"{name}: the contract declares no trace")
        elif trace.shape == "partial":
            if found in ("whole_call", "none"):
                wrong.append(f"{name}: declared partial, but {_found(ef, found)}")
        elif found != trace.shape:
            wrong.append(f"{name}: declared {trace.shape}, but {_found(ef, found)}")
    assert not wrong, f"{target}:\n" + "\n".join(wrong)


def test_untimed_builds_are_the_listed_ones(classified):
    _, results = classified
    untimed = {
        name.split("/")[0]: ef.contract.trace.reason
        for name, ef, _ in results
        if ef.contract and ef.contract.trace and ef.contract.trace.shape != "whole_call"
    }
    assert set(untimed) <= UNTIMED, f"newly untimed: {untimed}"
