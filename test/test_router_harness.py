# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# RUN: %pytest %s

"""The router's model and shrinker, with the real aie-opt; no NPU required."""

import argparse
import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ROUTER_TESTING = ROOT / "utils/router_testing"
sys.path.insert(0, str(ROOT / "test/create-packet-flows/nightly"))
sys.path.insert(0, str(ROUTER_TESTING))
import router_properties as rp  # noqa: E402  (paths set just above)
import router_shrink  # noqa: E402

# A case module with one variant, for router_hw.py's argument validation:
# these never route, compile or touch an NPU.
HW_CASE = """\
from hw_case import HwCase, Variant


class Case(HwCase):
    name = "fixture"
    description = "a fixture case; never actually routed"
    outputs = []

    def variants(self):
        return [
            Variant(
                name="real",
                design="",
                pinned=None,
                expect="PASS",
                options="",
                note="",
            )
        ]

    def check(self, outputs):
        return []
"""

# The task-complete-token route aiecc adds at a shim, and a flow beside it.
PRIORITY = """module {
  aie.device(npu1_1col) {
    %t00 = aie.tile(0, 0)
    %t02 = aie.tile(0, 2)
    aie.packet_flow(15) { aie.packet_source<%t00, TileControl : 0> aie.packet_dest<%t00, South : 0> } {keep_pkt_header = true, priority_route = true}
    aie.packet_flow(3) { aie.packet_source<%t02, DMA : 0> aie.packet_dest<%t00, DMA : 0> }
  }
}"""

# Seven circuit flows up across the six channels from (0, 2) to (0, 3), and
# one that need not cross.
OVER_CUT = """module {
  aie.device(npu1_1col) {
    %t01 = aie.tile(0, 1)
    %t02 = aie.tile(0, 2)
    %t03 = aie.tile(0, 3)
    %t04 = aie.tile(0, 4)
    %t05 = aie.tile(0, 5)
    aie.flow(%t01, DMA : 0, %t03, DMA : 0)
    aie.flow(%t01, DMA : 1, %t03, DMA : 1)
    aie.flow(%t01, DMA : 2, %t04, DMA : 0)
    aie.flow(%t01, DMA : 3, %t04, DMA : 1)
    aie.flow(%t01, DMA : 4, %t05, DMA : 0)
    aie.flow(%t01, DMA : 5, %t05, DMA : 1)
    aie.flow(%t02, DMA : 0, %t03, Core : 0)
    aie.flow(%t02, DMA : 1, %t01, DMA : 0)
  }
}"""
OVERUSED = r"cross from row 2 to row 3, which takes 7 channels across, and it has 6"


@pytest.fixture(scope="module", autouse=True)
def aie_opt():
    assert shutil.which("aie-opt"), "aie-opt is not on PATH"


@pytest.fixture(scope="module")
def priority():
    d = rp.canonical(rp.load_design(PRIORITY))
    routed = rp.aie_opt(d.emit(), True)
    assert routed.returncode == 0, routed.stderr
    return d, routed.stdout


def test_the_model_accepts_the_routers_overlay_marks(priority):
    d, routed = priority
    assert "aie.is_ctrl_pkt_overlay" in routed
    assert rp.verify(d, rp.Analysis(d), routed, True)[0] == []


def test_the_model_rejects_a_priority_hop_without_its_mark(priority):
    d, routed = priority
    unmarked = routed.replace(
        "{aie.is_ctrl_pkt_overlay, keep_pkt_header = true}",
        "{keep_pkt_header = true}",
    )
    assert unmarked != routed
    problems = rp.verify(d, rp.Analysis(d), unmarked, True)[0]
    assert "priority id 15 at (0, 0) South:0 not ctrl" in problems


def test_the_model_rejects_a_mark_no_priority_flow_takes(priority):
    d, routed = priority
    lines = routed.splitlines()
    k = next(
        i
        for i, line in enumerate(lines)
        if "aie.masterset(South : 2" in line and "aie.is_ctrl" not in line
    )
    lines[k] += " {aie.is_ctrl_pkt_overlay}"
    problems = rp.verify(d, rp.Analysis(d), "\n".join(lines), True)[0]
    assert "ctrl masterset South:2 at (0, 0) carries no priority flow" in problems


def test_a_spent_shrink_budget_keeps_the_design_as_it_is():
    d = rp.load_design(OVER_CUT)
    holds = router_shrink.holds_fn(OVERUSED, True)
    assert holds(d)
    small, finished = router_shrink.shrink(d, holds, budget_s=0)
    assert not finished
    assert holds.calls[0] == 1
    assert router_shrink.size(small) == router_shrink.size(d)


def test_an_unbounded_shrink_finishes_on_what_fails():
    d = rp.load_design(OVER_CUT)
    holds = router_shrink.holds_fn(OVERUSED, True)
    small, finished = router_shrink.shrink(d, holds)
    assert finished
    assert len(small.flows) == 7
    assert holds(small)


@pytest.fixture
def hw_case(tmp_path):
    path = tmp_path / "fixture.py"
    path.write_text(HW_CASE)
    return path


@pytest.fixture(scope="module")
def router_hw():
    # router_hw.py imports pyxrt at module scope, an optional dependency this
    # file's other tests do not need; load it lazily so only these two tests
    # require it, and skip them where XRT is not installed.
    pytest.importorskip("pyxrt")
    spec = importlib.util.spec_from_file_location(
        "router_hw", ROUTER_TESTING / "router_hw.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_an_unknown_variant_name_fails_instead_of_running_nothing(
    router_hw, hw_case, tmp_path
):
    args = argparse.Namespace(
        case=hw_case,
        variant=["typo"],
        runs=20,
        process_timeout_s=120,
        work=tmp_path / "work",
        verdicts=None,
        peano="unused",
    )
    assert router_hw.run_case(args) != 0
    assert not (tmp_path / "work").exists()


def test_a_nonpositive_runs_count_is_rejected(router_hw, hw_case):
    result = subprocess.run(
        [
            sys.executable,
            router_hw.__file__,
            "run",
            str(hw_case),
            "--runs",
            "0",
            "--peano",
            "unused",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "--runs" in result.stderr
