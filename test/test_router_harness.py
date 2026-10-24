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
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ROUTER_TESTING = ROOT / "utils/router_testing"
sys.path.insert(0, str(ROOT / "test/create-packet-flows/nightly"))
sys.path.insert(0, str(ROUTER_TESTING))
import router_mutation  # noqa: E402
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


def test_link_program_proves_too_many_trees_for_a_cut_unroutable():
    d = rp.canonical(rp.load_design(OVER_CUT))
    assert rp.LinkProgram(d, rp.Analysis(d)).unroutable(60)


def test_link_program_gives_streams_that_cannot_share_an_arbiter_a_channel_each():
    # Five circuit trees and two such packet streams on six channels: seven.
    path = ROOT / "test/create-packet-flows/cut_counts_packets_kept_apart.mlir"
    d = rp.canonical(rp.load_design(path.read_text()))
    assert rp.LinkProgram(d, rp.Analysis(d)).unroutable(60)


@pytest.mark.parametrize("seed", range(5))
def test_link_program_never_proves_a_design_with_a_witness_unroutable(seed):
    case = rp.generate(seed, device="npu1")
    d = case["design"]
    assert not rp.LinkProgram(d, case["analysis"]).unroutable(60)


def test_the_verdict_rejects_what_link_program_proves_unroutable():
    v = rp.verdict(rp.canonical(rp.load_design(OVER_CUT)), True, 0)
    assert v["routable"] is False
    assert "LinkProgram" in v["reason"]


def test_a_spent_shrink_budget_keeps_the_design_as_it_is():
    d = rp.load_design(OVER_CUT)
    holds = router_shrink.holds_fn(OVERUSED, True)
    assert holds(d)
    small, finished = router_shrink.shrink(d, holds, budget_s=0)
    assert not finished
    assert holds.calls[0] == 1
    assert router_shrink.size(small) == router_shrink.size(d)


# With no aie-opt to run, any judgement raises, so these see whether
# shrink_failure routes the design at all.
def test_a_spent_shared_budget_routes_nothing_before_shrinking(monkeypatch, tmp_path):
    d = rp.load_design(OVER_CUT)
    monkeypatch.setenv("AIE_OPT", str(tmp_path / "no-aie-opt"))
    small, finished = router_mutation.shrink_failure(
        "mutator", "port-cap", (d, "route"), True, 0, False, time.monotonic()
    )
    assert not finished
    assert small is d


def test_a_live_shared_budget_judges_before_shrinking(monkeypatch, tmp_path):
    d = rp.load_design(OVER_CUT)
    monkeypatch.setenv("AIE_OPT", str(tmp_path / "no-aie-opt"))
    with pytest.raises(FileNotFoundError):
        router_mutation.shrink_failure(
            "mutator", "port-cap", (d, "route"), True, 0, False, time.monotonic() + 60
        )


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


def test_a_router_failure_leaves_the_input_and_a_command_to_reproduce_it(
    router_hw, tmp_path
):
    out = tmp_path / "out"
    out.mkdir()
    # Seven circuit flows over six channels: a real, unroutable design.
    variant = router_hw.Variant(
        name="over-cut",
        design=OVER_CUT,
        pinned=None,
        expect="PASS",
        options="",
        note="",
    )
    with pytest.raises(RuntimeError, match=r"the router failed") as excinfo:
        router_hw.route(variant, out, timeout_s=60)
    assert (out / "input.mlir").read_text() == OVER_CUT
    repro = str(excinfo.value).splitlines()[-1]
    assert repro.startswith("aie-opt ") and str(out / "input.mlir") in repro
    result = subprocess.run(repro, shell=True, capture_output=True, text=True)
    assert result.returncode != 0
    assert "cross from row 2 to row 3" in result.stderr


def test_a_route_timeout_also_leaves_the_input_and_a_command(router_hw, tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    variant = router_hw.Variant(
        name="t",
        design="module { aie.device(npu1_1col) {} }",
        pinned=None,
        expect="PASS",
        options="",
        note="",
    )
    # A timeout real aie-opt startup alone cannot meet, to force the timeout
    # branch deterministically rather than waiting out a genuine hang.
    with pytest.raises(RuntimeError, match=r"did not finish within"):
        router_hw.route(variant, out, timeout_s=0.0001)
    assert (out / "input.mlir").exists()


def test_a_compile_failure_names_the_command_to_reproduce_it(router_hw, tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "routed.mlir").write_text("module { aie.device(npu1_1col) {} }")
    with pytest.raises(RuntimeError, match=r"aiecc failed") as excinfo:
        router_hw.compile_variant(out, peano="/no/such/peano", timeout_s=60)
    repro = str(excinfo.value).splitlines()[-1]
    assert str(out) in repro and "routed.mlir" in repro
    result = subprocess.run(repro, shell=True, capture_output=True, text=True)
    assert result.returncode != 0
