# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# RUN: %pytest %s

"""The router's model and shrinker, with the real aie-opt; no NPU required."""

import argparse
import dis
import functools
import importlib.util
import inspect
import json
import random
import shutil
import signal
import subprocess
import sys
import time
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
ROUTER_TESTING = ROOT / "utils/router_testing"
sys.path.insert(0, str(ROOT / "test/create-packet-flows/nightly"))
sys.path.insert(0, str(ROUTER_TESTING))
import router_mutation  # noqa: E402
import router_properties as rp  # noqa: E402  (paths set just above)
import router_shrink  # noqa: E402
from hw_case import HwCase  # noqa: E402

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
# Two packet sources sharing two receivers: a strict witness exists.
SHARED = """module {
  aie.device(npu1_1col) {
    %t02 = aie.tile(0, 2)
    %t03 = aie.tile(0, 3)
    %t04 = aie.tile(0, 4)
    %t05 = aie.tile(0, 5)
    aie.packet_flow(1) { aie.packet_source<%t02, DMA : 0> aie.packet_dest<%t03, DMA : 0> aie.packet_dest<%t04, DMA : 0> }
    aie.packet_flow(2) { aie.packet_source<%t05, DMA : 0> aie.packet_dest<%t03, DMA : 0> aie.packet_dest<%t04, DMA : 0> }
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


@pytest.mark.parametrize("receiver", [rp.CORE, rp.DMA])
@pytest.mark.parametrize("sender", [rp.CORE, rp.DMA])
def test_wait_chain_through_a_stream_receiver(receiver, sender):
    d = rp.Design("npu1")
    streams = [
        rp.Stream((0, 2, rp.DMA, 0), (0, 3, receiver, 0)),
        rp.Stream((0, 4, sender, 0), (0, 3, receiver, 1 if receiver == rp.CORE else 0)),
    ]
    graph = rp.WaitGraph(d, streams, rp.Volumes(d, streams))
    s = graph.agent_at(streams[0].src, True)
    r = graph.agent_at(streams[0].dst, False)
    t = graph.agent_at(streams[1].src, True)
    expected = [] if receiver == rp.DMA and sender == rp.DMA else [s, r, t]
    assert graph.wait_chain([s], [t], []) == expected
    assert graph.wait_chain([s], [t], [r]) == []
    assert graph.wait_chain([r], [t], []) == [r, t]


@pytest.mark.parametrize("kind", [rp.WaitGraph.LOCK, rp.WaitGraph.HOST])
def test_wait_chain_preserves_non_stream_dependencies_on_a_dma_sender(kind):
    d = rp.Design("npu1")
    streams = [
        rp.Stream((0, 2, rp.DMA, 0), (0, 3, rp.DMA, 0)),
        rp.Stream((0, 4, rp.DMA, 0), (0, 3, rp.DMA, 0)),
    ]
    graph = rp.WaitGraph(d, streams, rp.Volumes(d, streams))
    s = graph.agent_at(streams[0].src, True)
    r = graph.agent_at(streams[0].dst, False)
    t = graph.agent_at(streams[1].src, True)
    graph.add_edge(r, t, kind)
    assert graph.wait_chain([s], [t], []) == [s, r, t]


def test_wait_chain_reaches_a_dma_receiver_again_free():
    # S's data reaches R first, taking, where T is no wait of R's; R reached
    # again through C1, C2 and B is free, and waits on T.
    d = rp.Design("npu1")
    streams = [
        rp.Stream((0, 2, rp.DMA, 0), (0, 3, rp.DMA, 0)),
        rp.Stream((0, 4, rp.DMA, 0), (0, 3, rp.DMA, 0)),
    ]
    graph = rp.WaitGraph(d, streams, rp.Volumes(d, streams))
    s = graph.agent_at(streams[0].src, True)
    r = graph.agent_at(streams[0].dst, False)
    t = graph.agent_at(streams[1].src, True)
    c1, c2, b = (
        graph.get_or_create((1, row), graph.CHANNEL, rp.MM2S, 0) for row in (2, 3, 4)
    )
    for p, q in [(s, c1), (c1, c2), (c2, b), (b, r), (r, b)]:
        graph.add_edge(p, q, graph.LOCK)
    assert graph.wait_chain([s], [t], []) == [s, c1, c2, b, r, t]
    assert graph.wait_chain([s], [t], [c1]) == []


def test_the_model_accepts_the_routers_priority_marks(priority):
    d, routed = priority
    assert "aie.priority_route" in routed
    assert rp.verify(d, rp.Analysis(d), routed, True)[0] == []


def test_the_model_rejects_a_priority_hop_without_its_mark(priority):
    d, routed = priority
    unmarked = routed.replace(
        "{aie.priority_route, keep_pkt_header = true}",
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
        if "aie.masterset(South : 2" in line and "aie.priority_route" not in line
    )
    lines[k] += " {aie.priority_route}"
    problems = rp.verify(d, rp.Analysis(d), "\n".join(lines), True)[0]
    assert "ctrl masterset South:2 at (0, 0) carries no priority flow" in problems


def test_the_mutation_nightly_verifies_a_strict_routing_too(priority):
    d, _ = priority
    strict = rp.aie_opt(d.emit(), True, strict=True)
    assert strict.returncode == 0, strict.stderr
    assert router_mutation.judge_strict(d, "route", True, strict)[0] == "ok"
    # The shim mux no longer takes id 3 down to its DMA channel.
    cut = strict.stdout.replace("aie.connect<North : 2, DMA : 0>", "")
    assert cut != strict.stdout
    broken = subprocess.CompletedProcess(strict.args, 0, cut, strict.stderr)
    outcome, detail = router_mutation.judge_strict(d, "route", True, broken)
    assert outcome == "illegal", detail
    assert "id 3 from (0, 2) DMA:0" in detail


def test_a_strict_rejection_without_a_witness_says_so(priority):
    d, _ = priority
    assert rp.strict_witness(d, rp.Analysis(d)) is None
    rejected = subprocess.CompletedProcess([], 1, "", "error: no legal routing")
    assert router_mutation.judge_strict(d, "route", True, rejected) == (
        "ok",
        "rejected strict, and no routing without a hold cycle is known",
    )


def test_a_packet_owed_twice_and_received_once_counts_as_missing():
    a, b = [1, 2], [3, 4]
    got = np.array([a, b, b], dtype=np.int32)
    assert HwCase().check_packets([got], [[a, a, b]], 2) == [
        "output 0: packets differ from the expected set (1 missing, 1 unexpected)"
    ]
    assert HwCase().check_packets([np.array([b, a], dtype=np.int32)], [[a, b]], 2) == []


def test_a_batch_gets_one_limit_not_one_per_design():
    assert list(inspect.signature(rp.run_end).parameters) == ["deadline"]
    assert "ROUTE_TIMEOUT_S" in code_names(rp.run_end.__code__)
    start = time.monotonic()
    assert rp.run_end() - start <= rp.ROUTE_TIMEOUT_S + 1
    assert not {"SPLIT", "count"} & code_names(rp.aie_opt.__code__)


def test_every_judged_router_run_reads_a_crash_as_a_crash():
    crashed = subprocess.CompletedProcess(["aie-opt"], -signal.SIGSEGV, "", "")
    assert rp.run_failure(crashed)[0] == "crash"
    spent = rp.aie_opt("module {}", deadline=time.monotonic())
    assert rp.run_failure(spent)[0] == "timeout"
    # judge() routes twice: the design the verifier rejects, and any other.
    for fn, runs in ((router_mutation.judge, 2), (router_mutation.judge_strict, 1)):
        calls = [i for i in dis.get_instructions(fn) if i.argval == "run_failure"]
        assert len(calls) == runs, fn.__name__


def test_the_unknown_tier_names_a_timeout_a_timeout():
    for fn in (rp.main, rp.model_run):
        assert "run_failure" in code_names(fn.__code__), fn.__name__
    spent = rp.aie_opt("module {}", deadline=time.monotonic())
    assert rp.run_failure(spent)[0] == "timeout"


def test_a_router_timeout_is_not_called_a_crash(priority):
    d, _ = priority
    slow = rp.aie_opt(d.emit(), True, strict=True, deadline=time.monotonic())
    assert slow.returncode == rp.TIMEOUT
    outcome, detail = router_mutation.judge_strict(d, "route", True, slow)
    assert outcome == "timeout", detail


def test_a_timeout_on_a_design_the_verifier_rejects_is_a_timeout():
    d = rp.load_design(PRIORITY)
    d.packet_flows.append(d.packet_flows[-1])
    assert router_mutation.must_reject(d) == "a packet flow twice"
    with pytest.raises(Exception, match="duplicates an earlier packet flow"):
        rp.canonical(d)
    outcome, detail = router_mutation.judge(
        d, "route", True, 0, deadline=time.monotonic()
    )
    assert outcome == "timeout", detail


def code_names(code):
    names = set(code.co_names)
    for const in code.co_consts:
        if isinstance(const, types.CodeType):
            names |= code_names(const)
    return names


# Seconds the slowest judged design takes on a busy hosted runner.
SLOWEST_JUDGED_S = 258


def test_the_nightly_limit_leaves_room_over_the_slowest_judged_design():
    assert SLOWEST_JUDGED_S <= 0.75 * rp.ROUTE_TIMEOUT_S


def test_no_router_run_chooses_its_own_limit():
    for runner in (rp.aie_opt, rp.find_flows):
        params = inspect.signature(runner).parameters
        assert "timeout" not in params and "deadline" in params
        assert "run_end" in code_names(runner.__code__)
    for module in (rp, router_mutation, router_shrink):
        for name, fn in vars(module).items():
            if not inspect.isfunction(fn) or fn.__module__ != module.__name__:
                continue
            if fn in (rp.aie_opt, rp.find_flows):
                continue
            assert "subprocess" not in code_names(fn.__code__), name
    for name, value in vars(router_mutation).items():
        assert not isinstance(value, functools.partial), name
        if inspect.isfunction(value):
            assert value.__module__ != rp.__name__, name


def test_shrinking_stops_a_judgement_at_its_budget():
    seed = 207560359
    d = rp.generate(seed, device="npu2")["design"]
    ctx = {"shared": set(), "witness": lambda: None}
    rng = random.Random(f"{seed}-port-cap")
    _, e, expect = router_mutation.MUTATORS["port-cap"](rng, d, ctx)
    start = time.monotonic()
    _, finished = router_mutation.shrink_failure(
        "mutator", "port-cap", (e, expect), True, seed, False, start + 5
    )
    assert not finished
    # The router alone takes over a minute on this design.
    assert time.monotonic() - start < 30


def test_a_timeout_in_the_models_router_runs_is_a_timeout(priority):
    d, _ = priority
    with pytest.raises(rp.RouterTimeout):
        rp.priority_trees(d, {}, deadline=time.monotonic())
    outcome, detail = router_mutation.judge(
        d, "model", True, 0, deadline=time.monotonic()
    )
    assert outcome == "timeout", detail
    assert detail.startswith("model: routing the prioritized flows alone"), detail


def test_a_seed_remembers_its_prioritized_routing_timed_out(priority):
    d, _ = priority
    trees = {}
    with pytest.raises(rp.RouterTimeout):
        rp.priority_trees(d, trees, deadline=time.monotonic())
    with pytest.raises(rp.RouterTimeout):
        rp.priority_trees(d, trees)
    assert rp.priority_trees(d, {}) is not None


def test_the_generator_names_the_seeds_the_model_could_not_decide():
    got = list(rp.generated_cases(0, 5, "npu1", deadline=time.monotonic()))
    assert [(seed, outcome) for seed, _, outcome in got if outcome] == [
        (0, "timeout"),
        (4, "timeout"),
    ]


def test_the_properties_report_keeps_the_generators_undecided_seeds(tmp_path):
    path = tmp_path / "npu1-properties.json"
    run = rp.PropertiesReport(path, "npu1", 0, 3)
    run.seed_tried(0, "timeout")
    run.seed_tried(1, None)
    partial = json.loads(path.read_text())
    assert (partial["partial"], partial["seeds_tried"]) == (True, 2)
    assert partial["undecided"] == ["seed 0: timeout"]
    run.add("unroutable", True, "0 left to the model")
    run.add("strict-completeness", True, "0 undecided")
    run.write(partial=False, counts={})
    final = json.loads(path.read_text())
    assert (final["partial"], final["undecided"]) == (False, ["seed 0: timeout"])


def test_a_crash_in_the_models_router_runs_is_a_crash():
    crashed = subprocess.CompletedProcess(["aie-opt"], -signal.SIGSEGV, "", "")
    with pytest.raises(rp.RouterCrash):
        rp.model_run(crashed, "a crashed run")
    for fn in (rp.priority_trees, rp.witness_problem):
        assert "model_run" in code_names(fn.__code__), fn.__name__


def test_a_strict_witness_timeout_is_no_strict_pass():
    d = rp.canonical(rp.load_design(SHARED))
    strict = rp.aie_opt(d.emit(), strict=True)
    failed = subprocess.CompletedProcess(strict.args, 1, "", "no routing")
    assert router_mutation.judge_strict(d, "route", True, failed)[0] == "strict-miss"
    outcome, detail = router_mutation.judge_strict(
        d, "route", True, failed, deadline=time.monotonic()
    )
    assert outcome == "timeout", detail


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


def test_a_last_candidate_judged_past_the_budget_leaves_shrinking_unfinished():
    d = rp.load_design(OVER_CUT)
    last = len(list(router_shrink.candidates(d)))
    calls = [0]

    def holds(e):
        calls[0] += 1
        if calls[0] == last:
            time.sleep(1.5)
        return False

    small, finished = router_shrink.shrink(d, holds, budget_s=1)
    assert calls[0] == last
    assert not finished
    assert small is d


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


def test_a_case_not_on_npu2_is_not_retargeted(router_hw, hw_case, tmp_path, capsys):
    args = argparse.Namespace(
        case=hw_case,
        variant=[],
        device="npu1",
        runs=20,
        process_timeout_s=120,
        work=tmp_path / "work",
        verdicts=None,
        peano="unused",
    )
    assert router_hw.run_case(args) != 0
    assert "cannot retarget to npu1" in capsys.readouterr().err
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
