# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# RUN: %pytest %s

"""llama_chain_hw's parsers and gates on output captured from the real runs."""

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CHECKS = ROOT / "utils/component_checks"
# Captured on Strix (npu2) as the scripts printed them, minus aiecc's
# runtime-offset warnings; tracebacks have the checkout's path as <repo>.
FIXTURES = CHECKS / "fixtures"


def output(name):
    return (FIXTURES / f"llama_{name}.txt").read_text()


@pytest.fixture(scope="module")
def llama():
    spec = importlib.util.spec_from_file_location(
        "llama_chain_hw", CHECKS / "llama_chain_hw.py"
    )
    module = importlib.util.module_from_spec(spec)
    # The check imports its siblings, as when run as a script.
    sys.path.insert(0, str(CHECKS))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(CHECKS))
    return module


def test_a_decode_whose_tokens_all_match_passes(llama):
    assert llama.parse_correctness(output("decode_pass"), 0) == {
        "passed": True,
        "tokens_passed": 4,
        "tokens_total": 4,
    }


def test_a_decode_whose_tokens_differ_fails(llama):
    got = llama.parse_correctness(output("decode_sampled_fail"), 1)
    assert not got["passed"]
    assert (got["tokens_passed"], got["tokens_total"]) == (3, 8)


@pytest.mark.parametrize(
    "name, returncode", [("timed_crash", 1), ("timed_nofit", 2), ("decode_pass", 1)]
)
def test_a_decode_that_did_not_exit_cleanly_fails(llama, name, returncode):
    got = llama.parse_correctness(output(name), returncode)
    assert not got["passed"]


def test_a_decode_that_checked_no_token_fails(llama):
    got = llama.parse_correctness(output("timed_crash"), 0)
    assert got["tokens_total"] == 0
    assert not got["passed"]


def test_a_decode_that_stopped_before_its_summary_fails(llama):
    out = output("decode_pass").split("\nchain_persist_grow_mh:")[0]
    got = llama.parse_correctness(out, 0)
    assert (got["tokens_passed"], got["tokens_total"]) == (4, 4)
    assert not got["passed"]


def test_the_timed_decode_publishes_its_statistic(llama):
    timing = llama.parse_timing(output("timed_pass"), 0)
    assert list(timing) == [llama.DISPATCH_RATE_AT]
    t = timing[llama.DISPATCH_RATE_AT]
    assert t["min_ms"] == pytest.approx(
        t["ms_per_token"] * t["tokens_per_dispatch"], rel=1e-3
    )
    assert t["tok_s"] == pytest.approx(1000 / t["ms_per_token"], rel=1e-3)
    assert t["n_iters"] == llama.N_ITERS
    assert llama.timing_sane(timing)
    rows = llama.aggregate(
        llama.parse_correctness(output("decode_pass"), 0),
        timing,
        llama.parse_generate("", 1),
        {},
        {},
    )
    names = {r["name"].rsplit("/", 1)[1]: r for r in rows}
    assert names["ms_per_token"]["value"] == t["ms_per_token"]
    assert "dma_floor_ratio" not in names


@pytest.mark.parametrize(
    "name, returncode", [("timed_crash", 1), ("timed_nofit", 2), ("timed_pass", 1)]
)
def test_a_timed_decode_that_failed_publishes_no_timing(llama, name, returncode):
    timing = llama.parse_timing(output(name), returncode)
    assert timing == {}
    assert llama.timing_sane(timing)


def test_generate_passes_when_the_oracle_passes_every_token(llama):
    got = llama.parse_generate(output("generate_pass"), 0)
    assert got["generate_passed"]
    assert (got["oracle_passed"], got["oracle_total"]) == (32, 32)
    assert got["ttft_ms"] > 0 and got["decode_tok_s"] > 0


def test_generate_fails_on_its_exit_status(llama):
    assert not llama.parse_generate(output("generate_pass"), 1)["generate_passed"]


def test_generate_without_its_oracle_line_fails(llama):
    got = llama.parse_generate(output("timed_crash"), 0)
    assert not got["generate_passed"]
    assert got["oracle_total"] == 0


def test_generate_rows_publish_only_after_its_oracle_passed(llama):
    correctness = llama.parse_correctness(output("decode_pass"), 0)
    timing = llama.parse_timing(output("timed_pass"), 0)
    passed = llama.aggregate(
        correctness, timing, llama.parse_generate(output("generate_pass"), 0), {}, {}
    )
    failed = llama.aggregate(
        correctness, timing, llama.parse_generate(output("generate_pass"), 1), {}, {}
    )
    names = {r["name"].rsplit("/", 1)[1]: r for r in passed}
    assert names["ttft_ms"]["value"] == 124.6
    assert names["ttft_min_ms"]["value"] == 124.3
    assert names["ttft_npu_dispatch_ms"]["value"] == 119.4
    assert names["ttft_host_ms"]["value"] == 5.2
    assert names["decode_tok_s"]["value"] == 22.14
    assert names["ttft_ms"]["range"] == "median of 5"
    assert names["ttft_min_ms"]["range"] == "fastest of 5"
    assert names["greedy_agree"]["value"] == 28
    assert names["greedy_agree"]["range"].startswith("of 32")
    assert "prompt_tok_s" not in names
    generate_rows = {"ttft_ms", "ttft_min_ms", "decode_tok_s", "greedy_agree"}
    assert not generate_rows & {r["name"].rsplit("/", 1)[1] for r in failed}
    assert "n_cores" not in names


def test_the_context_sweep_publishes_every_context(llama):
    out = output("decode_contexts_pass")
    correctness = llama.parse_correctness(out, 0)
    assert correctness["passed"]
    assert (correctness["tokens_passed"], correctness["tokens_total"]) == (36, 36)
    timing = llama.parse_timing(out, 0)
    assert set(timing) == {llama.DISPATCH_RATE_AT, *llama.CONTEXTS}
    assert llama.timing_sane(timing)
    rows = llama.aggregate(correctness, timing, llama.parse_generate("", 1), {}, {})
    case = f"llama/persist_grow/N={llama.N_LAYERS}"
    names = {r["name"]: r["value"] for r in rows}
    assert names[f"{case}/ms_per_token"] == 47.23
    assert names[f"{case}/tok_s"] == 21.17
    for at in llama.CONTEXTS:
        assert names[f"{case}/ctx={at}/ms_per_token"] == timing[at]["ms_per_token"]
        assert names[f"{case}/ctx={at}/decode_tok_s"] == timing[at]["tok_s"]
    assert names[f"{case}/ctx=131068/decode_tok_s"] == 2.09


def test_this_checkout_has_the_data_the_runs_read_or_names_what_it_lacks(llama):
    missing = llama.missing_data()
    data = ROOT / "programming_examples/ml/llama_3_2_1b/data"
    assert all(not (data / f).exists() for f in missing)
    assert all((data / f).exists() for f in set(llama.DATA_FILES) - set(missing))
