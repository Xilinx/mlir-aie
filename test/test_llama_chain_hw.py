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


def test_the_timed_decode_publishes_its_statistic(llama):
    timing = llama.parse_timing(output("timed_pass"), 0)
    assert timing["ms_per_token"] > 0
    assert timing["tok_s"] == pytest.approx(1000 / timing["ms_per_token"], rel=1e-3)
    assert timing["n_iters"] == llama.N_ITERS
    assert llama.timing_sane(timing)


@pytest.mark.parametrize(
    "name, returncode", [("timed_crash", 1), ("timed_nofit", 2), ("timed_pass", 1)]
)
def test_a_timed_decode_that_failed_publishes_no_timing(llama, name, returncode):
    timing = llama.parse_timing(output(name), returncode)
    assert set(timing.values()) == {None}
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
    names = {r["name"].rsplit("/", 1)[1] for r in passed}
    assert {"ttft_ms", "decode_tok_s", "greedy_agree"} <= names
    assert not {"ttft_ms", "decode_tok_s", "greedy_agree"} & {
        r["name"].rsplit("/", 1)[1] for r in failed
    }
    assert "n_cores" not in names


def test_this_checkout_has_the_data_the_runs_read_or_names_what_it_lacks(llama):
    missing = llama.missing_data()
    data = ROOT / "programming_examples/ml/llama_3_2_1b/data"
    assert all(not (data / f).exists() for f in missing)
    assert all((data / f).exists() for f in set(llama.DATA_FILES) - set(missing))
