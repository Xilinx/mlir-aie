# test_benchmark_stats.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# RUN: %pytest %s
"""Unit tests for aie.utils.benchmark robust statistics (no NPU required)."""

import pytest
from aie.utils.benchmark import BenchmarkResult, Stats, print_benchmark, run_iters

# ---------------------------------------------------------------------------
# Stats.from_samples
# ---------------------------------------------------------------------------


def test_single_sample_has_zero_spread():
    s = Stats.from_samples([7.0])
    assert s.avg_us == s.min_us == s.max_us == s.median_us == s.p95_us == 7.0
    assert s.mad_us == 0.0
    # stdev is undefined for n == 1; CoV must degrade to 0 rather than raise.
    assert s.cov == 0.0
    assert s.n == 1


def test_median_is_robust_to_an_outlier():
    # The mean is dragged far above every sample but one; the median is not.
    s = Stats.from_samples([1.0, 2.0, 3.0, 4.0, 100.0])
    assert s.median_us == 3.0
    assert s.avg_us == 22.0
    # MAD is the median of |x - median| = median([2, 1, 0, 1, 97]) = 1
    assert s.mad_us == 1.0


def test_min_max_and_n_survive_unsorted_input():
    s = Stats.from_samples([5.0, 1.0, 3.0])
    assert (s.min_us, s.max_us, s.n) == (1.0, 5.0, 3)
    assert s.median_us == 3.0


def test_p95_picks_the_top_sample_of_twenty():
    # ceil(0.95 * 20) - 1 == 18 → the 19th of 20 sorted samples.
    s = Stats.from_samples([float(i) for i in range(20)])
    assert s.p95_us == 18.0


def test_p95_never_indexes_past_the_end():
    for n in range(1, 40):
        s = Stats.from_samples([float(i) for i in range(n)])
        assert s.min_us <= s.p95_us <= s.max_us


def test_cov_is_zero_for_constant_samples():
    assert Stats.from_samples([4.0, 4.0, 4.0]).cov == 0.0


def test_cov_is_positive_when_samples_vary():
    assert Stats.from_samples([1.0, 2.0, 3.0]).cov > 0.0


def test_empty_samples_rejected():
    with pytest.raises(ValueError):
        Stats.from_samples([])


def test_as_dict_is_flat_and_prefixable():
    d = Stats.from_samples([1.0, 2.0]).as_dict(prefix="npu_")
    assert d["npu_n"] == 2
    assert set(d) == {
        "npu_median_us",
        "npu_mad_us",
        "npu_p95_us",
        "npu_min_us",
        "npu_max_us",
        "npu_avg_us",
        "npu_cov",
        "npu_n",
    }
    # Raw samples stay out of the emitted record; they can be huge.
    assert not any("sample" in k for k in d)


# ---------------------------------------------------------------------------
# run_iters
# ---------------------------------------------------------------------------


def test_warmup_iterations_are_excluded_from_samples():
    calls = []
    run_iters(lambda: calls.append(1), warmup=3, iters=5)
    assert len(calls) == 8  # warmup + iters actually ran


def test_sample_count_matches_iters():
    r = run_iters(lambda: None, warmup=2, iters=4)
    assert r.e2e.n == 4
    assert len(r.e2e.samples_us) == 4


def test_npu_stats_are_none_without_npu_time():
    assert run_iters(lambda: None, iters=2).npu is None


def test_npu_time_is_extracted_from_a_kernel_result_tuple():
    class KernelResult:
        npu_time = 1500  # ns

    r = run_iters(lambda: ("handle", KernelResult()), iters=3)
    assert r.npu is not None
    assert r.npu.median_us == 1.5  # 1500 ns → 1.5 us
    assert r.npu.n == 3


def test_arg_sets_rotate_across_iterations():
    seen = []
    run_iters(lambda x: seen.append(x), arg_sets=[(1,), (2,), (3,)], iters=6)
    assert seen == [1, 2, 3, 1, 2, 3]


def test_arg_sets_rotation_includes_warmup():
    seen = []
    run_iters(lambda x: seen.append(x), arg_sets=[(1,), (2,)], warmup=1, iters=2)
    # Rotation is driven by the loop index, so warmup consumes the first entry.
    assert seen == [1, 2, 1]


def test_arg_sets_and_positional_args_are_mutually_exclusive():
    with pytest.raises(ValueError):
        run_iters(lambda x: None, 1, arg_sets=[(1,)], iters=1)


def test_kwargs_are_forwarded_with_arg_sets():
    seen = []
    run_iters(
        lambda x, scale: seen.append(x * scale),
        arg_sets=[(2,), (3,)],
        iters=2,
        scale=10,
    )
    assert seen == [20, 30]


@pytest.mark.parametrize("iters,warmup", [(0, 0), (-1, 0), (1, -1)])
def test_invalid_iteration_counts_rejected(iters, warmup):
    with pytest.raises(ValueError):
        run_iters(lambda: None, iters=iters, warmup=warmup)


# ---------------------------------------------------------------------------
# print_benchmark
# ---------------------------------------------------------------------------


def test_print_benchmark_keeps_the_canonical_aligned_columns(capsys):
    # programming_examples/getting_started/00_memcpy and basic/inline_kernel
    # hand-roll the same two lines; the column alignment must match theirs.
    e2e = Stats.from_samples([10.0, 20.0])
    print_benchmark(BenchmarkResult(e2e=e2e, npu=Stats.from_samples([1.0, 2.0])))
    out = capsys.readouterr().out
    assert "NPU time     (avg/min/max us): " in out
    assert "End-to-end   (avg/min/max us): " in out
    # Robust stats are appended, not substituted for the canonical trio.
    assert "median" in out and "MAD" in out and "p95" in out


def test_print_benchmark_omits_the_npu_line_without_npu_time(capsys):
    print_benchmark(BenchmarkResult(e2e=Stats.from_samples([1.0]), npu=None))
    out = capsys.readouterr().out
    assert "NPU time" not in out
    assert "End-to-end" in out


def test_empty_arg_sets_is_rejected_not_divided_by():
    with pytest.raises(ValueError, match="at least one"):
        run_iters(lambda: None, arg_sets=[], iters=1)
