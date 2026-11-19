# test_attention_merge_e2e.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# RUN: %run_on_npu1_xrt% %pytest %s
# RUN: %run_on_npu2_xrt% %pytest %s
# REQUIRES: xrt_python_bindings

"""Split-KV attention's merge, several parts on one core.

``attention_merge`` folds one part a call into sums the core keeps, so its
contract (one part a call) never rescales them. Here a core folds every part
of a split in turn, FIRST to LAST, and the merge is held to its bound
against the exact merge of the same parts, in float64. The parts come from
real attention split over keys, whose merge is attention over all the keys,
and from draws that stress the rescale: maxima 80 apart, empty parts, and
rows no part attended.
"""

import aie.iron as iron
import numpy as np
import pytest
from aie.iron import CompileTime, In, ObjectFifo, Out, Program, Runtime, Worker, kernels
from aie.iron.algorithms import kernel_design as kd
from aie.iron.buffer import Buffer
from aie.utils.verify import Tolerance, compare
from ml_dtypes import bfloat16


@iron.jit
def _merge_design(
    o_in: In,
    stats_in: In,
    out: Out,
    *,
    parts: CompileTime[int],
    rows: CompileTime[int],
    dim: CompileTime[int],
):
    kernel = kernels.attention_merge(rows, dim)
    _, o_ty, stats_ty, acc_ty, out_ty = kernel.arg_types()
    of_o = ObjectFifo(o_ty, name="o")
    of_stats = ObjectFifo(stats_ty, name="stats")
    of_out = ObjectFifo(out_ty, name="out")
    # NaN until the FIRST call overwrites it: a kernel that read the sums it
    # was meant to start would return NaN.
    acc = Buffer(
        acc_ty,
        name="acc",
        initial_value=np.full(rows * dim + 2 * rows, np.nan, np.float32),
    )

    def core(of_o, of_stats, of_out, acc, kernel):
        merged = of_out.acquire(1)
        for j in range(parts):
            flags = (kernels.ATTENTION_MERGE_FIRST if j == 0 else 0) | (
                kernels.ATTENTION_MERGE_LAST if j == parts - 1 else 0
            )
            o, stats = of_o.acquire(1), of_stats.acquire(1)
            kernel(flags, o, stats, acc, merged)
            of_o.release(1)
            of_stats.release(1)
        of_out.release(1)

    worker = Worker(core, [of_o.cons(), of_stats.cons(), of_out.prod(), acc, kernel])
    o_host = np.ndarray[(parts * rows * dim,), np.dtype[np.float32]]
    stats_host = np.ndarray[(parts * 2 * rows,), np.dtype[np.float32]]

    def seq(o, stats, merged, h_o, h_stats, h_out):
        h_o.fill(o)
        h_stats.fill(stats)
        h_out.drain(merged, wait=True)

    rt = Runtime(
        seq,
        [o_host, stats_host, out_ty, of_o.prod(), of_stats.prod(), of_out.cons()],
    )
    return Program(iron.get_current_device(), rt, workers=[worker]).resolve_program()


def _merge(o, stats, parts, rows, dim):
    o_t = iron.tensor(o.ravel(), dtype=np.float32, device="npu")
    stats_t = iron.tensor(stats.ravel(), dtype=np.float32, device="npu")
    out = iron.tensor(
        np.full(rows * dim, np.nan, bfloat16), dtype=bfloat16, device="npu"
    )
    _merge_design(o_t, stats_t, out, parts=parts, rows=rows, dim=dim)
    return out.numpy().copy()


def _attention_parts(rng, parts, rows, dim, keys):
    """``rows`` queries over ``keys`` keys split into ``parts``: each part's normalized attention and stats, and attention over all the keys, all in float64."""
    q = rng.standard_normal((rows, dim)) * 2
    k = rng.standard_normal((parts * keys, dim))
    v = rng.uniform(-4, 4, (parts * keys, dim))
    s = q @ k.T / np.sqrt(dim)
    full = np.exp(s - s.max(axis=1, keepdims=True))
    full = (full / full.sum(axis=1, keepdims=True)) @ v
    o, stats = [], []
    for j in range(parts):
        sj = s[:, j * keys : (j + 1) * keys]
        m = sj.max(axis=1)
        p = np.exp(sj - m[:, None])
        o.append(p / p.sum(axis=1, keepdims=True) @ v[j * keys : (j + 1) * keys])
        stats.append(np.stack([m, p.sum(axis=1)], axis=1))
    return np.stack(o), np.stack(stats), full


def _held(got, o, stats, parts, rows, dim):
    dims = dict(parts=parts, rows=rows, dim=dim)
    o, stats = o.reshape(1, -1), stats.reshape(1, -1)
    exact = kernels.attention_merge_ref(o, stats, **dims)
    bound = kernels.attention_merge_bound(o, stats, **dims)
    tol = Tolerance.bounded(kernels.attention_merge_bound, note="the merge's bound")
    return compare(got.reshape(1, -1), exact, tol, bound=bound)


# (parts, rows, dim): llama 3.2 1B's 32 packed heads over 8 parts; 2 parts;
# rows of 16 (the 16-lane weights); one part; a 128-wide head over 16.
_SHAPES = [(8, 32, 64), (2, 32, 64), (4, 16, 64), (1, 32, 64), (16, 16, 128)]


@pytest.mark.parametrize(
    "parts, rows, dim", _SHAPES, ids=[f"p{p}-r{r}-d{d}" for p, r, d in _SHAPES]
)
def test_merging_split_attention_is_attention_over_all_keys(parts, rows, dim):
    rng = np.random.default_rng(parts * rows + dim)
    o, stats, full = _attention_parts(rng, parts, rows, dim, keys=64)
    dims = dict(parts=parts, rows=rows, dim=dim)
    # The merge's formula, in float64, is attention over all the keys...
    np.testing.assert_allclose(
        kernels.attention_merge_ref(o.reshape(1, -1), stats.reshape(1, -1), **dims),
        full.reshape(1, -1),
        rtol=1e-12,
        atol=1e-12,
    )
    # ...and the device holds it to its bound on the float32 parts.
    o, stats = o.astype(np.float32), stats.astype(np.float32)
    got = _merge(o, stats, parts, rows, dim)
    verdict = _held(got, o, stats, parts, rows, dim)
    assert verdict, verdict.detail


@pytest.mark.parametrize(
    "parts, rows, dim", _SHAPES, ids=[f"p{p}-r{r}-d{d}" for p, r, d in _SHAPES]
)
@pytest.mark.parametrize("draw", ["sample", "far_maxima", "empty_rows"])
def test_the_merge_holds_its_bound_where_the_rescale_is_stressed(
    parts, rows, dim, draw
):
    rng = np.random.default_rng(7 * parts + rows + dim)
    o, stats = kernels.attention_merge_sample(rng, 1, parts=parts, rows=rows, dim=dim)
    o, stats = o.reshape(parts, rows, dim), stats.reshape(parts, rows, 2)
    if draw == "far_maxima":
        # Each part's maximum up to 80 below the row's: weights down to e^-80.
        stats[..., 0] = rng.uniform(-80, 0, (parts, rows))
        o = (o * 10.0 ** rng.uniform(-3, 3, o.shape)).astype(np.float32)
    if draw == "empty_rows":
        # Row 0 attended by no part (0 out); row 1 by the last part alone.
        stats[:, 0] = -np.inf, 0
        stats[:-1, 1] = -np.inf, 0
        stats[-1, 1, 1] = 3
    got = _merge(o, stats, parts, rows, dim)
    verdict = _held(got, o, stats, parts, rows, dim)
    assert verdict, verdict.detail
    if draw == "empty_rows":
        np.testing.assert_array_equal(got.reshape(rows, dim)[0].astype(np.float32), 0)


@pytest.mark.parametrize(
    "tile_size, options",
    [(2048, {}), (160, dict(addend=True)), (1024, dict(masked=True))],
    ids=["decode-row", "addend", "masked"],
)
def test_softmax_stats_hold_on_attention_rows(tile_size, options):
    """A float32 softmax's (m, l), the merge's weights, on rows whose every key counts.

    The contract's sample spreads logits wide, so one term dominates l;
    attention's scores sit within a few units, so l sums thousands of bf16
    exponentials, each rounded.
    """
    fn = kernels.softmax(tile_size, input_dtype=np.float32, stats=True, **options)
    harness = kd.KernelDesign(fn)
    rng = np.random.default_rng(tile_size)
    inputs = harness.sample_inputs(rng)
    inputs[0] = (rng.standard_normal(np.shape(inputs[0])) * 2).astype(np.float32)
    raw = harness.run([np.array(x, copy=True) for x in inputs])
    verdict = harness.check(raw, inputs)
    assert verdict, verdict
