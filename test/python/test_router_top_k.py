# test_router_top_k.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# RUN: %pytest %s
"""router_top_k's reference (no NPU required), against computations that
share none of its code: a stable sort for the chosen experts and a running
sum of the kernel's own terms for the weights.
"""

from pathlib import Path

import numpy as np
import pytest
from aie.iron.device import from_name
from aie.iron.kernels import moe
from aie.iron.kernels._common import ARCH_TRAITS
from aie.utils import config
from aie.utils.compile.utils import compile_cxx_core_function
from aie.utils.hostruntime import set_current_device
from aie.iron.kernels.sample import exp64_ref, summary_words
from ml_dtypes import bfloat16

SUMMARY = summary_words(32, 8)


def _row(rng, experts):
    return rng.normal(0, 3, experts).astype(bfloat16)


def _chosen(row, top_k):
    """The top_k of ``row``, ties to the lower index, in index order within
    the strictly larger ones, then the tied ones, as the kernel writes them.
    """
    order = np.argsort(-row.astype(np.float32), kind="stable")[:top_k]
    tau = row.astype(np.float32)[order[-1]]
    above = np.sort(order[row.astype(np.float32)[order] > tau])
    ties = np.sort(order[row.astype(np.float32)[order] == tau])
    return np.concatenate([above, ties])


@pytest.mark.parametrize("experts, top_k", [(32, 8), (64, 16), (60, 4), (128, 8)])
def test_the_weights_are_the_kernel_s_terms_summed_in_order(experts, top_k):
    rng = np.random.default_rng(experts + top_k)
    for _ in range(200):
        row = _row(rng, experts)
        ids, weights = moe.router_top_k_ref(row, top_k)
        np.testing.assert_array_equal(ids, _chosen(row, top_k))
        values = row[ids].astype(np.float32)
        terms = exp64_ref((values - values.max()).astype(np.float64))
        total = 0.0
        for t in terms:
            total += t
        want = (terms / total).astype(np.float32).astype(bfloat16)
        np.testing.assert_array_equal(
            weights[0::2].view(np.uint16), want.view(np.uint16)
        )
        assert not weights[1::2].astype(np.float32).any()


@pytest.mark.parametrize("experts, top_k", [(60, 4), (32, 8)])
def test_unnormalized_weights_are_shares_of_the_whole_row_s_softmax(experts, top_k):
    rng = np.random.default_rng(experts * top_k)
    for _ in range(200):
        row = _row(rng, experts)
        row[rng.choice(experts, 3, replace=False)] = -np.inf
        ids, weights = moe.router_top_k_ref(row, top_k, normalize=False)
        np.testing.assert_array_equal(ids, _chosen(row, top_k))
        every = row.astype(np.float32)
        terms = exp64_ref((every - every.max()).astype(np.float64))
        total = 0.0
        for t in terms:
            total += t
        want = (terms[ids] / total).astype(np.float32).astype(bfloat16)
        np.testing.assert_array_equal(
            weights[0::2].view(np.uint16), want.view(np.uint16)
        )


@pytest.mark.parametrize("normalize", [True, False])
def test_packed_weights_are_the_padded_ones_without_their_pads(normalize):
    rng = np.random.default_rng(7)
    for _ in range(50):
        row = _row(rng, 64)
        ids, padded = moe.router_top_k_ref(row, 6, normalize=normalize)
        same, packed = moe.router_top_k_ref(row, 6, normalize=normalize, packed=True)
        np.testing.assert_array_equal(ids, same)
        assert packed.shape == (6,)
        np.testing.assert_array_equal(
            packed.view(np.uint16), padded[0::2].view(np.uint16)
        )


@pytest.mark.parametrize(
    "kwargs, flags, shapes",
    [
        ({}, [], [(SUMMARY,), (8,), (16,)]),
        (dict(packed=True), ["-DROUTER_PACKED"], [(SUMMARY,), (8,), (8,)]),
        (
            dict(normalize=False),
            ["-DROUTER_UNNORMALIZED"],
            [(SUMMARY,), (32,), (8,), (16,)],
        ),
    ],
)
def test_factory_options_set_the_flags_and_arguments(kwargs, flags, shapes):
    fn = moe.router_top_k(slice_size=32, top_k=8, **kwargs)
    options = [f for f in fn._compile_flags if f.startswith("-DROUTER")]
    assert options == flags
    assert [fn.arg_shape(i) for i in range(len(shapes))] == shapes
    assert len(fn.contract.roles) == len(shapes)


def test_the_default_router_s_flags_are_unchanged():
    fn = moe.router_top_k(slice_size=32, top_k=8)
    assert [f for f in fn._compile_flags if f.startswith(("-D", "-f"))] == [
        "-ffp-contract=off",
        "-fno-fast-math",
        "-DSAMPLE_SLICE=32",
        "-DSAMPLE_K_MAX=8",
    ]


def test_packed_weights_need_an_even_top_k():
    with pytest.raises(ValueError, match="even top_k"):
        moe.router_top_k(slice_size=32, top_k=5, packed=True)


def _peano_available() -> bool:
    try:
        return Path(config.peano_cxx_path()).is_file()
    except RuntimeError:
        return False


@pytest.mark.skipif(not _peano_available(), reason="needs an installed Peano")
@pytest.mark.parametrize("arch", list(ARCH_TRAITS))
@pytest.mark.parametrize(
    "kwargs",
    [{}, dict(packed=True), dict(normalize=False), dict(normalize=False, packed=True)],
)
def test_every_option_builds_for_every_architecture(arch, kwargs, tmp_path):
    with set_current_device(from_name(ARCH_TRAITS[arch].device, n_cols=1)):
        fn = moe.router_top_k(slice_size=60, top_k=4, **kwargs)
        source = Path(fn._source_file)
        compile_cxx_core_function(
            str(source),
            arch,
            str(tmp_path / f"{fn.name}.o"),
            include_dirs=[*fn.include_dirs, str(source.parent)],
            compile_args=list(fn._compile_flags),
        )
    assert (tmp_path / f"{fn.name}.o").stat().st_size > 0
