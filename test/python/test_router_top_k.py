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

import numpy as np
import pytest
from aie.iron.kernels import moe
from aie.iron.kernels.sample import exp64_ref
from ml_dtypes import bfloat16


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
