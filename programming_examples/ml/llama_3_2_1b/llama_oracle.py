#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""The decode's oracle: a token through the device's layers and lm_head, in numpy."""

import numpy as np
from aie2_decode import KV_TB
from numpy_layer_mh import numpy_layer_mh_forward, numpy_rmsnorm_int8_dyn

# A greedy token other than the oracle's still passes as a near-tie, judged
# against the logit spread: the oracle does not yet mirror the vector fp32
# multiplies of rmsnorm, silu and up_requant, which leave rare 1-ulp
# differences that grow with depth.
TOL_GAP_STD = 0.15
TOL_RANK = 5


def chain_forward_logits(x_i8, res_scale, layers, position, embed_i8, embed_sc, gamma):
    """One token's forward at `position`, attending [0, position].

    Appends the token's KV row to the layers' caches and sets each layer's
    `scales`.

    Returns:
        (fp32 logits over the vocab, the final residual: int8[D], its scale).
    """
    x_cur = (x_i8.copy(), float(res_scale))
    for L in range(len(layers)):
        x_cur, scales = numpy_layer_mh_forward(
            x_cur,
            layers[L],
            position=position,
            residual_dyn=True,
            attn_perslot=True,
            attn_flowkv=True,
            attn_tb=KV_TB,
            attn_cores=2,
            attn_interleaved=True,
        )
        layers[L]["scales"] = scales
    xo_i8, xo_scale = x_cur
    normed_i8, norm_scale = numpy_rmsnorm_int8_dyn(xo_i8, gamma, np.float32(xo_scale))
    acc = embed_i8.astype(np.int64) @ normed_i8.astype(np.int64)
    logits = (
        acc.astype(np.float32) * np.float32(norm_scale) * embed_sc.astype(np.float32)
    ).astype(np.float32)
    return logits, xo_i8, np.float32(xo_scale)


def greedy_verdict(logits, token):
    """Judge the device's greedy `token` against the oracle's `logits`.

    Args:
        logits: The oracle's fp32 logits for the position.
        token: The token the device picked.

    Returns:
        (whether it passes, the oracle's argmax, a tag saying how).
    """
    ref = int(np.argmax(logits))
    if token == ref:
        return True, ref, "exact"
    gap = float(logits.max() - logits[token]) / float(logits.std())
    rank = int(np.where(np.argsort(logits)[::-1] == token)[0][0])
    return (
        gap <= TOL_GAP_STD and rank < TOL_RANK,
        ref,
        f"near-tie gap/std={gap:.3f} rank={rank}",
    )
