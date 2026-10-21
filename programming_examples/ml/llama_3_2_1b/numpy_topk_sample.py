#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Numpy oracle for llama_topk_sample.cc -- the one-stream sampler + embed
gather. Bit-exact mirror of the kernel's CLEAN top-k renormalization (softmax +
inverse-CDF over EXACTLY the k survivors), NOT the legacy masked-tail path of
numpy_sample(_streamed).py.

Why clean renorm (not masked-tail): the legacy samplers add exp(-6.4) mass for
every one of the V-k filtered tokens; at V=128256 that tail is 84%+ of sum_exp
and the inverse-CDF picks a filtered token most of the time -- a latent bug
(see project_topk_masked_tail_bug memory). Clean renorm is the correct top-k
semantics and the only one a single resident-set pass can express.

Reuses the xoshiro PRNG / exp LUT / quant_shifted from numpy_sample.
"""

from __future__ import annotations

import numpy as np
from gen_exp_lut import exp_lut
from llama_config import EXP_QUANT_SCALE, SAMPLE_EXP_LOW
from numpy_sample import (
    lookup_exp,
    quant_shifted,
    xoshiro_seed,
    xoshiro_uniform,
)


def topk_set_reference(logits_f32: np.ndarray, k_set: int):
    """Mirror llama_lmhead_topk_insert's set: an unsorted set of capacity
    k_set streamed in index order, ordered by logit and then by lower index on
    equal logits. The set is the k_set largest under that order, ties
    included, whatever the streaming order. Returns the set's global indices
    in slot order, as the kernel holds them."""
    V = logits_f32.size
    set_idx: list[int] = []
    for g in range(V):
        lg = logits_f32[g]
        if len(set_idx) < k_set:
            set_idx.append(g)
            continue
        mi = 0
        for i in range(1, k_set):
            a, b = logits_f32[set_idx[i]], logits_f32[set_idx[mi]]
            if a < b or (a == b and set_idx[i] > set_idx[mi]):
                mi = i
        mv = logits_f32[set_idx[mi]]
        if lg > mv or (lg == mv and g < set_idx[mi]):
            set_idx[mi] = g
    return set_idx


def topk_sample_reference(
    logits_f32: np.ndarray,
    temperature: float,
    top_k: int,
    seed: int,
    k_set: int = 64,
    lut: np.ndarray | None = None,
) -> int:
    """Return the sampled global token id. k_set is the resident-set capacity
    (the kernel's kKset); sampling draws from the set's best top_k members (by
    logit, then lower id), and top_k 0 means the whole set. Greedy
    (temperature <= 0) is the argmax whatever top_k is.

    Raises:
        ValueError: top_k is outside 0..k_set.
    """
    if not 0 <= top_k <= k_set:
        raise ValueError(f"top_k must be 0 (the whole set) or 1..{k_set}, got {top_k}")
    top_k = top_k or k_set
    logits = logits_f32.astype(np.float32)
    set_idx = topk_set_reference(logits, k_set)

    # Greedy: global argmax with first-occurrence (smallest gidx) tie-break.
    if temperature <= 0.0:
        wv = max(logits[g] for g in set_idx)
        winners = [g for g in set_idx if logits[g] == wv]
        return int(min(winners))

    if lut is None:
        lut = exp_lut(EXP_QUANT_SCALE, SAMPLE_EXP_LOW).astype(np.float32)
    inv_temp = np.float32(1) / np.float32(temperature)

    # The survivors: the set's best top_k by logit, then lower id.
    by_logit = sorted(set_idx, key=lambda g: (-logits[g], g))
    survivors = sorted(by_logit[:top_k])  # ascending global-index order

    zs = [np.float32(logits[g] * inv_temp) for g in survivors]
    max_z = np.float32(max(zs))

    sum_exp = np.float32(0.0)
    for z in zs:
        q = quant_shifted(np.array([z - max_z], dtype=np.float32))[0]
        sum_exp = np.float32(sum_exp + lookup_exp(np.array([q]), lut)[0])

    s = xoshiro_seed(int(seed) & 0xFFFFFFFF)
    u_unit, s = xoshiro_uniform(s)
    u = np.float32(u_unit * sum_exp)

    c = np.float32(0.0)
    pick = survivors[-1]  # fallback if u falls off the end (fp roundoff)
    for gi, z in zip(survivors, zs):
        q = quant_shifted(np.array([z - max_z], dtype=np.float32))[0]
        c = np.float32(c + lookup_exp(np.array([q]), lut)[0])
        if c > u:
            pick = gi
            break
    return int(pick)
