"""Numpy oracle for llama_topk_sample.cc -- the one-stream sampler + embed
gather. Bit-exact mirror of the kernel's CLEAN top-k renormalization (softmax +
inverse-CDF over EXACTLY the k survivors), NOT the legacy masked-tail path of
numpy_sample(_streamed).py.

Why clean renorm (not masked-tail): the legacy samplers add exp(-6.4) mass for
every one of the V-k filtered tokens; at V=128256 that tail is 84%+ of sum_exp
and the inverse-CDF picks a filtered token most of the time -- a latent bug
(see project_topk_masked_tail_bug memory). Clean renorm is the correct top-k
semantics and the only one a single resident-set pass can express.

Reuses the xoshiro PRNG / exp LUT / quant_shifted / sw_recip from numpy_sample.
"""

from __future__ import annotations

import numpy as np

from numpy_sample import (
    EXP_QUANT_SCALE,
    lookup_exp,
    quant_shifted,
    xoshiro_seed,
    xoshiro_uniform,
)
from gen_exp_lut import exp_lut
from test_flowkv import sw_recip


def topk_set_reference(logits_f32: np.ndarray, k_set: int):
    """Mirror llama_topk_insert: min-eviction unsorted set of capacity k_set
    streamed in index order. For distinct fp32 logits this yields exactly the
    k_set largest; ties keep the incumbent (first-seen). Returns the set's
    global indices (unordered, as the kernel holds them)."""
    V = logits_f32.size
    set_idx: list[int] = []
    for g in range(V):
        lg = logits_f32[g]
        if len(set_idx) < k_set:
            set_idx.append(g)
            continue
        # current-min slot (first-seen on ties)
        mi = 0
        mv = logits_f32[set_idx[0]]
        for i in range(1, k_set):
            if logits_f32[set_idx[i]] < mv:
                mv = logits_f32[set_idx[i]]
                mi = i
        if lg > mv:
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
    """Return the sampled global token id. top_k is the sampling cutoff; k_set
    is the resident-set capacity (kernel's kKset). Requires top_k <= k_set and
    top_k > 0 (full-vocab top_k=0 is out of scope for the one-stream design)."""
    logits = logits_f32.astype(np.float32)
    V = logits.size

    set_idx = topk_set_reference(logits, k_set)

    # Greedy: global argmax with first-occurrence (smallest gidx) tie-break.
    if temperature <= 0.0:
        wv = max(logits[g] for g in set_idx)
        winners = [g for g in set_idx if logits[g] == wv]
        return int(min(winners))

    if lut is None:
        lut = exp_lut(EXP_QUANT_SCALE).astype(np.float32)
    inv_temp = sw_recip(np.float32(temperature))

    # The sampling survivors = the top_k of the resident set (the set may hold
    # k_set >= top_k; cut to top_k by logit, first-seen tie-break). The kernel
    # currently samples over the WHOLE resident set, so for bit-exactness the
    # caller passes k_set == top_k; we cut here to top_k to stay general.
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
