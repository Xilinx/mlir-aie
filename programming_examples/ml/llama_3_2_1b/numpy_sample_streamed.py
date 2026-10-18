"""Numpy oracle for llama_sample_streamed.cc -- bit-exact mirror of the
streamed sampler kernel on fp32 logits (real vocab).

Differs from numpy_sample.py in two ways that match the streamed kernel:
  - operates on fp32 logits (not int8), as produced by the real lm_head.
  - top-k uses a VALUE threshold = k-th largest z (the kernel keeps an
    unsorted top-k value set and thresholds by its min), not the reference's
    index-mask. For distinct fp32 logits these are identical.

Reuses the xoshiro PRNG, exp LUT, quant_shifted, sw_recip from numpy_sample.
"""

from __future__ import annotations

import numpy as np

from numpy_sample import (
    EXP_QUANT_SCALE,
    MASK_SENTINEL,
    lookup_exp,
    quant_shifted,
    xoshiro_seed,
    xoshiro_uniform,
)
from gen_exp_lut import exp_lut
from test_flowkv import sw_recip


def sample_streamed_reference(
    logits_f32: np.ndarray,
    temperature: float,
    top_k: int,
    seed: int,
    lut: np.ndarray | None = None,
) -> int:
    V = logits_f32.size
    logits = logits_f32.astype(np.float32)

    # Greedy short-circuit: first-occurrence argmax over raw logits.
    if temperature <= 0.0:
        best_v = 0
        best = logits[0]
        for v in range(1, V):
            if logits[v] > best:
                best = logits[v]
                best_v = v
        return best_v

    if lut is None:
        lut = exp_lut(EXP_QUANT_SCALE).astype(np.float32)

    inv_temp = sw_recip(np.float32(temperature))
    z = (logits * inv_temp).astype(np.float32)

    use_topk = 0 < top_k < V
    if use_topk:
        # k-th largest via value threshold (= top-k set min). np.partition
        # gives the k largest; their min is the threshold. Identical to the
        # kernel's unsorted-set min for distinct values.
        kth = np.sort(z)[-top_k]
        threshold = np.float32(kth)
    else:
        threshold = np.float32(-3.0e38)

    max_z = np.float32(z.max())

    # Pass 2: sum_exp in index order (masked entries -> sentinel -> exp(-128)).
    sum_exp = np.float32(0.0)
    for v in range(V):
        zv = z[v]
        if use_topk and zv < threshold:
            zv = MASK_SENTINEL
        q = quant_shifted(np.array([zv - max_z], dtype=np.float32))[0]
        sum_exp = np.float32(sum_exp + lookup_exp(np.array([q]), lut)[0])

    # Pass 3: inverse-CDF draw.
    s = xoshiro_seed(int(seed) & 0xFFFFFFFF)
    u_unit, s = xoshiro_uniform(s)
    u = np.float32(u_unit * sum_exp)

    c = np.float32(0.0)
    pick = V - 1
    for v in range(V):
        zv = z[v]
        if use_topk and zv < threshold:
            zv = MASK_SENTINEL
        q = quant_shifted(np.array([zv - max_z], dtype=np.float32))[0]
        c = np.float32(c + lookup_exp(np.array([q]), lut)[0])
        if c > u:
            pick = v
            break
    return int(pick)
