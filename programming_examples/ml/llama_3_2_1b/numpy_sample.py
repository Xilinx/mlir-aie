"""Numpy reference for llama_sample.cc -- bit-exact, matches the kernel's
xoshiro128++ PRNG, splitmix64-style seed expansion, exp LUT, sw_recip,
and inverse-CDF draw.

NOT the same as cautious-eureka/npu2/sampling.py: that file is the
algorithmic spec and uses numpy default_rng + np.exp + np.partition.
Here we reproduce the kernel's bitwise arithmetic.
"""

from __future__ import annotations

import numpy as np

from gen_exp_lut import exp_lut
from test_flowkv import sw_recip   # IEEE fp32 reciprocal NR, matches kernel

EXP_QUANT_SCALE = 0.05
MASK_SENTINEL   = np.float32(-1.0e9)


# --- xoshiro128++ PRNG (uint32 arithmetic, must match the C kernel) ---

def _u32(x):
    return np.uint32(x & 0xFFFFFFFF)


def _u64(x):
    return np.uint64(x & 0xFFFFFFFFFFFFFFFF)


def _rotl32(x: np.uint32, k: int) -> np.uint32:
    x = _u32(x)
    return _u32((int(x) << k) | (int(x) >> (32 - k)))


def xoshiro_seed(seed: int) -> list:
    """splitmix64-style expansion of one uint32 seed into 4 uint32 state."""
    x = _u64(int(seed) * 0x9E3779B97F4A7C15 + 1)
    s = []
    for _ in range(4):
        x = _u64(int(x) ^ (int(x) >> 30)); x = _u64(int(x) * 0xBF58476D1CE4E5B9)
        x = _u64(int(x) ^ (int(x) >> 27)); x = _u64(int(x) * 0x94D049BB133111EB)
        x = _u64(int(x) ^ (int(x) >> 31))
        s.append(_u32(int(x) ^ (int(x) >> 32)))
    return s


def xoshiro_next(s: list) -> tuple:
    result = _u32(int(_rotl32(_u32(int(s[0]) + int(s[3])), 7)) + int(s[0]))
    t = _u32(int(s[1]) << 9)
    s[2] = _u32(int(s[2]) ^ int(s[0]))
    s[3] = _u32(int(s[3]) ^ int(s[1]))
    s[1] = _u32(int(s[1]) ^ int(s[2]))
    s[0] = _u32(int(s[0]) ^ int(s[3]))
    s[2] = _u32(int(s[2]) ^ int(t))
    s[3] = _rotl32(s[3], 11)
    return result, s


def xoshiro_uniform(s: list) -> tuple:
    r, s = xoshiro_next(s)
    u = np.float32(np.float32(int(r) >> 8) * np.float32(1.0 / 16777216.0))
    return u, s


# --- Kernel building blocks (mirror llama_sample.cc one-for-one) ---

def quant_shifted(shifted_f32: np.ndarray) -> np.ndarray:
    inv = np.float32(1.0) / np.float32(EXP_QUANT_SCALE)
    v = (shifted_f32.astype(np.float32) * inv).astype(np.float32)
    q = np.where(v >= 0, np.floor(v + np.float32(0.5)),
                         np.ceil(v - np.float32(0.5))).astype(np.int32)
    return np.clip(q, -128, 0)


def lookup_exp(q: np.ndarray, lut: np.ndarray) -> np.ndarray:
    return lut[(q + 128).astype(np.int32)].astype(np.float32)


def sample_reference(logits_i8: np.ndarray,
                     temperature: float, top_k: int, seed: int,
                     lut: np.ndarray | None = None) -> int:
    V = logits_i8.size

    # Greedy short-circuit.
    if temperature <= 0.0:
        # First-occurrence argmax via scalar loop (matches kernel; np.argmax
        # tie-breaks the same way on a contiguous array, but be explicit).
        best_v = 0
        best   = int(logits_i8[0])
        for v in range(1, V):
            l = int(logits_i8[v])
            if l > best:
                best = l; best_v = v
        return best_v

    if lut is None:
        lut = exp_lut(EXP_QUANT_SCALE).astype(np.float32)

    inv_temp = sw_recip(np.float32(temperature))

    # Pass 1: z[i] = logits[i] * inv_temp, fp32. Stored as raw fp32 (we
    # don't need the bit-cast game the kernel plays -- numpy preserves
    # fp32 values across array writes natively).
    z = (logits_i8.astype(np.float32) * inv_temp).astype(np.float32)

    # Top-k masking via "find max k times" loop (NOT np.partition: the
    # kernel uses this exact algorithm, and tie-breaks via first-seen.)
    if top_k > 0 and top_k < V:
        masked = np.zeros(V, dtype=np.int8)
        threshold = np.float32(0.0)
        for _ in range(top_k):
            best_v = -1
            best_z = np.float32(0.0)
            for v in range(V):
                if masked[v]:
                    continue
                if best_v < 0 or z[v] > best_z:
                    best_v = v; best_z = z[v]
            if best_v < 0:
                break
            threshold = best_z
            masked[best_v] = 1
        # Strict-less-than mask.
        z = np.where(z < threshold, MASK_SENTINEL, z).astype(np.float32)

    max_z = np.float32(z.max())

    # Pass 2: quantize shifted, accumulate sum_exp via sequential adds.
    # Match kernel's order (loop v=0..V-1) so fp32 rounding agrees.
    qvals = quant_shifted((z - max_z).astype(np.float32))
    sum_exp = np.float32(0.0)
    for v in range(V):
        sum_exp = np.float32(sum_exp + lookup_exp(np.array([qvals[v]]), lut)[0])

    # Inverse-CDF draw.
    s = xoshiro_seed(int(seed) & 0xFFFFFFFF)
    u_unit, s = xoshiro_uniform(s)
    u = np.float32(u_unit * sum_exp)

    c = np.float32(0.0)
    pick = V - 1
    for v in range(V):
        c = np.float32(c + lookup_exp(np.array([qvals[v]]), lut)[0])
        if c > u:
            pick = v
            break
    return int(pick)
