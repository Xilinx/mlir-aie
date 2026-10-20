#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Phase 2 int8 RoPE test (half-split rotation, Llama-3 layout)."""

from __future__ import annotations

import argparse
import sys

import aie.iron as iron
import aie_fp
import numpy as np
from aie.utils import DefaultNPURuntime
from aie2_rope_int8 import ACT_SCALE, build
from llama_npu import add_artifact_args, design_kernel
from ml_dtypes import bfloat16


def numpy_rope(x_i8, cos_bf, sin_bf, n_heads, head_dim, act_scale):
    inv_scale = aie_fp.sw_recip(act_scale)
    half = head_dim // 2
    x_f = x_i8.astype(np.float32).reshape(n_heads, head_dim) * act_scale
    c = cos_bf.astype(np.float32)
    s = sin_bf.astype(np.float32)
    out = np.empty_like(x_f)
    out[:, :half] = aie_fp.aie_add(
        x_f[:, :half] * c[:half], -(x_f[:, half:] * s[:half])
    )
    out[:, half:] = aie_fp.aie_add(x_f[:, half:] * c[half:], x_f[:, :half] * s[half:])
    return aie_fp.round_to_i8((out * inv_scale).flatten())


def numpy_rope_dyn(x_i8, cos_bf, sin_bf, n_heads, head_dim):
    """Simplified rope matching the `llama_rope_int8_dyn` kernel that
    drops the redundant `* act_scale` and `* inv_scale` ops. They cancel
    mathematically (rope is norm-preserving) and dropping them removes
    two scalar-fp32 multiplies whose ULP noise was flipping rounding-tie
    boundaries differently than numpy at chain depth >= 4.

    No act_scale arg (not needed for the math).
    """
    half = head_dim // 2
    x_f = x_i8.astype(np.float32).reshape(n_heads, head_dim)
    c = cos_bf.astype(np.float32)
    s = sin_bf.astype(np.float32)
    out = np.empty_like(x_f)
    out[:, :half] = aie_fp.aie_add(
        x_f[:, :half] * c[:half], -(x_f[:, half:] * s[:half])
    )
    out[:, half:] = aie_fp.aie_add(x_f[:, half:] * c[half:], x_f[:, :half] * s[half:])
    return aie_fp.round_to_i8(out.flatten())


def main():
    p = argparse.ArgumentParser()
    add_artifact_args(p)
    p.add_argument("--head-dim", type=int, default=64)
    p.add_argument("--n-heads", type=int, default=8)
    p.add_argument("--dyn", action="store_true")
    opts = p.parse_args()

    head_dim, n_heads, dyn = opts.head_dim, opts.n_heads, opts.dyn
    D = head_dim * n_heads
    half = head_dim // 2

    rng = np.random.default_rng(0)
    x = rng.integers(-128, 128, size=D, dtype=np.int8)
    # Realistic cos/sin: doubled-halves over a random angle per pair.
    ang = rng.uniform(0, 2 * np.pi, size=half).astype(np.float32)
    cos_half = np.cos(ang).astype(bfloat16)
    sin_half = np.sin(ang).astype(bfloat16)
    cos = np.concatenate([cos_half, cos_half])
    sin = np.concatenate([sin_half, sin_half])

    cs = np.concatenate([cos, sin])  # packed cos || sin

    if dyn:
        x_buf = np.zeros(D + 8, dtype=np.int8)
        x_buf[:D] = x
        x_buf[D : D + 4] = np.frombuffer(np.float32(ACT_SCALE).tobytes(), dtype=np.int8)
        x_buf[D + 4 : D + 8] = np.frombuffer(np.float32(0.0).tobytes(), dtype=np.int8)
        x_t = iron.tensor(x_buf, dtype=np.int8)
        o_t = iron.zeros([D + 8], dtype=np.int8)
    else:
        x_t = iron.tensor(x, dtype=np.int8)
        o_t = iron.zeros([D], dtype=np.int8)
    cs_t = iron.tensor(cs, dtype=bfloat16)

    npu = design_kernel(build, opts, head_dim=head_dim, n_heads=n_heads, DYN=dyn)
    rc = DefaultNPURuntime.run_test(
        npu,
        [x_t, cs_t, o_t],
        {},
        verify=False,
        verbosity=opts.verbosity,
    )
    if rc != 0:
        return rc

    o_t.to("cpu")
    out_full = o_t.numpy()
    actual = out_full[:D]
    if dyn:
        expected = numpy_rope_dyn(x, cos, sin, n_heads, head_dim)
        expected_tail = np.concatenate(
            [
                np.frombuffer(np.float32(ACT_SCALE).tobytes(), dtype=np.int8),
                np.frombuffer(np.float32(0.0).tobytes(), dtype=np.int8),
            ]
        )
        tail_mismatch = not np.array_equal(out_full[D : D + 8], expected_tail)
    else:
        expected = numpy_rope(x, cos, sin, n_heads, head_dim, ACT_SCALE)
        tail_mismatch = False

    diff = actual.astype(np.int16) - expected.astype(np.int16)
    n_diff = int((diff != 0).sum())
    max_abs = int(np.abs(diff).max()) if n_diff else 0
    cos_sim = float(
        np.dot(actual.astype(np.float32), expected.astype(np.float32))
        / (
            np.linalg.norm(actual.astype(np.float32))
            * np.linalg.norm(expected.astype(np.float32))
            + 1e-12
        )
    )

    print(
        f"rope_int8 NPU vs numpy: heads={n_heads} d={head_dim} dyn={dyn}  "
        f"mismatches={n_diff}/{D}  max|int8 diff|={max_abs}  cos={cos_sim:.6f}  "
        f"tail_mismatch={tail_mismatch}"
    )

    if n_diff == 0 and not tail_mismatch:
        print("BIT-EXACT PASS")
        return 0
    print("FAIL")
    for i in np.argwhere(diff != 0).flatten()[:8]:
        print(f"  i={i}: NPU={actual[i]}  expected={expected[i]}  x={x[i]}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
