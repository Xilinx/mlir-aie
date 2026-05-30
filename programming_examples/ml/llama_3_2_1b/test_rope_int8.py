"""Phase 2 int8 RoPE test (half-split rotation, Llama-3 layout)."""

from __future__ import annotations

import sys

import numpy as np
from ml_dtypes import bfloat16

import aie.iron as iron
import aie.utils.test as test_utils
from aie.utils import DefaultNPURuntime

from aie2_rope_int8 import ACT_SCALE


def round_to_i8(v):
    r = np.where(v >= 0, np.floor(v + 0.5), np.ceil(v - 0.5)).astype(np.int32)
    return np.clip(r, -128, 127).astype(np.int8)


def numpy_rope(x_i8, cos_bf, sin_bf, n_heads, head_dim, act_scale):
    inv_scale = 1.0 / act_scale
    half = head_dim // 2
    x_f = x_i8.astype(np.float32).reshape(n_heads, head_dim) * act_scale
    c = cos_bf.astype(np.float32)
    s = sin_bf.astype(np.float32)
    out = np.empty_like(x_f)
    out[:, :half]  = x_f[:, :half] * c[:half]  - x_f[:, half:] * s[:half]
    out[:, half:]  = x_f[:, half:] * c[half:]  + x_f[:, :half] * s[half:]
    return round_to_i8((out * inv_scale).flatten())


def main():
    p = test_utils.create_default_argparser()
    p.add_argument("--head-dim", type=int, default=64)
    p.add_argument("--n-heads", type=int, default=8)
    p.add_argument("--max-mismatches", type=int, default=64)
    opts = p.parse_args()

    head_dim, n_heads = opts.head_dim, opts.n_heads
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

    x_t  = iron.tensor(x,  dtype=np.int8)
    cs_t = iron.tensor(cs, dtype=bfloat16)
    o_t  = iron.zeros([D], dtype=np.int8)

    npu_opts = test_utils.create_npu_kernel(opts)
    rc = DefaultNPURuntime.run_test(
        npu_opts.npu_kernel,
        [x_t, cs_t, o_t],
        {},
        verify=False,
        verbosity=opts.verbosity,
    )
    if rc != 0:
        return rc

    o_t.to("cpu")
    actual = o_t.numpy()
    expected = numpy_rope(x, cos, sin, n_heads, head_dim, ACT_SCALE)

    diff = actual.astype(np.int16) - expected.astype(np.int16)
    n_diff = int((diff != 0).sum())
    max_abs = int(np.abs(diff).max()) if n_diff else 0
    cos_sim = float(np.dot(actual.astype(np.float32), expected.astype(np.float32)) /
                    (np.linalg.norm(actual.astype(np.float32)) *
                     np.linalg.norm(expected.astype(np.float32)) + 1e-12))

    print(
        f"rope_int8 NPU vs numpy: heads={n_heads} d={head_dim}  "
        f"mismatches={n_diff}/{D}  max|int8 diff|={max_abs}  cos={cos_sim:.6f}"
    )

    if n_diff <= opts.max_mismatches and max_abs <= 1:
        print(f"PASS (within 1-LSB bf16 noise: <={opts.max_mismatches} diffs)")
        return 0
    print("FAIL")
    for i in np.argwhere(diff != 0).flatten()[:8]:
        print(f"  i={i}: NPU={actual[i]}  expected={expected[i]}  x={x[i]}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
