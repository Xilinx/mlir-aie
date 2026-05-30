"""Phase 2 flowkv test: full-softmax attention pair, 1 head, M=1 decode.

Numpy reference replicates the kernel's arithmetic EXACTLY:
  - int32 dot products
  - scores = dot * (q_scale * k_scale / sqrt(head_dim))   (one combined fp32 multiply)
  - shifted = score - max
  - q = clamp(round(shifted / EXP_QUANT_SCALE), -128, 0)  (kernel's quant)
  - exp_v = exp_lut[q + 128]   (shared LUT from gen_exp_lut.py)
  - probs = exp_v * (1.0 / sum_exp)   (exact fp32 division, not aie::inv)
  - sv: probs @ V_dequant, requant via round-half-away-from-zero

Goal: 0/head_dim mismatches (true bit-exact).
"""

from __future__ import annotations

import sys

import numpy as np

import aie.iron as iron
import aie.utils.test as test_utils
from aie.utils import DefaultNPURuntime

from aie2_flowkv import Q_SCALE, K_SCALE, V_SCALE, INV_OUT_SCALE
from gen_exp_lut import exp_lut

# Must match the kernel #define LLAMA_FLOWKV_EXP_QUANT_SCALE.
EXP_QUANT_SCALE = 0.05


def round_to_i8(v):
    r = np.where(v >= 0, np.floor(v + 0.5), np.ceil(v - 0.5)).astype(np.int32)
    return np.clip(r, -128, 127).astype(np.int8)


def quant_shifted(shifted: np.ndarray) -> np.ndarray:
    """Replicates the kernel's quant_shifted exactly."""
    inv = 1.0 / EXP_QUANT_SCALE
    v = (shifted * inv).astype(np.float32)
    q = np.where(v >= 0, np.floor(v + 0.5), np.ceil(v - 0.5)).astype(np.int32)
    return np.clip(q, -128, 0)


def numpy_attention(q_i8, k_i8, v_i8, head_dim, t,
                    q_scale, k_scale, v_scale, inv_out_scale,
                    lut):
    qk_scale = np.float32(q_scale * k_scale * (1.0 / np.sqrt(head_dim)))

    # int32 dot products per key.
    k_mat = k_i8.astype(np.int32).reshape(t, head_dim)
    dots = (k_mat @ q_i8.astype(np.int32))             # (t,) int32
    scores = (dots.astype(np.float32) * qk_scale)      # one combined mul

    max_s = scores.max()
    shifted = scores - max_s
    q = quant_shifted(shifted)
    exp_v = lut[q + 128]

    sum_e = float(exp_v.sum())
    probs = (exp_v * (1.0 / sum_e)).astype(np.float32)

    # sv path.
    v_mat = v_i8.astype(np.int32).reshape(t, head_dim)
    acc = (probs[:, None].astype(np.float32) * v_mat.astype(np.float32)).sum(axis=0)
    out_f = acc * v_scale * inv_out_scale
    return round_to_i8(out_f)


def main():
    p = test_utils.create_default_argparser()
    p.add_argument("--head-dim", type=int, default=64)
    p.add_argument("-T", type=int, default=16)
    opts = p.parse_args()

    head_dim, t = opts.head_dim, opts.T
    rng = np.random.default_rng(0)
    q = rng.integers(-128, 128, size=head_dim,       dtype=np.int8)
    k = rng.integers(-128, 128, size=t * head_dim,   dtype=np.int8)
    v = rng.integers(-128, 128, size=t * head_dim,   dtype=np.int8)

    lut = exp_lut(EXP_QUANT_SCALE)

    q_t = iron.tensor(q, dtype=np.int8)
    k_t = iron.tensor(k, dtype=np.int8)
    v_t = iron.tensor(v, dtype=np.int8)
    o_t = iron.zeros([head_dim], dtype=np.int8)

    npu_opts = test_utils.create_npu_kernel(opts)
    rc = DefaultNPURuntime.run_test(
        npu_opts.npu_kernel,
        [q_t, k_t, v_t, o_t],
        {},
        verify=False,
        verbosity=opts.verbosity,
    )
    if rc != 0:
        return rc

    o_t.to("cpu")
    actual = o_t.numpy()
    expected = numpy_attention(q, k, v, head_dim, t,
                               Q_SCALE, K_SCALE, V_SCALE, INV_OUT_SCALE, lut)

    diff = actual.astype(np.int16) - expected.astype(np.int16)
    n_diff = int((diff != 0).sum())
    max_abs = int(np.abs(diff).max()) if n_diff else 0
    print(f"flowkv NPU vs numpy: head_dim={head_dim} T={t}  "
          f"mismatches={n_diff}/{head_dim}  max|int8 diff|={max_abs}")

    if n_diff == 0:
        print("BIT-EXACT PASS")
        return 0
    print("FAIL")
    for i in np.argwhere(diff != 0).flatten()[:8]:
        print(f"  i={i}: NPU={actual[i]}  expected={expected[i]}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
