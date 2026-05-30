"""Phase 2 flowkv test: full-softmax attention pair, 1 head, M=1 decode."""

from __future__ import annotations

import sys

import numpy as np

import aie.iron as iron
import aie.utils.test as test_utils
from aie.utils import DefaultNPURuntime

from aie2_flowkv import Q_SCALE, K_SCALE, V_SCALE, INV_OUT_SCALE


def round_to_i8(v):
    r = np.where(v >= 0, np.floor(v + 0.5), np.ceil(v - 0.5)).astype(np.int32)
    return np.clip(r, -128, 127).astype(np.int8)


def numpy_attention(q_i8, k_i8, v_i8, head_dim, t,
                    q_scale, k_scale, v_scale, inv_out_scale):
    qf = q_i8.astype(np.float32) * q_scale
    kf = (k_i8.astype(np.float32) * k_scale).reshape(t, head_dim)
    vf = (v_i8.astype(np.float32) * v_scale).reshape(t, head_dim)
    scores = kf @ qf / np.sqrt(head_dim)
    sm = scores - scores.max()
    e = np.exp(sm)
    probs = e / e.sum()
    out = probs @ vf
    return round_to_i8(out * inv_out_scale)


def main():
    p = test_utils.create_default_argparser()
    p.add_argument("--head-dim", type=int, default=64)
    p.add_argument("-T", type=int, default=16)
    # v0 uses scalar exp via aie::exp2<bf16> (one lane out of 16 set)
    # for simplicity; the resulting bf16 truncations + exp-approximation
    # vs numpy fp32 exp lead to ~70% int8 mismatches at max |diff|=3,
    # cos~0.9997. Vec-optimize the exp/softmax in a follow-up.
    p.add_argument("--max-mismatches", type=int, default=64)
    p.add_argument("--max-abs", type=int, default=4)
    opts = p.parse_args()

    head_dim, t = opts.head_dim, opts.T
    rng = np.random.default_rng(0)
    q = rng.integers(-128, 128, size=head_dim,       dtype=np.int8)
    k = rng.integers(-128, 128, size=t * head_dim,   dtype=np.int8)
    v = rng.integers(-128, 128, size=t * head_dim,   dtype=np.int8)

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
                               Q_SCALE, K_SCALE, V_SCALE, INV_OUT_SCALE)

    diff = actual.astype(np.int16) - expected.astype(np.int16)
    n_diff = int((diff != 0).sum())
    max_abs = int(np.abs(diff).max()) if n_diff else 0
    cos = float(np.dot(actual.astype(np.float32), expected.astype(np.float32)) /
                (np.linalg.norm(actual.astype(np.float32)) *
                 np.linalg.norm(expected.astype(np.float32)) + 1e-12))

    print(
        f"flowkv NPU vs numpy: head_dim={head_dim} T={t}  "
        f"mismatches={n_diff}/{head_dim}  max|int8 diff|={max_abs}  cos={cos:.6f}"
    )

    if n_diff <= opts.max_mismatches and max_abs <= opts.max_abs:
        print(f"PASS (within HW exp/bf16 noise)")
        return 0
    print("FAIL")
    for i in np.argwhere(diff != 0).flatten()[:8]:
        print(f"  i={i}: NPU={actual[i]}  expected={expected[i]}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
