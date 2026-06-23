"""Test the fused lm_head GEMM + top-k insert probe at the real slot layout.

Builds real lm_head weight slots [prefix | rows | bias | scales] + an int8
activation with a per-token scale tail; computes the numpy oracle (GEMM logits
= sum_i32 * act_scale * w_scale, then topk_sample_reference + embed requant);
dispatches and checks token + embed seed bit-exact.
"""

from __future__ import annotations

import sys

import numpy as np

import aie.iron as iron
import aie.utils.test as test_utils
from aie.utils import DefaultNPURuntime

from aie2_lmhead_topk_probe import (
    D,
    N_TILE,
    PREFIX,
    V,
    KSET,
    N_TILES,
    SLOT,
    TEMP,
    TOPK,
    SEED,
)
from numpy_topk_sample import topk_sample_reference


def embed_seed_ref(row_i8: np.ndarray, embed_sc: float):
    amax = int(np.max(np.abs(row_i8.astype(np.int32))))
    if amax < 1:
        amax = 1
    inv = (np.float32(127.0) / np.float32(amax)).astype(np.float32)
    xin = np.empty(D, np.int8)
    for i in range(D):
        v = np.float32(np.float32(row_i8[i]) * inv)
        r = np.floor(v + 0.5) if v >= 0 else np.ceil(v - 0.5)
        xin[i] = np.int8(np.clip(r, -128, 127))
    scale = np.float32(
        np.float32(amax) * np.float32(embed_sc) * np.float32(1.0 / 127.0)
    )
    return xin, scale


def main():
    opts = test_utils.create_default_argparser().parse_args()
    rng = np.random.default_rng(SEED + 1)

    # int8 activation + per-token scale tail.
    act_i8 = rng.integers(-127, 128, size=D, dtype=np.int8)
    act_scale = np.float32(rng.random() * 0.01 + 0.005)

    # Real lm_head weight slots: rows int8[V,D], i32 bias, f32 per-row scale.
    rows = rng.integers(-127, 128, size=(V, D), dtype=np.int8)
    bias = rng.integers(-1000, 1000, size=V, dtype=np.int32)
    w_sc = (rng.random(V).astype(np.float32) * 0.001 + 0.0005).astype(np.float32)

    # Pack act buffer [int8 D | f32 scale | f32 pad].
    act_buf = np.zeros(D + 8, np.int8)
    act_buf[:D] = act_i8
    act_buf[D : D + 4] = np.frombuffer(act_scale.tobytes(), np.int8)

    # Pack the table.
    tbl = np.zeros(N_TILES * SLOT, np.int8)
    for t in range(N_TILES):
        b = t * SLOT
        lo = t * N_TILE
        # rows at +PREFIX
        tbl[b + PREFIX : b + PREFIX + N_TILE * D] = rows[lo : lo + N_TILE].reshape(-1)
        # bias after rows
        bo = b + PREFIX + N_TILE * D
        tbl[bo : bo + N_TILE * 4] = np.frombuffer(
            bias[lo : lo + N_TILE].tobytes(), np.int8
        )
        # scales after bias
        so = bo + N_TILE * 4
        tbl[so : so + N_TILE * 4] = np.frombuffer(
            w_sc[lo : lo + N_TILE].tobytes(), np.int8
        )

    # Oracle logits: sum_i32 = rows @ act + bias; logit = sum * act_scale * w_sc.
    sum_i32 = rows.astype(np.int32) @ act_i8.astype(np.int32) + bias
    logits = (sum_i32.astype(np.float32) * act_scale * w_sc).astype(np.float32)

    ref_token = topk_sample_reference(logits, TEMP, TOPK, SEED, k_set=KSET)
    ref_xin, ref_scale = embed_seed_ref(rows[ref_token], float(w_sc[ref_token]))

    act_t = iron.tensor(act_buf, dtype=np.int8)
    tbl_t = iron.tensor(tbl, dtype=np.int8)
    token_t = iron.zeros([1], dtype=np.int32)
    seed_t = iron.zeros([D + 8], dtype=np.int8)
    npu = test_utils.create_npu_kernel(opts).npu_kernel
    rc = DefaultNPURuntime.run_test(
        npu, [act_t, tbl_t, token_t, seed_t], {}, verify=False, verbosity=opts.verbosity
    )
    if rc != 0:
        print(f"dispatch returned {rc}", file=sys.stderr)
        return rc
    token_t.to("cpu")
    seed_t.to("cpu")
    dev_token = int(token_t.numpy()[0])
    dev_seed = seed_t.numpy()
    dev_xin = dev_seed[:D]
    dev_scale = np.frombuffer(dev_seed[D : D + 4].tobytes(), np.float32)[0]

    tok_ok = dev_token == ref_token
    xin_ok = np.array_equal(dev_xin, ref_xin)
    sc_ok = dev_scale == ref_scale

    mode = "greedy" if TEMP <= 0.0 else f"multinomial(temp={TEMP},top_k={TOPK})"
    print(f"mode: {mode}  V={V} D={D} KSET={KSET} seed={SEED}")
    print(f"token: dev={dev_token} ref={ref_token} {'OK' if tok_ok else 'DIFF'}")
    print(
        f"seed xin: {'OK' if xin_ok else 'DIFF'}"
        + ("" if xin_ok else f" (ndiff={int(np.sum(dev_xin != ref_xin))})")
    )
    print(
        f"seed scale: dev={dev_scale:.8g} ref={ref_scale:.8g} {'OK' if sc_ok else 'DIFF'}"
    )

    ok = tok_ok and xin_ok and sc_ok
    print(
        f"\nlmhead_topk_probe: {'PASS' if ok else 'FAIL'}  (fused GEMM+insert -> token + seed)"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
