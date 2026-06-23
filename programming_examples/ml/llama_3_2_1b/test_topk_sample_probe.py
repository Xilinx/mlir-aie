"""Test the one-stream top-k sampler + embed gather probe.

Builds a packed V-row table [logits f32 | embed_sc f32 | embed rows i8], runs
the device, and checks BOTH outputs vs the numpy oracle:
  - token id == topk_sample_reference (greedy or multinomial-top-k, clean renorm)
  - seed (next-token embedding) == requant of the winner's embed row.

KSET == TOPK so the kernel's "sample over the whole resident set" equals
sampling over the top_k survivors.
"""

from __future__ import annotations

import sys

import numpy as np

import aie.iron as iron
import aie.utils.test as test_utils
from aie.utils import DefaultNPURuntime

from aie2_topk_sample_probe import V, D, KSET, N_TILE, N_TILES, TEMP, TOPK, SEED
from numpy_topk_sample import topk_sample_reference


def embed_seed_ref(row_i8: np.ndarray, embed_sc: float):
    """Mirror llama_topk_finalize's requant (== llama_embed_select)."""
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

    # Distinct fp32 logits (top-k set is well-defined). embed_sc positive.
    logits = rng.standard_normal(V).astype(np.float32)
    embed_sc = (rng.random(V).astype(np.float32) * 0.02 + 0.001).astype(np.float32)
    rows = rng.integers(-127, 128, size=(V, D), dtype=np.int8)

    # Pack the table: per tile [N_TILE logits | N_TILE scales | N_TILE*D rows].
    SLOT = N_TILE * 4 + N_TILE * 4 + N_TILE * D
    tbl = np.zeros(N_TILES * SLOT, np.int8)
    for t in range(N_TILES):
        b = t * SLOT
        lo = t * N_TILE
        tbl[b : b + N_TILE * 4] = np.frombuffer(
            logits[lo : lo + N_TILE].tobytes(), np.int8
        )
        tbl[b + N_TILE * 4 : b + N_TILE * 8] = np.frombuffer(
            embed_sc[lo : lo + N_TILE].tobytes(), np.int8
        )
        tbl[b + N_TILE * 8 : b + SLOT] = rows[lo : lo + N_TILE].reshape(-1)

    # Oracle.
    ref_token = topk_sample_reference(logits, TEMP, TOPK, SEED, k_set=KSET)
    ref_xin, ref_scale = embed_seed_ref(rows[ref_token], float(embed_sc[ref_token]))

    tbl_t = iron.tensor(tbl, dtype=np.int8)
    token_t = iron.zeros([1], dtype=np.int32)
    seed_t = iron.zeros([D + 8], dtype=np.int8)
    npu = test_utils.create_npu_kernel(opts).npu_kernel
    rc = DefaultNPURuntime.run_test(
        npu, [tbl_t, token_t, seed_t], {}, verify=False, verbosity=opts.verbosity
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
        + (
            ""
            if xin_ok
            else f" (first dev={dev_xin[:6]} ref={ref_xin[:6]}; "
            f"ndiff={int(np.sum(dev_xin != ref_xin))})"
        )
    )
    print(
        f"seed scale: dev={dev_scale:.8g} ref={ref_scale:.8g} {'OK' if sc_ok else 'DIFF'}"
    )

    ok = tok_ok and xin_ok and sc_ok
    print(
        f"\ntopk_sample_probe: {'PASS' if ok else 'FAIL'}  (one stream -> token + seed)"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
