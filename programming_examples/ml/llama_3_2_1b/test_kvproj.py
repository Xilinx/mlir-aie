"""Stage 1b-i test: KV-projection front half bit-exact vs numpy.

Validates h1 = rmsnorm_dyn(x), then k_fp = wk@h1 * act_scale1 * wk_sc and
v_fp = wv@h1 * act_scale1 * wv_sc (the numpy oracle's k_proj/v_proj, before
rope) match the device drain.
"""

from __future__ import annotations

import sys

import numpy as np
from ml_dtypes import bfloat16

import aie.iron as iron
import aie.utils.test as test_utils
from aie.utils import DefaultNPURuntime

from aie2_kvproj import (
    D,
    KV_DIM,
    N_TILE,
    WK_SLOT,
    WV_SLOT,
    N_TILES_K,
    OFF_GAMMA,
    OFF_WK,
    OFF_WV,
    GAMMA_BYTES,
    WEIGHTS_BYTES,
)
from numpy_layer_mh import gen_layer_mh, requant
from test_rmsnorm_int8_dyn import numpy_rmsnorm_int8_dyn
from test_ffn_half import pack_perchan_slots


def run_one(seed: int, opts, npu_kernel) -> int:
    rng = np.random.default_rng(seed)
    layer = gen_layer_mh(rng)
    gamma_in = layer["gamma_in"]
    wk_i8, wk_sc = layer["wk_i8"], layer["wk_sc"]
    wv_i8, wv_sc = layer["wv_i8"], layer["wv_sc"]
    bk = np.zeros(KV_DIM, np.int32)
    bv = np.zeros(KV_DIM, np.int32)

    # Per-token residual seed (residual_dyn), as the real layer gets it.
    x_fp = rng.uniform(-1.6, 1.6, size=D).astype(np.float32)
    res_scale = np.float32(np.maximum(np.abs(x_fp).max(), 1e-12) / 127.0)
    x_i8 = requant(x_fp, np.float32(1.0) / res_scale)

    # numpy oracle: h1 then k_fp/v_fp.
    h1, act_scale1 = numpy_rmsnorm_int8_dyn(x_i8, gamma_in, float(res_scale))
    k_fp = (
        (wk_i8.astype(np.int32) @ h1.astype(np.int32)).astype(np.float32)
        * np.float32(act_scale1)
        * wk_sc.astype(np.float32)
    )
    v_fp = (
        (wv_i8.astype(np.int32) @ h1.astype(np.int32)).astype(np.float32)
        * np.float32(act_scale1)
        * wv_sc.astype(np.float32)
    )

    # Pack wblob.
    wblob = np.zeros(WEIGHTS_BYTES, dtype=np.int8)
    wblob[OFF_GAMMA : OFF_GAMMA + GAMMA_BYTES] = gamma_in.view(np.int8)
    wk_packed = pack_perchan_slots(wk_i8, wk_sc, bk, N_TILE, prefix_bytes=b"\x00" * 64)
    wv_packed = pack_perchan_slots(wv_i8, wv_sc, bv, N_TILE, prefix_bytes=b"\x00" * 64)
    assert wk_packed.size == N_TILES_K * WK_SLOT, (wk_packed.size, N_TILES_K * WK_SLOT)
    wblob[OFF_WK : OFF_WK + wk_packed.size] = wk_packed
    wblob[OFF_WV : OFF_WV + wv_packed.size] = wv_packed

    xin = np.zeros(D + 8, dtype=np.int8)
    xin[:D] = x_i8
    xin[D : D + 4] = np.frombuffer(np.float32(res_scale).tobytes(), dtype=np.int8)

    x_t = iron.tensor(xin, dtype=np.int8)
    w_t = iron.tensor(wblob, dtype=np.int8)
    kfp_t = iron.zeros([KV_DIM], dtype=np.float32)
    vfp_t = iron.zeros([KV_DIM], dtype=np.float32)
    rc = DefaultNPURuntime.run_test(
        npu_kernel, [x_t, w_t, kfp_t, vfp_t], {}, verify=False, verbosity=opts.verbosity
    )
    if rc != 0:
        print(f"seed {seed}: dispatch returned {rc}", file=sys.stderr)
        return rc
    kfp_t.to("cpu")
    vfp_t.to("cpu")
    k_dev = kfp_t.numpy()
    v_dev = vfp_t.numpy()

    # fp32 gemm: allow tiny ULP; report exact + max abs error.
    k_exact = np.array_equal(k_dev, k_fp.astype(np.float32))
    v_exact = np.array_equal(v_dev, v_fp.astype(np.float32))
    k_err = float(np.abs(k_dev - k_fp).max())
    v_err = float(np.abs(v_dev - v_fp).max())
    k_rel = k_err / max(float(np.abs(k_fp).max()), 1e-12)
    v_rel = v_err / max(float(np.abs(v_fp).max()), 1e-12)
    ok = (k_exact and v_exact) or (k_rel < 1e-5 and v_rel < 1e-5)
    tag = "BIT-EXACT" if (k_exact and v_exact) else ("PASS(ulp)" if ok else "FAIL")
    print(
        f"seed {seed}: {tag}  k max|d|={k_err:.3g} (rel {k_rel:.2e})  "
        f"v max|d|={v_err:.3g} (rel {v_rel:.2e})"
    )
    return 0 if ok else 1


def main():
    p = test_utils.create_default_argparser()
    p.add_argument("--seeds", type=str, default="0,1,7,42")
    opts = p.parse_args()
    seeds = [int(s) for s in opts.seeds.split(",")]
    npu_kernel = test_utils.create_npu_kernel(opts).npu_kernel
    fails = sum(run_one(s, opts, npu_kernel) != 0 for s in seeds)
    print(f"\nkvproj: {len(seeds) - fails}/{len(seeds)} seeds PASS")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
