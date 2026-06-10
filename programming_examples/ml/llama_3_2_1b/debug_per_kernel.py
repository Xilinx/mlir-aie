"""Debug: per-kernel cross-check at chain-layer-0 data.

Reproduces chain layer 0's intermediate computation in numpy, then for
each kernel call dispatches the corresponding standalone NPU xclbin
with the SAME inputs numpy used and compares. The first kernel that
diverges between NPU and numpy is the bug.

Requires the standalone xclbins to be built at the chain's sizes:
  - rmsnorm at D=64        (build/final_rmsnorm_int8_64.xclbin)
  - gemm at K=64,N=64       (build/final_gemm_srs_64x64.xclbin)
  - silu at HD=256          (build/final_silu_mul_256.xclbin)
  - rope at head_dim=64,
    n_heads=1               (build/final_rope_64x1.xclbin)
  - flowkv at head_dim=64,
    T=16                    (build/final_flowkv_64x16.xclbin)
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
from ml_dtypes import bfloat16

import aie.iron as iron
import aie.utils.test as test_utils
from aie.utils import DefaultNPURuntime

from aie2_chain_real import (
    D,
    QD,
    KVD,
    HD,
    HEAD_D,
    N_HEADS,
    T,
    RIGHT_SHIFT,
    ACT_SCALE,
    INV_ACT_SCALE,
    UP_SCALE,
    INV_OUT_SCALE_SI,
    Q_SCALE,
    K_SCALE,
    V_SCALE,
    INV_OUT_SCALE_AT,
)
from test_chain_real import gen_layer
from test_rmsnorm_int8 import numpy_rmsnorm_int8
from test_gemm_int8_srs_real import banker_srs
from test_rope_int8 import numpy_rope
from test_silu_mul_int8 import numpy_silu_mul
from test_flowkv import numpy_attention, EXP_QUANT_SCALE
from gen_silu_lut import silu_lut
from gen_exp_lut import exp_lut


def numpy_gemm(act, weights, bias, rs=RIGHT_SHIFT):
    acc = weights.astype(np.int32) @ act.astype(np.int32)
    return banker_srs(acc + bias, rs).clip(-128, 127).astype(np.int8)


def dispatch_npu(xclbin, insts, tensors):
    """Run a single NPU dispatch and return the result."""
    saved_argv = sys.argv
    sys.argv = ["x", "-x", xclbin, "-i", insts, "-k", "MLIR_AIE"]
    p = test_utils.create_default_argparser()
    opts = p.parse_args()
    sys.argv = saved_argv
    npu_opts = test_utils.create_npu_kernel(opts)
    rc = DefaultNPURuntime.run_test(
        npu_opts.npu_kernel, tensors, {}, verify=False, verbosity=0
    )
    if rc != 0:
        raise RuntimeError(f"NPU dispatch failed rc={rc}")


def cmp(name, npu, numpy, top=4):
    diff = npu.astype(np.int16) - numpy.astype(np.int16)
    n = int((diff != 0).sum())
    ma = int(np.abs(diff).max()) if n else 0
    status = "OK   " if n == 0 else "FAIL "
    print(f"  {status} {name}: diffs={n}/{npu.size}  max|diff|={ma}")
    if n > 0 and top:
        bad = np.argwhere(diff != 0).flatten()[:top]
        for i in bad:
            print(f"      i={i}: NPU={int(npu[i])}  numpy={int(numpy[i])}")
    return n == 0


def main():
    rng = np.random.default_rng(0)
    x_in = rng.integers(-32, 33, size=D, dtype=np.int8)
    layer = gen_layer(rng)  # uses LLAMA_CHAIN_ACTIVE_DATA env flag

    print(f"x_in[:8]={x_in[:8]}")
    print(
        f"wq mean={float(np.abs(layer['wq']).mean()):.2f}  "
        f"bq mean={float(np.abs(layer['bq']).mean()):.2f}"
    )

    # --- Step 1: rmsnorm1 ---
    h1_numpy = numpy_rmsnorm_int8(x_in, layer["gamma_in"], ACT_SCALE, INV_ACT_SCALE)
    print(f"\nStep 1: rmsnorm1 (x_in [-32..32], gamma~1.0)")
    print(f"  numpy h1[:8]={h1_numpy[:8]}")
    x_t = iron.tensor(x_in.copy(), dtype=np.int8)
    g_t = iron.tensor(layer["gamma_in"], dtype=bfloat16)
    o_t = iron.zeros([D], dtype=np.int8)
    dispatch_npu(
        "build/final_rmsnorm_int8_64.xclbin",
        "build/insts_rmsnorm_int8_64.bin",
        [x_t, g_t, o_t],
    )
    o_t.to("cpu")
    h1_npu = o_t.numpy()
    print(f"  NPU   h1[:8]={h1_npu[:8]}")
    rms_ok = cmp("rmsnorm1", h1_npu, h1_numpy)

    # Use NPU's h1 for the next step (so any rmsnorm error doesn't cascade)
    h1 = h1_npu if rms_ok else h1_numpy

    # --- Step 2: q_proj gemm ---
    qf_numpy = numpy_gemm(h1, layer["wq"], layer["bq"])
    print(f"\nStep 2: q_proj gemm (D->QD, K=N=64)")
    print(f"  numpy qf[:8]={qf_numpy[:8]}")
    w_packed = np.concatenate(
        [layer["wq"].flatten().view(np.int8), layer["bq"].view(np.int8).flatten()]
    )
    act_t = iron.tensor(h1.copy(), dtype=np.int8)
    w_t = iron.tensor(w_packed, dtype=np.int8)
    out_t = iron.zeros([QD], dtype=np.int8)
    dispatch_npu(
        "build/final_gemm_srs_64x64.xclbin",
        "build/insts_gemm_srs_64x64.bin",
        [act_t, w_t, out_t],
    )
    out_t.to("cpu")
    qf_npu = out_t.numpy()
    print(f"  NPU   qf[:8]={qf_npu[:8]}")
    cmp("q_proj gemm", qf_npu, qf_numpy)
    qf = qf_npu

    # --- Step 3: rope (need a rope xclbin at head_dim=64, n_heads=1) ---
    # Skipping rope if no matching xclbin -- chain uses n_heads=1.
    # The existing standalone rope is at head_dim=64, n_heads=8 (8x size).
    qr_numpy = numpy_rope(qf, layer["cos"], layer["sin"], N_HEADS, HEAD_D, ACT_SCALE)
    print(f"\nStep 3: rope (head_dim=64, n_heads=1)")
    print(f"  numpy qr[:8]={qr_numpy[:8]}")
    # Use NPU rope at head_dim=64,n_heads=8 -- need a single-head xclbin.
    # For now, compute on NPU by using the qf padded to 8 heads (8x copies)
    # and taking the first head's output.
    if os.path.exists("build/final_rope_64x1.xclbin"):
        cs_packed = np.concatenate([layer["cos"], layer["sin"]])
        x_t = iron.tensor(qf.copy(), dtype=np.int8)
        cs_t = iron.tensor(cs_packed, dtype=bfloat16)
        o_t = iron.zeros([QD], dtype=np.int8)
        dispatch_npu(
            "build/final_rope_64x1.xclbin",
            "build/insts_rope_64x1.bin",
            [x_t, cs_t, o_t],
        )
        o_t.to("cpu")
        qr_npu = o_t.numpy()
        print(f"  NPU   qr[:8]={qr_npu[:8]}")
        cmp("rope", qr_npu, qr_numpy)
        qr = qr_npu
    else:
        print("  (skip; no build/final_rope_64x1.xclbin)")
        qr = qr_numpy

    # --- Step 4: flowkv (head_dim=64, T=16) ---
    print(f"\nStep 4: flowkv (head_dim=64, T=16)")
    lut_exp = exp_lut(EXP_QUANT_SCALE)
    af_numpy = numpy_attention(
        qr,
        layer["kcache"],
        layer["vcache"],
        HEAD_D,
        T,
        Q_SCALE,
        K_SCALE,
        V_SCALE,
        INV_OUT_SCALE_AT,
        lut_exp,
    )
    print(f"  numpy af[:8]={af_numpy[:8]}")
    q_t = iron.tensor(qr.copy(), dtype=np.int8)
    k_t = iron.tensor(layer["kcache"], dtype=np.int8)
    v_t = iron.tensor(layer["vcache"], dtype=np.int8)
    o_t = iron.zeros([QD], dtype=np.int8)
    dispatch_npu(
        "build/final_flowkv_64x16.xclbin",
        "build/insts_flowkv_64x16.bin",
        [q_t, k_t, v_t, o_t],
    )
    o_t.to("cpu")
    af_npu = o_t.numpy()
    print(f"  NPU   af[:8]={af_npu[:8]}")
    cmp("flowkv", af_npu, af_numpy)


if __name__ == "__main__":
    main()
