"""M1 test: standalone tiled lm_head + greedy argmax, bit-exact vs numpy.

Builds a final hidden state (int8[D] + fp32 scale), runs the device path
(final_norm rmsnorm_dyn -> tiled lm_head GEMM over V=128256 -> running argmax),
and asserts the device token == the device-faithful numpy argmax oracle.

The oracle mirrors the device arithmetic exactly: numpy_rmsnorm_int8_dyn for
final_norm (two-pass dynamic int8 requant), then int32-accumulate lm_head GEMM
dequantized as acc * norm_scale * embed_sc[n] (bias = 0), argmax with
first-occurrence tie-break.

Run:
  make lmhead_sample
  python test_lmhead_sample.py -x build/final_lmhead_sample.xclbin \\
      -i build/insts_lmhead_sample.bin -k MLIR_AIE
"""

from __future__ import annotations

import os
import struct
import sys
from pathlib import Path

import numpy as np

import aie.iron as iron
import aie.utils.test as test_utils
from aie.utils import DefaultNPURuntime

from aie2_lmhead_sample import D, VOCAB, N_TILE, WLM_SLOT, WLM_TOTAL, N_TILES_LM
from numpy_llama_ref import _load_qw, _load_bf16, VOCAB_SIZE, EMB_DIM
from numpy_layer_mh import requant
from test_rmsnorm_int8_dyn import numpy_rmsnorm_int8_dyn, sw_recip
from test_ffn_half import pack_perchan_slots

DATA_DIR = Path(__file__).parent / "data"


def oracle_token(hidden_i8, hidden_scale, gamma, embed_i8, embed_sc):
    """Device-faithful greedy argmax over the lm_head logits."""
    normed_i8, norm_scale = numpy_rmsnorm_int8_dyn(hidden_i8, gamma, hidden_scale)
    # int32 accumulate (exact via int64), dequant per row.
    acc = embed_i8.astype(np.int64) @ normed_i8.astype(np.int64)  # (V,)
    logits = (
        acc.astype(np.float32) * np.float32(norm_scale) * embed_sc.astype(np.float32)
    )
    return int(np.argmax(logits)), normed_i8, norm_scale


def pack_lmw(embed_i8, embed_sc):
    """Pack lm_head weights into per-tile slots:
    [64 B pad | N_TILE*D i8 | N_TILE i32 bias=0 | N_TILE fp32 wscale].
    Reuses pack_perchan_slots with a 64 B zero prefix (alignment pad; the
    GEMM reads act_scale from the activation tail, not this prefix)."""
    bias = np.zeros(VOCAB, np.int32)
    blob = pack_perchan_slots(
        embed_i8, embed_sc.astype(np.float32), bias, N_TILE, prefix_bytes=b"\x00" * 64
    )
    assert blob.size == WLM_TOTAL, (blob.size, WLM_TOTAL)
    return blob


def run_one(seed, gamma, embed_i8, embed_sc, lmw, npu_kernel, opts):
    rng = np.random.default_rng(seed)
    hidden_fp = rng.uniform(-1.6, 1.6, size=D).astype(np.float32)
    h_scale = np.float32(max(float(np.abs(hidden_fp).max()), 1e-12) / 127.0)
    hidden_i8 = requant(hidden_fp, np.float32(1.0) / h_scale)

    ref_tok, _, _ = oracle_token(hidden_i8, h_scale, gamma, embed_i8, embed_sc)

    xin = np.zeros(D + 8, dtype=np.int8)
    xin[:D] = hidden_i8
    xin[D : D + 4] = np.frombuffer(np.float32(h_scale).tobytes(), dtype=np.int8)

    x_t = iron.tensor(xin, dtype=np.int8)
    g_t = iron.tensor(gamma, dtype=gamma.dtype)
    w_t = iron.tensor(lmw, dtype=np.int8)
    o_t = iron.zeros([8], dtype=np.int8)

    rc = DefaultNPURuntime.run_test(
        npu_kernel, [x_t, g_t, w_t, o_t], {}, verify=False, verbosity=opts.verbosity
    )
    if rc != 0:
        print(f"seed {seed}: dispatch returned {rc}", file=sys.stderr)
        return 1
    o_t.to("cpu")
    out = o_t.numpy()
    dev_tok = struct.unpack("<i", out[:4].tobytes())[0]
    dev_val = struct.unpack("<f", out[4:8].tobytes())[0]

    ok = dev_tok == ref_tok
    print(
        f"seed {seed}: {'PASS' if ok else 'FAIL'}  dev_tok={dev_tok} "
        f"ref_tok={ref_tok}  dev_max_logit={dev_val:.6g}"
    )
    return 0 if ok else 1


def main():
    p = test_utils.create_default_argparser()
    p.add_argument("--seeds", type=str, default="0,1,7,42")
    opts = p.parse_args()
    seeds = [int(s) for s in opts.seeds.split(",")]

    assert VOCAB == VOCAB_SIZE and D == EMB_DIM, (VOCAB, VOCAB_SIZE, D, EMB_DIM)
    print(f"loading embed/final_norm from {DATA_DIR} ...", flush=True)
    embed_i8, embed_sc = _load_qw(DATA_DIR, "embed", (VOCAB, D))  # (V,D) i8, (V,) f32
    gamma = _load_bf16(DATA_DIR, "final_norm", (D,))  # bf16[D]
    print("packing lm_head weights (262 MB) ...", flush=True)
    lmw = pack_lmw(embed_i8, embed_sc)

    npu_kernel = test_utils.create_npu_kernel(opts).npu_kernel
    print(f"lmhead_sample  V={VOCAB}  N_TILES={N_TILES_LM}  seeds={seeds}", flush=True)
    fails = 0
    for s in seeds:
        fails += run_one(s, gamma, embed_i8, embed_sc, lmw, npu_kernel, opts) != 0
    print(f"\nlmhead_sample: {len(seeds) - fails}/{len(seeds)} seeds PASS")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
