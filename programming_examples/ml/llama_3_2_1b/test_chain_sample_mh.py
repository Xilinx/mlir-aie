"""M3b test: fused N-layer chain + on-chip lm_head + sampler -> int32 token.

Reuses test_chain_mh's random-fixture chain (pack_blobs + numpy forward) to get
the device's final hidden state, then runs the REAL embed/final_norm lm_head +
sampler on it. The device (LLAMA_CHAIN_SAMPLE=1 xclbin) does the whole thing in
one dispatch and returns a token; we compare to the numpy oracle.

The chain layers use random fixtures (no vocab), but the lm_head uses the real
embedding weights -- so the token is arbitrary-but-deterministic and still a
valid bit-exact check of the fused final_norm + lm_head + sampler path.

Run:
  make chain_sample_mh CHAIN_MH_N=2
  python test_chain_sample_mh.py -x build/final_chain_sample_N2_T128.xclbin \\
      -i build/insts_chain_sample_N2_T128.bin -k MLIR_AIE
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path

import numpy as np

import aie.iron as iron
import aie.utils.test as test_utils
from aie.utils import DefaultNPURuntime

from aie2_chain_dynscale_mh import (
    D,
    T,
    N_LAYERS,
    N_HEADS_KV,
    HEAD_D,
    VOCAB,
    N_TILE,
    WLM_SLOT,
    WLM_TOTAL,
    SAMPLE_TEMP,
    SAMPLE_TOPK,
    SAMPLE_SEED,
)
from numpy_layer_mh import gen_layer_mh, numpy_layer_mh_forward, requant
from numpy_llama_ref import _load_qw, _load_bf16, VOCAB_SIZE, EMB_DIM
from test_rmsnorm_int8_dyn import numpy_rmsnorm_int8_dyn
from test_ffn_half import pack_perchan_slots
from numpy_sample_streamed import sample_streamed_reference
from numpy_sample import EXP_QUANT_SCALE
from gen_exp_lut import exp_lut
from gen_silu_lut import silu_lut
from test_flowkv import EXP_QUANT_SCALE as FKV_EXP_QUANT_SCALE
from aie2_chain_dynscale_mh import SILU_GATE_SCALE
from test_chain_mh import pack_blobs

DATA_DIR = Path(__file__).parent / "data"


def pack_lmw(embed_i8, embed_sc, gamma):
    """lmw = [final_norm gamma bf16(D) | lm_head per-tile weight slots]."""
    bias = np.zeros(VOCAB, np.int32)
    wpacked = pack_perchan_slots(
        embed_i8, embed_sc.astype(np.float32), bias, N_TILE, prefix_bytes=b"\x00" * 64
    )
    assert wpacked.size == WLM_TOTAL, (wpacked.size, WLM_TOTAL)
    gbytes = np.frombuffer(gamma.tobytes(), dtype=np.int8)
    blob = np.zeros(gbytes.size + WLM_TOTAL, dtype=np.int8)
    blob[: gbytes.size] = gbytes
    blob[gbytes.size :] = wpacked
    return blob


def oracle_token(hidden_i8, hidden_scale, gamma, embed_i8, embed_sc, lut):
    normed_i8, norm_scale = numpy_rmsnorm_int8_dyn(hidden_i8, gamma, hidden_scale)
    acc = embed_i8.astype(np.int64) @ normed_i8.astype(np.int64)
    logits = acc.astype(np.float32) * np.float32(norm_scale) * embed_sc.astype(
        np.float32
    )
    return sample_streamed_reference(logits, SAMPLE_TEMP, SAMPLE_TOPK, SAMPLE_SEED, lut)


def run_one_seed(seed, opts, lut_exp, lut_silu, npu, embed_i8, embed_sc, gamma, lmw, lut):
    rng = np.random.default_rng(seed)
    x_fp = rng.uniform(-1.6, 1.6, size=D).astype(np.float32)
    res_scale = np.float32(np.maximum(np.abs(x_fp).max(), 1e-12) / 127.0)
    x_i8 = requant(x_fp, np.float32(1.0) / res_scale)

    P = T // 2
    layers = []
    for L in range(N_LAYERS):
        layer = gen_layer_mh(rng)
        layer["lut_exp"] = lut_exp
        layer["lut_silu"] = lut_silu
        k_slot = rng.uniform(0.02, 0.08, size=(N_HEADS_KV, T)).astype(np.float32)
        v_slot = rng.uniform(0.02, 0.08, size=(N_HEADS_KV, T)).astype(np.float32)
        for h in range(N_HEADS_KV):
            layer["kcaches"][h][P * HEAD_D : (P + 1) * HEAD_D] = 0
            layer["vcaches"][h][P * HEAD_D : (P + 1) * HEAD_D] = 0
            k_slot[h, P] = 0.0
            v_slot[h, P] = 0.0
        layer["k_scales_slot"] = k_slot
        layer["v_scales_slot"] = v_slot
        layer["t_used"] = P + 1
        layers.append(layer)

    kin = [[c.copy() for c in lyr["kcaches"]] for lyr in layers]
    vin = [[c.copy() for c in lyr["vcaches"]] for lyr in layers]
    kslot_in = [lyr["k_scales_slot"].copy() for lyr in layers]
    vslot_in = [lyr["v_scales_slot"].copy() for lyr in layers]

    x_cur = (x_i8.copy(), float(res_scale))
    for L in range(N_LAYERS):
        x_cur, scales = numpy_layer_mh_forward(
            x_cur, layers[L], position=P, residual_dyn=True,
            attn_perslot=True, attn_lut=True,
        )
        layers[L]["scales"] = scales
    xo_i8, xo_scale = x_cur

    for L in range(N_LAYERS):
        layers[L]["kcaches"] = kin[L]
        layers[L]["vcaches"] = vin[L]
        layers[L]["k_scales_slot"] = kslot_in[L]
        layers[L]["v_scales_slot"] = vslot_in[L]
    wblob, kvblob = pack_blobs(layers)

    ref_tok = oracle_token(xo_i8, np.float32(xo_scale), gamma, embed_i8, embed_sc, lut)

    xin = np.zeros(D + 8, dtype=np.int8)
    xin[:D] = x_i8
    xin[D : D + 4] = np.frombuffer(np.float32(res_scale).tobytes(), dtype=np.int8)

    x_t = iron.tensor(xin, dtype=np.int8)
    w_t = iron.tensor(wblob, dtype=np.int8)
    kv_t = iron.tensor(kvblob, dtype=np.int8)
    lmw_t = iron.tensor(lmw, dtype=np.int8)
    tok_t = iron.zeros([1], dtype=np.int32)
    rc = DefaultNPURuntime.run_test(
        npu, [x_t, w_t, kv_t, lmw_t, tok_t], {}, verify=False, verbosity=opts.verbosity
    )
    if rc != 0:
        print(f"seed {seed}: dispatch returned {rc}", file=sys.stderr)
        return 1
    tok_t.to("cpu")
    dev_tok = int(tok_t.numpy()[0])
    ok = dev_tok == ref_tok
    print(f"seed {seed}: {'PASS' if ok else 'FAIL'}  dev_tok={dev_tok} ref_tok={ref_tok}")
    return 0 if ok else 1


def main():
    p = test_utils.create_default_argparser()
    p.add_argument("--seeds", type=str, default="0,1,7,42")
    opts = p.parse_args()
    seeds = [int(s) for s in opts.seeds.split(",")]

    assert VOCAB == VOCAB_SIZE and D == EMB_DIM
    print(f"loading embed/final_norm from {DATA_DIR} ...", flush=True)
    embed_i8, embed_sc = _load_qw(DATA_DIR, "embed", (VOCAB, D))
    gamma = _load_bf16(DATA_DIR, "final_norm", (D,))
    print("packing lm_head weights (262 MB) ...", flush=True)
    lmw = pack_lmw(embed_i8, embed_sc, gamma)

    npu = test_utils.create_npu_kernel(opts).npu_kernel
    lut_exp = exp_lut(FKV_EXP_QUANT_SCALE).astype(np.float32)
    lut_silu = silu_lut(SILU_GATE_SCALE)
    lut = exp_lut(EXP_QUANT_SCALE).astype(np.float32)

    print(
        f"chain_sample_mh N_LAYERS={N_LAYERS} T={T}  "
        f"temp={SAMPLE_TEMP} topk={SAMPLE_TOPK} seed={SAMPLE_SEED}  seeds={seeds}",
        flush=True,
    )
    fails = 0
    for s in seeds:
        fails += run_one_seed(
            s, opts, lut_exp, lut_silu, npu, embed_i8, embed_sc, gamma, lmw, lut
        ) != 0
    print(f"\nchain_sample_mh: {len(seeds) - fails}/{len(seeds)} seeds PASS")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
