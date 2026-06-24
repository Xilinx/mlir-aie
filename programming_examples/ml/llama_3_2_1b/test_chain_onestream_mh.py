"""One-stream test: fused N-layer chain + lm_head GEMM + top-k insert + finalize
-> token AND next-token embed seed, in ONE 262 MB table pass.

Same random-fixture chain as test_chain_sample_mh, but the device
(LLAMA_CHAIN_ONESTREAM=1 xclbin) produces both the token id and the requantised
next-token embedding seed (int8[D]+scale) on-chip, with no DDR logits scratch and
no second gather stream. Verifies both against the numpy oracle (clean top-k
renormalisation -- see numpy_topk_sample / project_topk_masked_tail_bug).

Run:
  make chain_onestream_mh CHAIN_MH_N=2 ONESTREAM_KSET=8
  python test_chain_onestream_mh.py -x build/final_chain_onestream_N2_T128.xclbin \\
      -i build/insts_chain_onestream_N2_T128.bin -k MLIR_AIE
"""

from __future__ import annotations

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
    WLM_TOTAL,
    SAMPLE_TEMP,
    SAMPLE_TOPK,
    SAMPLE_SEED,
    ONESTREAM_KSET,
)
from numpy_layer_mh import gen_layer_mh, numpy_layer_mh_forward, requant
from numpy_llama_ref import _load_qw, _load_bf16, VOCAB_SIZE, EMB_DIM
from test_rmsnorm_int8_dyn import numpy_rmsnorm_int8_dyn
from test_ffn_half import pack_perchan_slots
from numpy_topk_sample import topk_sample_reference
from gen_exp_lut import exp_lut
from gen_silu_lut import silu_lut
from test_flowkv import EXP_QUANT_SCALE as FKV_EXP_QUANT_SCALE
from aie2_chain_dynscale_mh import SILU_GATE_SCALE
from test_chain_mh import pack_blobs

DATA_DIR = Path(__file__).parent / "data"


def pack_lmw(embed_i8, embed_sc, gamma):
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


def embed_seed_ref(row_i8, embed_sc):
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


def oracle(hidden_i8, hidden_scale, gamma, embed_i8, embed_sc):
    """final_norm -> lm_head logits (= sum_i32 * norm_scale * embed_sc) -> clean
    top-k sample -> token + next-token embed seed. Mirrors the fused kernel."""
    normed_i8, norm_scale = numpy_rmsnorm_int8_dyn(hidden_i8, gamma, hidden_scale)
    acc = embed_i8.astype(np.int64) @ normed_i8.astype(np.int64)
    logits = (
        acc.astype(np.float32) * np.float32(norm_scale) * embed_sc.astype(np.float32)
    ).astype(np.float32)
    tok = topk_sample_reference(
        logits, SAMPLE_TEMP, SAMPLE_TOPK, SAMPLE_SEED, k_set=ONESTREAM_KSET
    )
    xin, scale = embed_seed_ref(embed_i8[tok], float(embed_sc[tok]))
    return tok, xin, scale


def run_one_seed(seed, opts, lut_exp, lut_silu, npu, embed_i8, embed_sc, gamma, lmw):
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
            x_cur,
            layers[L],
            position=P,
            residual_dyn=True,
            attn_perslot=True,
            attn_lut=True,
        )
        layers[L]["scales"] = scales
    xo_i8, xo_scale = x_cur

    for L in range(N_LAYERS):
        layers[L]["kcaches"] = kin[L]
        layers[L]["vcaches"] = vin[L]
        layers[L]["k_scales_slot"] = kslot_in[L]
        layers[L]["v_scales_slot"] = vslot_in[L]
    wblob, kvblob = pack_blobs(layers)

    ref_tok, ref_xin, ref_scale = oracle(
        xo_i8, np.float32(xo_scale), gamma, embed_i8, embed_sc
    )

    xin = np.zeros(D + 8, dtype=np.int8)
    xin[:D] = x_i8
    xin[D : D + 4] = np.frombuffer(np.float32(res_scale).tobytes(), dtype=np.int8)

    x_t = iron.tensor(xin, dtype=np.int8)
    w_t = iron.tensor(wblob, dtype=np.int8)
    kv_t = iron.tensor(kvblob, dtype=np.int8)
    lmw_t = iron.tensor(lmw, dtype=np.int8)
    # packed output [seed int8[D] | scale f32 | token i32 | pad] -- ONE buffer
    # to stay at 5 runtime args (run_test segfaults at ~6).
    out_t = iron.zeros([D + 12], dtype=np.int8)
    rc = DefaultNPURuntime.run_test(
        npu,
        [x_t, w_t, kv_t, lmw_t, out_t],
        {},
        verify=False,
        verbosity=opts.verbosity,
    )
    if rc != 0:
        print(f"seed {seed}: dispatch returned {rc}", file=sys.stderr)
        return 1
    out_t.to("cpu")
    packed = out_t.numpy()
    dev_xin = packed[:D]
    dev_scale = np.frombuffer(packed[D : D + 4].tobytes(), np.float32)[0]
    dev_tok = int(np.frombuffer(packed[D + 4 : D + 8].tobytes(), np.int32)[0])

    tok_ok = dev_tok == ref_tok
    xin_ok = np.array_equal(dev_xin, ref_xin)
    sc_ok = dev_scale == ref_scale
    ok = tok_ok and xin_ok and sc_ok
    print(
        f"seed {seed}: {'PASS' if ok else 'FAIL'}  dev_tok={dev_tok} ref_tok={ref_tok}"
        f"  seed_xin={'OK' if xin_ok else 'DIFF'} seed_scale={'OK' if sc_ok else 'DIFF'}"
    )
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

    print(
        f"chain_onestream_mh N_LAYERS={N_LAYERS} T={T} KSET={ONESTREAM_KSET}  "
        f"temp={SAMPLE_TEMP} topk={SAMPLE_TOPK} seed={SAMPLE_SEED}  seeds={seeds}",
        flush=True,
    )
    fails = 0
    for s in seeds:
        fails += (
            run_one_seed(
                s, opts, lut_exp, lut_silu, npu, embed_i8, embed_sc, gamma, lmw
            )
            != 0
        )
    print(f"\nchain_onestream_mh: {len(seeds) - fails}/{len(seeds)} seeds PASS")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
