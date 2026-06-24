"""Persistent-loop test (capstone increment 1): PT tokens in ONE dispatch, the
sampled token's embed seed feeding back to the chain ON-CHIP (no host between
tokens). KV is held at a fixed position (host re-streams the pristine KV each
token), so each token is a forward over the SAME chain fixtures + KV with only
the layer-0 input changing.

The device returns PT packed records [seed int8[D] | scale f32 | token i32 | pad].
The numpy oracle runs the SAME loop: token 0 from the host xin; each subsequent
token's xin = the previous token's embed seed (embed[prev_token] requantised).
Bit-exact on every generated token id + embed seed proves the on-chip feedback.

Run:
  make chain_persist_mh CHAIN_MH_N=2 PT=4 ONESTREAM_KSET=8
  python test_chain_persist_mh.py -x build/final_chain_persist_N2_PT4.xclbin \\
      -i build/insts_chain_persist_N2_PT4.bin -k MLIR_AIE
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
    KV_BYTES,
    WLM_TOTAL,
    SAMPLE_TEMP,
    SAMPLE_TOPK,
    SAMPLE_SEED,
    ONESTREAM_KSET,
    PT,
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


def build_fixtures(seed):
    """Build the immutable chain fixtures + the layer-0 input for token 0. KV is
    fixed across tokens (numpy restores it each forward), so the only thing that
    changes per token is the layer-0 input."""
    rng = np.random.default_rng(seed)
    x_fp = rng.uniform(-1.6, 1.6, size=D).astype(np.float32)
    res_scale = np.float32(np.maximum(np.abs(x_fp).max(), 1e-12) / 127.0)
    x_i8 = requant(x_fp, np.float32(1.0) / res_scale)

    lut_exp = exp_lut(FKV_EXP_QUANT_SCALE).astype(np.float32)
    lut_silu = silu_lut(SILU_GATE_SCALE)
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
    return x_i8, res_scale, layers, P


def chain_forward_once(x_i8, res_scale, layers, P, embed_i8, embed_sc, gamma):
    """One token's forward: chain -> final_norm -> lm_head -> clean top-k sample
    -> token + embed seed. KV caches are restored by the caller each call."""
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

    # restore caches (fixed-position KV)
    for L in range(N_LAYERS):
        layers[L]["kcaches"] = kin[L]
        layers[L]["vcaches"] = vin[L]
        layers[L]["k_scales_slot"] = kslot_in[L]
        layers[L]["v_scales_slot"] = vslot_in[L]

    normed_i8, norm_scale = numpy_rmsnorm_int8_dyn(xo_i8, gamma, np.float32(xo_scale))
    acc = embed_i8.astype(np.int64) @ normed_i8.astype(np.int64)
    logits = (
        acc.astype(np.float32) * np.float32(norm_scale) * embed_sc.astype(np.float32)
    ).astype(np.float32)
    tok = topk_sample_reference(
        logits, SAMPLE_TEMP, SAMPLE_TOPK, SAMPLE_SEED, k_set=ONESTREAM_KSET
    )
    xin, scale = embed_seed_ref(embed_i8[tok], float(embed_sc[tok]))
    return tok, xin, scale


def main():
    p = test_utils.create_default_argparser()
    p.add_argument("--seed", type=int, default=0)
    opts = p.parse_args()
    fseed = opts.seed

    assert VOCAB == VOCAB_SIZE and D == EMB_DIM
    print(f"loading embed/final_norm from {DATA_DIR} ...", flush=True)
    embed_i8, embed_sc = _load_qw(DATA_DIR, "embed", (VOCAB, D))
    gamma = _load_bf16(DATA_DIR, "final_norm", (D,))
    print("packing lm_head weights (262 MB) ...", flush=True)
    lmw = pack_lmw(embed_i8, embed_sc, gamma)

    x_i8, res_scale, layers, P = build_fixtures(fseed)

    # Oracle: run the autoregressive loop in numpy. token 0 from host xin; each
    # next token's xin = previous token's embed seed. (chain_forward_once also
    # populates layer["scales"], needed by pack_blobs below -- the scales are
    # deterministic given the fixtures + fixed KV, identical every token.)
    ref_tokens = []
    cur_x, cur_sc = x_i8, res_scale
    for _ in range(PT):
        tok, nxin, nscale = chain_forward_once(
            cur_x, cur_sc, layers, P, embed_i8, embed_sc, gamma
        )
        ref_tokens.append(tok)
        cur_x, cur_sc = nxin, nscale

    # Pack the chain weights/KV AFTER a forward has populated layer["scales"].
    wblob, kvblob = pack_blobs(layers)

    # Device dispatch. xin = token 0's layer-0 input; KV buffer is [pristine |
    # scratch] (2x) -- fills read pristine, drains write scratch.
    xin = np.zeros(D + 8, dtype=np.int8)
    xin[:D] = x_i8
    xin[D : D + 4] = np.frombuffer(np.float32(res_scale).tobytes(), dtype=np.int8)
    kv2 = np.zeros(2 * KV_BYTES, dtype=np.int8)
    kv2[:KV_BYTES] = kvblob

    npu = test_utils.create_npu_kernel(opts).npu_kernel
    x_t = iron.tensor(xin, dtype=np.int8)
    w_t = iron.tensor(wblob, dtype=np.int8)
    kv_t = iron.tensor(kv2, dtype=np.int8)
    lmw_t = iron.tensor(lmw, dtype=np.int8)
    out_t = iron.zeros([PT * (D + 12)], dtype=np.int8)
    print(
        f"chain_persist_mh N_LAYERS={N_LAYERS} PT={PT} KSET={ONESTREAM_KSET}  "
        f"temp={SAMPLE_TEMP} topk={SAMPLE_TOPK} seed={SAMPLE_SEED}  fixture_seed={fseed}",
        flush=True,
    )
    rc = DefaultNPURuntime.run_test(
        npu, [x_t, w_t, kv_t, lmw_t, out_t], {}, verify=False, verbosity=opts.verbosity
    )
    if rc != 0:
        print(f"dispatch returned {rc}", file=sys.stderr)
        return rc
    out_t.to("cpu")
    packed = out_t.numpy()

    fails = 0
    for tok in range(PT):
        rec = packed[tok * (D + 12) : (tok + 1) * (D + 12)]
        dev_tok = int(np.frombuffer(rec[D + 4 : D + 8].tobytes(), np.int32)[0])
        ok = dev_tok == ref_tokens[tok]
        fails += not ok
        print(
            f"  token {tok}: {'PASS' if ok else 'FAIL'}  dev={dev_tok} ref={ref_tokens[tok]}"
        )

    print(
        f"\nchain_persist_mh: {PT - fails}/{PT} tokens PASS  "
        f"(on-chip autoregressive feedback)"
    )
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
