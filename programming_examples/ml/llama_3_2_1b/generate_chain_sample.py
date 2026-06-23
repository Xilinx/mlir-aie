"""End-to-end autoregressive generation on the fused N=16 chain+lm_head+sampler.

Host-driven loop, device does ALL the per-token compute (16 layers + KV append +
self-cal + final_norm + lm_head + sampler -> token id). The host only:
  - feeds each input token's embedding (embed[tok] -> int8[D]+scale), and
  - carries the device-drained KV cache forward,
  - feeds the device's output token back as the next input.
No host transformer/lm_head compute. (On-chip embed gather lands in the
persistent loop; here the host does the tiny embed lookup.)

Per token = one device dispatch of final_chain_sample_N16 (LLAMA_CHAIN_SAMPLE=1,
greedy). The prompt is prefilled token-by-token (on-chip KV append); then K new
tokens are generated autoregressively. A numpy autoregressive reference (device-
faithful int8 path + host lm_head argmax) gives the device-vs-numpy token
agreement along the generated sequence.

Run:
  make chain_sample_mh CHAIN_MH_N=16
  python generate_chain_sample.py -x build/final_chain_sample_N16_T128.xclbin \\
      -i build/insts_chain_sample_N16_T128.bin -k MLIR_AIE \\
      --prompt "The capital of France is" --new-tokens 8
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

import aie.iron as iron
import aie.utils.test as test_utils
from aie.utils import DefaultNPURuntime

from aie2_chain_dynscale_mh import D, T, N_LAYERS, HEAD_D, SILU_GATE_SCALE
from numpy_layer_mh import gen_real_layer_mh, numpy_layer_mh_forward, requant
from numpy_llama_ref import load_model, lm_head_logits, rope_cos_sin
from test_flowkv import EXP_QUANT_SCALE
from gen_exp_lut import exp_lut
from gen_silu_lut import silu_lut
from generate import load_tokenizer
from test_chain_mh import pack_blobs
from test_chain_sample_mh import pack_lmw
from bench_quality_mh import _set_kv_t_used

DATA_DIR = Path(__file__).parent / "data"


def embed_xin(model, tok):
    row = model.embed_i8[int(tok)].astype(np.float32) * model.embed_sc[int(tok)]
    sc = np.float32(max(float(np.abs(row).max()), 1e-12) / 127.0)
    xi = requant(row, np.float32(1.0) / sc)
    xin = np.zeros(D + 8, dtype=np.int8)
    xin[:D] = xi
    xin[D : D + 4] = np.frombuffer(np.float32(sc).tobytes(), dtype=np.int8)
    return xin, (xi, float(sc))


def main():
    p = test_utils.create_default_argparser()
    p.add_argument("--prompt", type=str, default="The capital of France is")
    p.add_argument("--new-tokens", type=int, default=8)
    opts = p.parse_args()

    print(f"loading model from {DATA_DIR} ...", flush=True)
    model = load_model(DATA_DIR)
    enc = load_tokenizer(DATA_DIR / "tokenizer.model")
    rng = np.random.default_rng(0)
    lut_exp = exp_lut(EXP_QUANT_SCALE).astype(np.float32)
    lut_silu = silu_lut(SILU_GATE_SCALE)
    layers = []
    for L in range(N_LAYERS):
        layer = gen_real_layer_mh(L, DATA_DIR, rng)
        layer["lut_exp"] = lut_exp
        layer["lut_silu"] = lut_silu
        layers.append(layer)
    cos_lut, sin_lut = rope_cos_sin(np.arange(T), HEAD_D)
    print("packing lm_head weights (262 MB) ...", flush=True)
    lmw = pack_lmw(model.embed_i8, model.embed_sc, model.final_norm)
    npu = test_utils.create_npu_kernel(opts).npu_kernel

    ids = [128000] + enc.encode(opts.prompt)
    print(f"prompt: {opts.prompt!r}  ({len(ids)} tokens incl BOS)", flush=True)

    for layer in layers:
        for h in range(len(layer["kcaches"])):
            layer["kcaches"][h] = np.zeros_like(layer["kcaches"][h])
            layer["vcaches"][h] = np.zeros_like(layer["vcaches"][h])
        if "k_scales_slot" in layer:
            layer["k_scales_slot"][:] = 1e-6
            layer["v_scales_slot"][:] = 1e-6

    kv_state = None
    gen = []
    match = 0
    cur = ids[0]
    pos = 0
    n_steps = len(ids) - 1 + opts.new_tokens
    for step in range(n_steps):
        if pos >= T:
            print(f"  reached cache T={T}; stopping")
            break
        xin, x = embed_xin(model, cur)
        c = cos_lut[pos].astype(bfloat16)
        s = sin_lut[pos].astype(bfloat16)
        for layer in layers:
            layer["cos"] = c
            layer["sin"] = s
        xc = (x[0].copy(), x[1])
        for L in range(N_LAYERS):
            xc, scales = numpy_layer_mh_forward(
                xc, layers[L], position=pos, residual_dyn=True, attn_lut=True
            )
            layers[L]["scales"] = scales
            layers[L]["t_used"] = pos + 1
        xo_i8, xo_scale = xc
        ref_logits = lm_head_logits(
            model, (xo_i8.astype(np.float32) * np.float32(xo_scale))[None, :]
        )[0]
        ref_tok = int(np.argmax(ref_logits))

        wblob, _kv = pack_blobs(layers)
        if kv_state is None:
            kv_state = np.zeros_like(_kv)
        _set_kv_t_used(kv_state, pos + 1)

        x_t = iron.tensor(xin, dtype=np.int8)
        w_t = iron.tensor(wblob, dtype=np.int8)
        kv_t = iron.tensor(kv_state, dtype=np.int8)
        lmw_t = iron.tensor(lmw, dtype=np.int8)
        tok_t = iron.zeros([1], dtype=np.int32)
        rc = DefaultNPURuntime.run_test(
            npu, [x_t, w_t, kv_t, lmw_t, tok_t], {}, verify=False, verbosity=0
        )
        if rc != 0:
            raise RuntimeError(f"dispatch returned {rc}")
        kv_t.to("cpu")
        kv_state = kv_t.numpy().copy()
        tok_t.to("cpu")
        dev_tok = int(tok_t.numpy()[0])

        in_prompt = step < len(ids) - 1
        agree = dev_tok == ref_tok
        match += agree
        if in_prompt:
            nxt = ids[step + 1]
            tag = "prefill"
        else:
            nxt = dev_tok
            gen.append(dev_tok)
            tag = "gen"
        extra = f"  -> {enc.decode([dev_tok])!r}" if not in_prompt else ""
        print(
            f"  step {step:2d} [{tag:7s}] pos={pos} dev_tok={dev_tok} "
            f"ref_tok={ref_tok} {'ok' if agree else 'DIFF'}{extra}"
        )
        cur = nxt
        pos += 1

    print(f"\ndevice/numpy token agreement: {match}/{n_steps}")
    print(f"generated: {enc.decode(gen)!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
