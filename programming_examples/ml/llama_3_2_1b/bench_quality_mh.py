"""Phase 8 quality benchmark: top-1 token agreement between fp16
numpy_llama_ref and our INT8 multi-head chain on a curated factoid
prompt set. Use this to track quantization-quality work (static scales,
mixed precision, etc.).

For each prompt:
  - fp16 ref:  numpy_llama_ref.forward_full -> argmax = "right" token
  - INT8 chain (numpy only, fast):
      prefill all but last prompt token via numpy_layer_mh_forward
      decode last prompt token's hidden -> argmax = predicted token
  - Report match (== fp16) per prompt and overall rate.

Numpy-only by default (orders of magnitude faster than device for 20
prompts). Pass --device to verify a subset on hardware too.

Run:
  python bench_quality_mh.py                     # numpy only, ~2-3 min
  python bench_quality_mh.py --device \\
      -x build/final_chain_mh_N16_T128.xclbin \\
      -i build/insts_chain_mh_N16_T128.bin -k MLIR_AIE
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

from aie2_chain_dynscale_mh import (
    D,
    N_LAYERS,
    ACT_SCALE,
    INV_ACT_SCALE,
    SILU_GATE_SCALE,
    HEAD_D,
    T,
)
from numpy_layer_mh import gen_real_layer_mh, numpy_layer_mh_forward, requant
from test_flowkv import EXP_QUANT_SCALE
from gen_exp_lut import exp_lut
from gen_silu_lut import silu_lut
from numpy_llama_ref import (
    load_model,
    lm_head_logits,
    rope_cos_sin,
    forward_full,
)
from generate import load_tokenizer

# Curated set: each prompt has a "naturally correct" next token for a
# well-tuned Llama 3.2 1B. We let the fp16 ref tell us the ground-truth
# token rather than hardcoding, so this list works for whatever model
# checkpoint sits in data/.
PROMPTS = [
    "The capital of France is",
    "The capital of Italy is",
    "The capital of Japan is",
    "The capital of Germany is",
    "The author of Hamlet is",
    "The largest planet in our solar system is",
    "The chemical symbol for water is",
    "The first president of the United States was",
    "The opposite of black is",
    "The number of legs on a spider is",
    "Two plus two equals",
    "The sun rises in the",
    "Romeo and Juliet was written by",
    "Mount Everest is in the country of",
    "The Eiffel Tower is located in",
    "The boiling point of water in Celsius is",
    "The smallest prime number is",
    "Shakespeare wrote in the",
    "The currency of Japan is the",
    "The Great Wall is in",
]


def embed_token_to_int8(model, token_id):
    row = model.embed_i8[token_id].astype(np.float32) * model.embed_sc[token_id]
    return requant(row, INV_ACT_SCALE)


def int8_chain_next_token(model, layers, cos_lut, sin_lut, token_ids,
                          residual_fp32=False, residual_dyn=False,
                          attn_lut=False):
    """Run our INT8 chain over `token_ids`. Returns the predicted next
    token id. Resets layer caches before running (so prompts are
    independent)."""
    # Reset KV caches per prompt (so prompt boundaries are clean).
    for layer in layers:
        for h in range(len(layer["kcaches"])):
            layer["kcaches"][h] = np.zeros_like(layer["kcaches"][h])
            layer["vcaches"][h] = np.zeros_like(layer["vcaches"][h])
        layer["k_scales"][:] = 1e-6
        layer["v_scales"][:] = 1e-6
        # Reset per-slot KV scales too (per-position; created lazily).
        if "k_scales_slot" in layer:
            layer["k_scales_slot"][:] = 1e-6
            layer["v_scales_slot"][:] = 1e-6

    # Process each token through the layers. The last token's hidden
    # output -> lm_head -> argmax = next token.
    last_hidden = None
    for pos, tok in enumerate(token_ids):
        if pos >= T:
            print(f"  WARNING: prompt longer than cache T={T}; truncating")
            break
        row = model.embed_i8[int(tok)].astype(np.float32) * model.embed_sc[int(tok)]
        if residual_fp32:
            x = row.astype(np.float32)
        elif residual_dyn:
            # Seed the residual with a per-token dynamic scale (int8, scale).
            sc = np.float32(max(float(np.abs(row).max()), 1e-12) / 127.0)
            x = (requant(row, np.float32(1.0) / sc), float(sc))
        else:
            x = embed_token_to_int8(model, int(tok))
        c = cos_lut[pos].astype(bfloat16)
        s = sin_lut[pos].astype(bfloat16)
        for layer in layers:
            layer["cos"] = c
            layer["sin"] = s
        xc = x.copy() if not residual_dyn else (x[0].copy(), x[1])
        for L in range(N_LAYERS):
            xc, scales = numpy_layer_mh_forward(
                xc, layers[L], position=pos,
                residual_fp32=residual_fp32, residual_dyn=residual_dyn,
                attn_lut=attn_lut,
            )
            layers[L]["scales"] = scales
            layers[L]["t_used"] = pos + 1
        last_hidden = xc
    # Final hidden -> fp32 for lm_head.
    if residual_fp32:
        hidden_fp = last_hidden
    elif residual_dyn:
        hidden_fp = last_hidden[0].astype(np.float32) * np.float32(last_hidden[1])
    else:
        hidden_fp = last_hidden.astype(np.float32) * ACT_SCALE
    logits = lm_head_logits(model, hidden_fp[None, :])[0]
    return int(np.argmax(logits))


def fp16_ref_next_token(model, token_ids):
    """Ground-truth next token via numpy_llama_ref.forward_full."""
    logits = forward_full(model, np.array(token_ids))  # (M, V)
    return int(np.argmax(logits[-1]))


def device_chain_next_token(model, layers, cos_lut, sin_lut, token_ids, dev,
                            self_prefill=False):
    """Real-weight device decode validation.

    Two modes:
    - host-prefill (default): numpy prefills the prompt (builds per-layer KV
      caches + last-token scales), then the device dispatches ONCE for the last
      token using the host-supplied cache. Validates the decode dataflow.
    - self_prefill=True (on-chip KV append): the device computes + stores K/V
      itself. Dispatch the chain ONCE PER PROMPT TOKEN, carrying the device-
      DRAINED cache forward as the next token's input cache -- no numpy K/V
      math. numpy still computes the per-layer dynamic SCALES each token (the
      device reads them from wblob prefixes; on-chip self-calibration is a
      future step). Returns final_norm->lm_head->argmax of the last token's
      DEVICE hidden.

    `dev` is a dict: {npu_kernel, pack_blobs, iron, DefaultNPURuntime}.
    Always device-faithful: residual_dyn=True, attn_lut=True.
    """
    import struct

    import numpy as _np

    iron = dev["iron"]
    run_test = dev["DefaultNPURuntime"].run_test
    npu = dev["npu_kernel"]
    pack_blobs = dev["pack_blobs"]

    # Reset caches per prompt.
    for layer in layers:
        for h in range(len(layer["kcaches"])):
            layer["kcaches"][h] = _np.zeros_like(layer["kcaches"][h])
            layer["vcaches"][h] = _np.zeros_like(layer["vcaches"][h])
        layer["k_scales"][:] = 1e-6
        layer["v_scales"][:] = 1e-6
        if "k_scales_slot" in layer:
            layer["k_scales_slot"][:] = 1e-6
            layer["v_scales_slot"][:] = 1e-6

    # Device-owned KV cache carried across per-token dispatches (self_prefill).
    # Built once; each dispatch reads it and the drain overwrites it in place.
    kv_state = None

    last_out = None  # (int8[D], scale) hidden of the last processed token
    for pos, tok in enumerate(token_ids):
        if pos >= T:
            print(f"  WARNING: prompt longer than cache T={T}; truncating")
            break
        row = model.embed_i8[int(tok)].astype(np.float32) * model.embed_sc[int(tok)]
        sc = np.float32(max(float(np.abs(row).max()), 1e-12) / 127.0)
        x = (requant(row, np.float32(1.0) / sc), float(sc))
        c = cos_lut[pos].astype(bfloat16)
        s = sin_lut[pos].astype(bfloat16)
        for layer in layers:
            layer["cos"] = c
            layer["sin"] = s

        # Numpy forward: in self_prefill it provides the per-layer SCALES only
        # (and advances its own caches, unused by the device); in host-prefill
        # it provides BOTH scales and the cache.
        xc = (x[0].copy(), x[1])
        for L in range(N_LAYERS):
            xc, scales = numpy_layer_mh_forward(
                xc, layers[L], position=pos, residual_dyn=True, attn_lut=True
            )
            layers[L]["scales"] = scales
            layers[L]["t_used"] = pos + 1

        if not self_prefill:
            last_out = x  # device re-seeds with the embedding (host-prefill)
            continue

        # self_prefill: dispatch the device for THIS token. wblob carries this
        # token's scales; kvblob carries the device-owned cache so far (slots
        # 0..pos-1), with slot pos to be written on-chip. The device appends
        # slot pos, attends over 0..pos, and drains the updated cache back.
        wblob, _kvblob_host = pack_blobs(layers)
        if kv_state is None:
            # First token: start from an all-zero device cache.
            kv_state = _np.zeros_like(_kvblob_host)
        # Set per-head T_used = pos+1 in the device cache (flowkv/append read it).
        _set_kv_t_used(kv_state, pos + 1)

        xin = _np.zeros(D + 8, dtype=_np.int8)
        xin[:D] = x[0]
        xin[D : D + 4] = _np.frombuffer(np.float32(x[1]).tobytes(), dtype=_np.int8)

        x_t = iron.tensor(xin, dtype=_np.int8)
        w_t = iron.tensor(wblob, dtype=_np.int8)
        kv_t = iron.tensor(kv_state, dtype=_np.int8)
        o_t = iron.zeros([D + 8], dtype=_np.int8)
        rc = run_test(npu, [x_t, w_t, kv_t, o_t], {}, verify=False, verbosity=0)
        if rc != 0:
            raise RuntimeError(f"device dispatch returned {rc}")
        kv_t.to("cpu")
        kv_state = kv_t.numpy().copy()  # device-owned cache for the next token
        o_t.to("cpu")
        last_out = o_t.numpy()

    if not self_prefill:
        # host-prefill: dispatch ONCE for the last token.
        wblob, kvblob = pack_blobs(layers)
        x_i8, res_scale = last_out
        xin = _np.zeros(D + 8, dtype=_np.int8)
        xin[:D] = x_i8
        xin[D : D + 4] = _np.frombuffer(np.float32(res_scale).tobytes(), dtype=_np.int8)
        x_t = iron.tensor(xin, dtype=_np.int8)
        w_t = iron.tensor(wblob, dtype=_np.int8)
        kv_t = iron.tensor(kvblob, dtype=_np.int8)
        o_t = iron.zeros([D + 8], dtype=_np.int8)
        rc = run_test(npu, [x_t, w_t, kv_t, o_t], {}, verify=False, verbosity=0)
        if rc != 0:
            raise RuntimeError(f"device dispatch returned {rc}")
        o_t.to("cpu")
        last_out = o_t.numpy()

    dev_scale = struct.unpack("<f", last_out[D : D + 4].tobytes())[0]
    hidden_fp = last_out[:D].astype(np.float32) * np.float32(dev_scale)
    logits = lm_head_logits(model, hidden_fp[None, :])[0]
    return int(np.argmax(logits))


def _set_kv_t_used(kvblob, t_used):
    """Write per-head T_used into every KV-head prefix of the device cache."""
    import numpy as _np
    from aie2_chain_dynscale_mh import (
        N_LAYERS, N_HEADS_KV, PER_LAYER_KV, PER_KV_HEAD_BYTES,
    )

    tb = _np.frombuffer(_np.int32(t_used).tobytes(), dtype=_np.int8)
    for L in range(N_LAYERS):
        base = L * PER_LAYER_KV
        for h in range(N_HEADS_KV):
            off = base + h * PER_KV_HEAD_BYTES
            kvblob[off : off + 4] = tb


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=str, default=str(Path(__file__).parent / "data")
    )
    parser.add_argument(
        "--prompts",
        type=str,
        default=None,
        help="Comma-separated indices into PROMPTS (default: all)",
    )
    parser.add_argument("--rng-seed", type=int, default=0)
    parser.add_argument(
        "--residual-fp32",
        action="store_true",
        help="carry the residual stream in fp32 (per-token dynamic) instead "
        "of int8@static-0.05 — proof-of-concept for the residual-format fix",
    )
    parser.add_argument(
        "--residual-dyn",
        action="store_true",
        help="device-faithful per-token int8 residual (int8 + fp32 scale "
        "tail; o_proj/down dynamic-out, rescale-add) — the Phase B target",
    )
    parser.add_argument(
        "--attn-lut",
        action="store_true",
        help="device-faithful per-slot KV attention with the exp-LUT softmax "
        "(flowkv_mh mirror) instead of the fp32-softmax golden — measures "
        "whether the LUT ULP noise affects top-1",
    )
    parser.add_argument(
        "--device",
        action="store_true",
        help="run the INT8 decode of the last token on HARDWARE via the N=16 "
        "chain xclbin (prefill stays in numpy; the device has no on-chip KV "
        "append). Requires -x/-i/-k. Implies residual_dyn + attn_lut.",
    )
    parser.add_argument(
        "--self-prefill",
        action="store_true",
        help="device self-prefills via on-chip KV append: dispatch the chain "
        "once per prompt token, carrying the device-drained cache forward (no "
        "numpy K/V math). Requires --device + an on-chip-append xclbin.",
    )
    # NPU kernel args (-x/-i/-k) only needed with --device.
    parser.add_argument("-x", "--xclbin", type=str, default=None, dest="xclbin")
    parser.add_argument("-i", "--instr", type=str, default=None, dest="instr")
    parser.add_argument("-k", "--kernel", type=str, default=None, dest="kernel")
    parser.add_argument("-v", "--verbosity", type=int, default=0)
    opts = parser.parse_args()

    data_dir = Path(opts.data_dir)
    print(f"loading model from {data_dir} ...", flush=True)
    model = load_model(data_dir)
    enc = load_tokenizer(data_dir / "tokenizer.model")

    rng = np.random.default_rng(opts.rng_seed)
    lut_exp = exp_lut(EXP_QUANT_SCALE).astype(np.float32)
    lut_silu = silu_lut(SILU_GATE_SCALE)
    layers = []
    for L in range(N_LAYERS):
        layer = gen_real_layer_mh(L, data_dir, rng)
        layer["lut_exp"] = lut_exp
        layer["lut_silu"] = lut_silu
        layers.append(layer)
    cos_lut, sin_lut = rope_cos_sin(np.arange(T), HEAD_D)
    print(f"loaded {N_LAYERS} mh layers + lm_head\n", flush=True)

    # Device setup (only with --device): build the NPU kernel + a dev dict
    # that device_chain_next_token uses to pack blobs and dispatch.
    dev = None
    if opts.device:
        import aie.iron as iron
        from aie.utils import DefaultNPURuntime
        import aie.utils.test as test_utils

        assert opts.xclbin and opts.instr and opts.kernel, (
            "--device requires -x <xclbin> -i <instr> -k <kernel>"
        )
        from test_chain_mh import pack_blobs as chain_pack_blobs

        npu_kernel = test_utils.create_npu_kernel(opts).npu_kernel
        dev = {
            "npu_kernel": npu_kernel,
            "pack_blobs": chain_pack_blobs,
            "iron": iron,
            "DefaultNPURuntime": DefaultNPURuntime,
        }
        print(f"device mode: N={N_LAYERS} chain xclbin {opts.xclbin}\n", flush=True)

    indices = (
        range(len(PROMPTS))
        if opts.prompts is None
        else [int(s) for s in opts.prompts.split(",")]
    )
    n_total = 0
    n_match = 0
    results = []  # list of (prompt, ref_tok, our_tok, match)

    for idx in indices:
        prompt = PROMPTS[idx]
        ids = [128000] + enc.encode(prompt)
        t0 = time.time()
        ref_tok = fp16_ref_next_token(model, ids)
        t_fp16 = time.time() - t0
        t0 = time.time()
        if opts.device:
            our_tok = device_chain_next_token(
                model, layers, cos_lut, sin_lut, ids, dev,
                self_prefill=opts.self_prefill,
            )
        else:
            our_tok = int8_chain_next_token(
                model, layers, cos_lut, sin_lut, ids,
                residual_fp32=opts.residual_fp32, residual_dyn=opts.residual_dyn,
                attn_lut=opts.attn_lut,
            )
        t_int8 = time.time() - t0
        match = ref_tok == our_tok
        n_total += 1
        n_match += int(match)
        results.append((prompt, ref_tok, our_tok, match))
        mark = "OK" if match else "FAIL"
        print(
            f"  [{mark}] {prompt!r:50} -> fp16={enc.decode([ref_tok])!r} "
            f"int8={enc.decode([our_tok])!r}  "
            f"(t_fp16={t_fp16:.1f}s t_int8={t_int8:.1f}s)",
            flush=True,
        )

    print(
        f"\ntop-1 agreement: {n_match}/{n_total} = "
        f"{100.0 * n_match / max(1, n_total):.1f}%"
    )
    return 0 if n_match == n_total else 1


if __name__ == "__main__":
    sys.exit(main())
