#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
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
prompts). Pass --device to run each prompt through the NPU decode
(aie2_decode.py) instead, one token a dispatch from an empty KV cache: the
host writes each prompt token's embed seed into the stage, and the device's
token after the last one is the prediction.

Run:
  python bench_quality_mh.py                     # numpy only, ~2-3 min
  LLAMA_CHAIN_N=16 python bench_quality_mh.py --device --bf16-ref
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import aie.iron as iron
import numpy as np
from accuracy_vs_hf import RefModel, ref_forward_logits
from aie2_decode import ACT_BYTES, REC_BYTES, build, kv_words
from aie2_layer_mh import ACT_SCALE, HEAD_D, INV_ACT_SCALE, SILU_GATE_SCALE, D
from gen_exp_lut import exp_lut
from gen_llama_data import WEIGHTS_DIR, SafetensorsReader
from gen_silu_lut import silu_lut
from llama_npu import DispatchSession
from ml_dtypes import bfloat16
from numpy_layer_mh import gen_real_layer_mh, numpy_layer_mh_forward, requant
from numpy_llama_ref import (
    forward_full,
    lm_head_logits,
    load_model,
    load_tokenizer,
    rope_cos_sin,
    rope_table,
)
from test_chain_persist_grow_mh import (
    ATTN_CORES,
    ONESTREAM_KSET,
    P_MAX,
    SEED_AT,
    dispatch_params,
    embed_seed_ref,
    pack_lmw,
    pack_stage,
    pack_wblob,
    tokens_of,
)
from test_flowkv import EXP_QUANT_SCALE

T = int(os.environ.get("LLAMA_CHAIN_T", "128"))
N_LAYERS = int(os.environ.get("LLAMA_CHAIN_N", "2"))

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


def int8_chain_next_token(
    model,
    layers,
    cos_lut,
    sin_lut,
    token_ids,
    residual_fp32=False,
    residual_dyn=False,
    attn_lut=False,
):
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
            sc = np.float32(max(float(np.abs(row).max()), 1e-12) / 127.0)
            x = (requant(row, np.float32(1.0) / sc), float(sc))
        else:
            x = embed_token_to_int8(model, int(tok))
        c = cos_lut[pos].astype(bfloat16)
        s = sin_lut[pos].astype(bfloat16)
        for layer in layers:
            layer["cos"] = c
            layer["sin"] = s
        if residual_dyn:
            assert isinstance(x, tuple)
            xc = (x[0].copy(), x[1])
        else:
            assert isinstance(x, np.ndarray)
            xc = x.copy()
        for L in range(N_LAYERS):
            xc, scales = numpy_layer_mh_forward(
                xc,
                layers[L],
                position=pos,
                residual_fp32=residual_fp32,
                residual_dyn=residual_dyn,
                attn_lut=attn_lut,
            )
            layers[L]["scales"] = scales
            layers[L]["t_used"] = pos + 1
        last_hidden = xc
    # Final hidden -> fp32 for lm_head.
    if residual_fp32:
        assert isinstance(last_hidden, np.ndarray)
        hidden_fp = last_hidden
    elif residual_dyn:
        assert isinstance(last_hidden, tuple)
        hidden_fp = last_hidden[0].astype(np.float32) * np.float32(last_hidden[1])
    else:
        assert isinstance(last_hidden, np.ndarray)
        hidden_fp = last_hidden.astype(np.float32) * ACT_SCALE
    logits = lm_head_logits(model, hidden_fp[None, :])[0]
    return int(np.argmax(logits))


def fp16_ref_next_token(model, token_ids):
    """Next token via numpy_llama_ref.forward_full. NOTE: despite the name this is
    the INT8 recipe forward (forward_full uses int8 weights) -- so scoring against
    it measures device-vs-int8-recipe agreement, NOT vs the true model. Use
    --bf16-ref for the true HF bf16 reference."""
    logits = forward_full(model, np.array(token_ids))  # (M, V)
    return int(np.argmax(logits[-1]))


_BF16_REF = None


def bf16_ref_next_token(token_ids):
    """Ground-truth next token via the TRUE HF Llama 3.2 1B bf16 weights (loaded
    from the safetensors, full-precision forward). This is the real reference for
    'does the NPU match the actual model'. Lazily loads the 2.5 GB checkpoint."""
    global _BF16_REF
    if _BF16_REF is None:
        hf_weights = WEIGHTS_DIR / "model.safetensors"
        print(f"loading bf16 HF reference from {hf_weights} ...", flush=True)
        _BF16_REF = (RefModel(SafetensorsReader(hf_weights)), ref_forward_logits)
    ref_model, fwd = _BF16_REF
    logits = fwd(ref_model, np.array(token_ids))  # (M, V)
    return int(np.argmax(logits[-1]))


def device_setup(model, layers):
    """Compile the decode and load its weights, for `device_next_token`.

    Args:
        model: numpy_llama_ref's model (the lm_head and final norm).
        layers: The layers' weights and norms (gen_real_layer_mh).

    Returns:
        The session and the weight, lm_head and stage tensors.
    """
    return {
        "session": DispatchSession(
            build,
            None,
            P_MAX=P_MAX,
            N_LAYERS=N_LAYERS,
            T=T,
            KSET=ONESTREAM_KSET,
            ATTN_CORES=ATTN_CORES,
        ),
        "w": iron.tensor(pack_wblob(layers, rope_table(T, HEAD_D)), dtype=np.int8),
        "lmw": iron.tensor(
            pack_lmw(model.embed_i8, model.embed_sc, model.final_norm), dtype=np.int8
        ),
        "stage": pack_stage(
            layers, model.final_norm, np.zeros(D, np.int8), np.float32(0)
        ),
    }


def device_next_token(model, token_ids, dev):
    """The NPU decode's next token after `token_ids`.

    One dispatch a prompt token, from an empty KV cache: the host writes the
    token's embed seed into the stage's seed slot (the seed the device's
    sampler feeds back for its own tokens), and the device appends the
    token's KV row; its token after the last prompt token is the prediction.

    Args:
        model: numpy_llama_ref's model (its embed table).
        token_ids: The prompt's token ids.
        dev: `device_setup`'s.

    Returns:
        The predicted token id.

    Raises:
        ValueError: The prompt does not fit the KV cache.
    """
    if len(token_ids) > T:
        raise ValueError(f"{len(token_ids)} prompt tokens do not fit in T={T}")
    kv_t = iron.tensor(np.zeros(kv_words(T, N_LAYERS), np.int32), dtype=np.int32)
    stage_t = iron.tensor(dev["stage"], dtype=np.int8)
    out_t = iron.zeros([P_MAX * REC_BYTES], dtype=np.int8)
    for pos, tok in enumerate(token_ids):
        xin, sc = embed_seed_ref(model.embed_i8[tok], float(model.embed_sc[tok]))
        seed = np.zeros(ACT_BYTES, np.int8)
        seed[:D] = xin
        seed[D : D + 4] = np.frombuffer(np.float32(sc).tobytes(), np.int8)
        stage_t[SEED_AT : SEED_AT + ACT_BYTES] = seed
        dev["session"](
            [dev["w"], kv_t, dev["lmw"], out_t, stage_t],
            **dispatch_params(pos, 1, (0.0, 0, 0)),
        )
    out_t.to("cpu")
    return tokens_of(out_t.numpy(), 1)[0]


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
        "--bf16-ref",
        action="store_true",
        help="score against the TRUE HF Llama 3.2 1B bf16 weights (full-precision "
        "forward from the safetensors) instead of the int8 forward_full. This is "
        "the real 'NPU vs the actual model' agreement; without it the reference is "
        "the int8 recipe (device-vs-recipe).",
    )
    parser.add_argument(
        "--device",
        action="store_true",
        help="prefill and predict on the NPU (aie2_decode.py, one token a "
        "dispatch) instead of the numpy INT8 chain",
    )
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

    dev = None
    if opts.device:
        dev = device_setup(model, layers)
        print(f"device mode: N={N_LAYERS} decode on the NPU\n", flush=True)

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
        if opts.bf16_ref:
            ref_tok = bf16_ref_next_token(ids)
        else:
            ref_tok = fp16_ref_next_token(model, ids)
        t_fp16 = time.time() - t0
        t0 = time.time()
        if dev is not None:
            our_tok = device_next_token(model, ids, dev)
        else:
            our_tok = int8_chain_next_token(
                model,
                layers,
                cos_lut,
                sin_lut,
                ids,
                residual_fp32=opts.residual_fp32,
                residual_dyn=opts.residual_dyn,
                attn_lut=opts.attn_lut,
            )
        t_int8 = time.time() - t0
        match = ref_tok == our_tok
        n_total += 1
        n_match += int(match)
        results.append((prompt, ref_tok, our_tok, match))
        mark = "OK" if match else "FAIL"
        reflbl = "bf16" if opts.bf16_ref else "ref"
        devlbl = "npu" if opts.device else "int8"
        print(
            f"  [{mark}] {prompt!r:50} -> {reflbl}={enc.decode([ref_tok])!r} "
            f"{devlbl}={enc.decode([our_tok])!r}  "
            f"(t_ref={t_fp16:.1f}s t_dev={t_int8:.1f}s)",
            flush=True,
        )

    refname = "true HF bf16" if opts.bf16_ref else "int8 recipe"
    devname = "NPU" if opts.device else "numpy int8"
    print(
        f"\n{devname} vs {refname} top-1 agreement: {n_match}/{n_total} = "
        f"{100.0 * n_match / max(1, n_total):.1f}%"
    )
    return 0 if n_match == n_total else 1


if __name__ == "__main__":
    sys.exit(main())
