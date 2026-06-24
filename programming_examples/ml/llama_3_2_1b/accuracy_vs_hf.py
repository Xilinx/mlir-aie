"""Accuracy of our INT8 W8A8-dynamic recipe vs the TRUE HF Llama 3.2 1B weights.

The device decode is already bit-exact vs the numpy int8 forward (numpy_llama_ref).
The open question is upstream: does the int8 RECIPE track the real (bf16) model?
ablate_precision.py only compares vs a DEQUANTIZED-int8 ceiling (isolates dataflow
noise, not weight PTQ). Here the reference is the UNQUANTIZED bf16 safetensors
(/scratch/roesti/models/llama_3.2_1b/model.safetensors) loaded as fp32 -- no
torch/transformers needed (SafetensorsReader + numpy_llama_ref's fp32 ops).

We run both forwards on a real text sample (teacher-forced) and report:
  - next-token top-1 agreement (int8 argmax == bf16 argmax), per position
  - top-5 agreement (int8 argmax in bf16 top-5)
  - perplexity of each model on the text (lower=better; the GAP is the recipe cost)

Run:
  python accuracy_vs_hf.py [--text-file F | --prompt STR] [--max-tokens N]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from numpy_llama_ref import (
    load_model,
    forward_full,
    rmsnorm,
    rope_cos_sin,
    apply_rope,
    attention_full_gqa,
    silu,
    N_LAYERS,
    N_HEADS,
    N_KV_GROUPS,
    Q_DIM,
    KV_DIM,
    HEAD_DIM,
)
from gen_llama_data import SafetensorsReader
from generate import load_tokenizer

DATA_DIR = Path(__file__).parent / "data"
HF_WEIGHTS = Path("/scratch/roesti/models/llama_3.2_1b/model.safetensors")
TOKENIZER = DATA_DIR / "tokenizer.model"

DEFAULT_TEXT = (
    "Machine learning is a field of study that gives computers the ability to "
    "learn without being explicitly programmed. It has become one of the most "
    "important technologies of the modern era, powering applications from search "
    "engines to self-driving cars and medical diagnosis."
)


# --- True bf16 reference: load HF weights as fp32, full-precision forward ---


class RefModel:
    """Unquantized (bf16->fp32) HF weights for a full-precision numpy forward."""

    def __init__(self, reader: SafetensorsReader):
        self.embed = reader.load_f32("model.embed_tokens.weight")  # (V, D)
        self.final_norm = reader.load_f32("model.norm.weight")  # (D,)
        self.layers = []
        for L in range(N_LAYERS):
            p = f"model.layers.{L}."
            self.layers.append(
                {
                    "gamma_in": reader.load_f32(p + "input_layernorm.weight"),
                    "gamma_post": reader.load_f32(
                        p + "post_attention_layernorm.weight"
                    ),
                    "wq": reader.load_f32(p + "self_attn.q_proj.weight"),
                    "wk": reader.load_f32(p + "self_attn.k_proj.weight"),
                    "wv": reader.load_f32(p + "self_attn.v_proj.weight"),
                    "wo": reader.load_f32(p + "self_attn.o_proj.weight"),
                    "wg": reader.load_f32(p + "mlp.gate_proj.weight"),
                    "wu": reader.load_f32(p + "mlp.up_proj.weight"),
                    "wd": reader.load_f32(p + "mlp.down_proj.weight"),
                }
            )


def ref_forward_logits(model: RefModel, token_ids: np.ndarray) -> np.ndarray:
    """Full-precision (fp32-weight) prefill -> logits (M, V). Mirrors
    numpy_llama_ref.forward_full structure but with true bf16 weights and
    full-precision matmuls (no int8 anywhere)."""
    x = model.embed[token_ids].astype(np.float32)  # (M, D)
    M = x.shape[0]
    for L in range(N_LAYERS):
        W = model.layers[L]
        h = rmsnorm(x, W["gamma_in"])
        q = h @ W["wq"].T
        k = h @ W["wk"].T
        v = h @ W["wv"].T
        pos = np.arange(0, M)
        cos, sin = rope_cos_sin(pos, HEAD_DIM)
        q = apply_rope(q.reshape(M, N_HEADS, HEAD_DIM), cos, sin).reshape(M, Q_DIM)
        k = apply_rope(k.reshape(M, N_KV_GROUPS, HEAD_DIM), cos, sin).reshape(M, KV_DIM)
        a = attention_full_gqa(q, k, v)
        a = a @ W["wo"].T
        x = x + a
        h = rmsnorm(x, W["gamma_post"])
        g = h @ W["wg"].T
        u = h @ W["wu"].T
        s = silu(g) * u
        d = s @ W["wd"].T
        x = x + d
    h = rmsnorm(x, model.final_norm)
    return h @ model.embed.T  # tied lm_head, (M, V)


# --- metrics ---


def log_softmax(logits: np.ndarray) -> np.ndarray:
    m = logits.max(axis=-1, keepdims=True)
    z = logits - m
    return z - np.log(np.exp(z).sum(axis=-1, keepdims=True))


def teacher_forced_perplexity(logits: np.ndarray, token_ids: np.ndarray) -> float:
    """logits (M, V) predict the NEXT token; target[i] = token_ids[i+1]."""
    lp = log_softmax(logits[:-1].astype(np.float64))
    tgt = token_ids[1:]
    nll = -lp[np.arange(len(tgt)), tgt]
    return float(np.exp(nll.mean()))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prompt", type=str, default=None)
    p.add_argument("--text-file", type=Path, default=None)
    p.add_argument("--max-tokens", type=int, default=128)
    opts = p.parse_args()

    if opts.text_file:
        text = opts.text_file.read_text()
    elif opts.prompt:
        text = opts.prompt
    else:
        text = DEFAULT_TEXT

    enc = load_tokenizer(TOKENIZER)
    ids = [128000] + enc.encode(text)  # BOS + text
    ids = np.array(ids[: opts.max_tokens], dtype=np.int64)
    M = len(ids)
    print(f"text: {len(text)} chars -> {M} tokens (incl BOS)", flush=True)

    print(f"loading INT8 model from {DATA_DIR} ...", flush=True)
    int8_model = load_model(DATA_DIR)
    print(f"loading bf16 HF reference from {HF_WEIGHTS} ...", flush=True)
    ref_model = RefModel(SafetensorsReader(HF_WEIGHTS))

    print("running int8 forward ...", flush=True)
    int8_logits = forward_full(int8_model, ids)  # (M, V)
    print("running bf16 reference forward ...", flush=True)
    ref_logits = ref_forward_logits(ref_model, ids)  # (M, V)

    # Next-token agreement (positions 0..M-1 each predict token at +1; compare
    # the model argmaxes to each other -- "does int8 pick what bf16 picks").
    int8_arg = int8_logits.argmax(axis=-1)
    ref_arg = ref_logits.argmax(axis=-1)
    top1 = float((int8_arg == ref_arg).mean())
    # top-5: int8 argmax within bf16's top-5
    ref_top5 = np.argpartition(ref_logits, -5, axis=-1)[:, -5:]
    in_top5 = np.array([int8_arg[i] in ref_top5[i] for i in range(M)])
    top5 = float(in_top5.mean())

    ppl_int8 = teacher_forced_perplexity(int8_logits, ids)
    ppl_ref = teacher_forced_perplexity(ref_logits, ids)

    print("\n=== INT8 recipe vs bf16 HF Llama 3.2 1B ===")
    print(
        f"next-token top-1 agreement: {top1*100:.1f}%  ({int((int8_arg==ref_arg).sum())}/{M})"
    )
    print(f"int8-argmax in bf16 top-5 : {top5*100:.1f}%")
    print(
        f"perplexity  int8={ppl_int8:.3f}  bf16={ppl_ref:.3f}  "
        f"(ratio {ppl_int8/ppl_ref:.3f}x)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
