#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Top-1 next-token agreement of the decode with a reference, on a curated
factoid prompt set.

For each prompt, from an empty KV cache:
  - the reference's next token: numpy_llama_ref.forward_full's argmax (the
    INT8 recipe), or with --bf16-ref the bf16 model's (accuracy_vs_hf);
  - the decode's next token: the NPU's (aie2_decode.py, the prompt prefilled
    on the device), or with --path numpy the oracle's argmax
    (llama_oracle.teacher_forced_logits), which the NPU matches exactly.

Exits 1 when the agreement is below --min-agree. Tokenizing needs tiktoken
(`pip install tiktoken`).

Run:
  python bench_quality_mh.py --bf16-ref
  python bench_quality_mh.py --path numpy
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from accuracy_vs_hf import RefModel, ref_forward_logits
from gen_llama_data import WEIGHTS_DIR, SafetensorsReader
from llama_config import HEAD_D, N_LAYERS
from llama_decode import Decoder, DecodeShape
from llama_oracle import Layer, teacher_forced_logits
from numpy_llama_ref import forward_full, load_model, load_tokenizer, rope_table

# Each prompt has a "naturally correct" next token for a well-tuned Llama 3.2
# 1B; the reference picks it, so the list works for whatever checkpoint
# data/ holds.
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).parent / "data")
    parser.add_argument(
        "--prompts",
        type=int,
        nargs="+",
        choices=range(len(PROMPTS)),
        help="indices into PROMPTS (default: all)",
    )
    parser.add_argument(
        "--bf16-ref",
        action="store_true",
        help="score against the bf16 model (the safetensors at --hf-weights) "
        "instead of the INT8 recipe",
    )
    parser.add_argument(
        "--hf-weights",
        type=Path,
        default=WEIGHTS_DIR / "model.safetensors",
        help="the bf16 Llama 3.2 1B model.safetensors for --bf16-ref "
        "(default: $IRON_EXAMPLE_WEIGHTS_DIR/llama3.2-1b/)",
    )
    parser.add_argument(
        "--path",
        choices=("device", "numpy"),
        default="device",
        help="predict on the NPU (aie2_decode.py), or with its numpy oracle",
    )
    parser.add_argument("--t", type=int, default=128, help="KV cache rows")
    parser.add_argument(
        "--min-agree",
        type=float,
        default=0.75,
        help="the agreement below which the bench exits 1 (default: 0.75)",
    )
    opts = parser.parse_args()

    print(f"loading model from {opts.data_dir} ...", flush=True)
    model = load_model(opts.data_dir)
    enc = load_tokenizer(opts.data_dir / "tokenizer.model")
    lm = (model.embed_i8, model.embed_sc, model.final_norm)
    rope = rope_table(opts.t, HEAD_D)
    layers = [Layer.real(L, opts.data_dir, opts.t) for L in range(N_LAYERS)]
    ref = None
    if opts.bf16_ref:
        print(f"loading the bf16 model from {opts.hf_weights} ...", flush=True)
        ref = RefModel(SafetensorsReader(opts.hf_weights))
    decoder = None
    if opts.path == "device":
        decoder = Decoder(DecodeShape(N_LAYERS, opts.t), layers, lm, rope)

    n_match = 0
    indices = opts.prompts or range(len(PROMPTS))
    for idx in indices:
        ids = [128000] + enc.encode(PROMPTS[idx])
        t0 = time.time()
        if ref is not None:
            ref_logits = ref_forward_logits(ref, np.array(ids))
        else:
            ref_logits = forward_full(model, np.array(ids))
        ref_tok = int(np.argmax(ref_logits[-1]))
        t_ref = time.time() - t0
        t0 = time.time()
        if decoder is not None:
            our_tok = decoder.generate(ids, 1, (0.0, 0, 0))[0][0]
        else:
            fresh = [Layer(x.weights, x.gamma_in, x.gamma_post, opts.t) for x in layers]
            our_tok = int(np.argmax(teacher_forced_logits(fresh, lm, rope, ids)[-1]))
        t_dev = time.time() - t0
        n_match += ref_tok == our_tok
        print(
            f"  [{'OK' if ref_tok == our_tok else 'MISS'}] {PROMPTS[idx]!r:50} -> "
            f"{'bf16' if ref else 'recipe'}={enc.decode([ref_tok])!r} "
            f"{opts.path}={enc.decode([our_tok])!r}  "
            f"(t_ref={t_ref:.1f}s t_{opts.path}={t_dev:.1f}s)",
            flush=True,
        )

    agree = n_match / len(indices)
    print(
        f"\n{opts.path} vs {'bf16' if ref else 'the INT8 recipe'} top-1 agreement: "
        f"{n_match}/{len(indices)} = {100.0 * agree:.1f}%"
    )
    return 0 if agree >= opts.min_agree else 1


if __name__ == "__main__":
    sys.exit(main())
