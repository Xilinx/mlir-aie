#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Prompt in, text out: Llama 3.2 1B on the NPU (aie2_decode.py), prefill and
decode both on the device, with time to first token and decode tokens/s.

Each of --repeats requests runs from the prompt's tokens: the time to first
token starts before tokenization and ends when the first generated token is
read back, split into the prefill dispatches' start-to-wait time and the
host's rest. The request with the median time to first token is reported,
with the fastest, and the median decode rate; every request must generate
the same tokens. The weights are already on the device, as on a server.
--prompt-tokens times prefill at a controlled length instead of --prompt's
real text: seeded random ids, not readable, but --check still scores them.

The prompt's tokens run token by token through the decode, each input from
its own embed row, which the device gathers; the token after the last one is
the first generated, and generation goes on from there.

Needs data/ (gen_llama_data.py) and tiktoken (`pip install tiktoken`).

Run (compiles the design, or reuses its JIT cache entry):
  python generate.py --prompt "The capital of France is" --max-new 32
  python generate.py --prompt-tokens 2048 --max-new 1   # TTFT at 2k tokens
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from llama_config import HEAD_D, N_LAYERS, VOCAB
from llama_decode import Decoder, DecodeShape
from llama_oracle import Layer, teacher_forced_logits
from numpy_llama_ref import forward_full, load_model, load_tokenizer, rope_table

BOS = 128000
STOP = (128001, 128009)


def main():
    p = argparse.ArgumentParser(description="Llama 3.2 1B on the NPU, prompt to text")
    p.add_argument("--prompt", help="text in")
    p.add_argument(
        "--prompt-tokens",
        type=int,
        help="instead of --prompt: this many seeded random token ids, for "
        "timing prefill at a controlled length (the output and --check's "
        "numpy_llama_ref agreement aren't readable text, but --check's oracle "
        "gate still holds)",
    )
    p.add_argument("--max-new", type=int, default=32, help="tokens to generate at most")
    p.add_argument("--t", type=int, default=128, help="KV cache rows (the build's T)")
    p.add_argument("--sample-temp", type=float, default=0.0, help="0: greedy")
    p.add_argument("--sample-topk", type=int, default=0, help="0: the whole set")
    p.add_argument("--sample-seed", type=int, default=0)
    p.add_argument("--data-dir", default=str(Path(__file__).parent / "data"))
    p.add_argument("--elf", help="a prebuilt full ELF of this build")
    p.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="requests timed; the one with the median time to first token is "
        "reported and checked",
    )
    p.add_argument(
        "--check",
        action="store_true",
        help="replay the device's tokens through the oracle of its arithmetic and "
        "fail unless each is its greedy pick; also report "
        "agreement with numpy_llama_ref's argmax (the INT8 recipe, not bit-exact)",
    )
    opts = p.parse_args()
    if bool(opts.prompt) == bool(opts.prompt_tokens):
        p.error("give exactly one of --prompt or --prompt-tokens")
    if opts.check and opts.sample_temp > 0:
        p.error("--check scores greedy decoding")
    if opts.repeats < 1:
        p.error("--repeats must be at least 1")
    data_dir = Path(opts.data_dir)

    enc = load_tokenizer(data_dir / "tokenizer.model")
    model = load_model(data_dir)
    layers = [Layer.real(L, data_dir, opts.t) for L in range(N_LAYERS)]
    decoder = Decoder(
        DecodeShape(N_LAYERS, opts.t),
        layers,
        (model.embed_i8, model.embed_sc, model.final_norm),
        rope_table(opts.t, HEAD_D),
        opts.elf,
    )
    sample = (opts.sample_temp, opts.sample_topk, opts.sample_seed)

    def tokenize():
        if opts.prompt_tokens:
            ids = np.random.default_rng(0).integers(0, VOCAB, opts.prompt_tokens)
            return [BOS] + ids.tolist()
        return [BOS] + enc.encode(opts.prompt)

    # Moves the weights to the device, as a server does before its first prompt.
    prompt = tokenize()
    decoder.generate(prompt, 2, sample)
    runs = []
    for _ in range(opts.repeats):
        started = time.perf_counter()
        prompt = tokenize()
        runs.append(decoder.generate(prompt, opts.max_new, sample, STOP, started))
    runs.sort(key=lambda g: g.ttft_s)
    gen = runs[len(runs) // 2]
    out = gen.tokens
    rates = [g.decode_tok_s for g in runs if g.decode_tok_s is not None]

    print(f"{opts.prompt}{enc.decode(out)}" if opts.prompt else enc.decode(out))
    print(
        f"prompt {len(prompt)} tokens, generated {len(out)}, median of "
        f"{len(runs)}: TTFT {gen.ttft_s * 1e3:.1f} ms (min "
        f"{runs[0].ttft_s * 1e3:.1f}; NPU dispatch "
        f"{gen.prefill_dispatch_s * 1e3:.1f}, host "
        f"{(gen.ttft_s - gen.prefill_dispatch_s) * 1e3:.1f}), "
        + (f"decode {np.median(rates):.2f} tok/s" if rates else "decode n/a")
    )
    differ = sum(g.tokens != out for g in runs)
    if differ:
        print(f"{differ}/{len(runs)} requests generated other tokens")
        return 1
    if not opts.check:
        return 0
    logits = forward_full(model, np.array(prompt + out[:-1]))
    ref = np.argmax(logits[len(prompt) - 1 :], axis=-1)
    agree = int(np.count_nonzero(ref == np.array(out)))
    print(f"check: {agree}/{len(out)} tokens are numpy_llama_ref's argmax")

    # The gate: the device's tokens through the oracle of its own arithmetic,
    # from empty caches, each generated token its greedy pick.
    tokens = prompt + out
    oracle = teacher_forced_logits(
        layers,
        (model.embed_i8, model.embed_sc, model.final_norm),
        rope_table(opts.t, HEAD_D),
        tokens[:-1],
    )
    fails = 0
    for pos in range(len(prompt) - 1, len(tokens) - 1):
        oracle_tok = int(np.argmax(oracle[pos]))
        if tokens[pos + 1] != oracle_tok:
            print(f"  position {pos + 1}: {tokens[pos + 1]}, oracle {oracle_tok}")
            fails += 1
    print(f"oracle: {len(out) - fails}/{len(out)} generated tokens pass")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
