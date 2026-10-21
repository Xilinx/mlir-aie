#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""The decode on NPU2, against its numpy oracle and the bf16 model.

The default tier runs the decode test's cases at 2 layers on synthetic
weights. The extensive tier needs data/ and the bf16 safetensors: at 16
layers, on a small, a medium and a large prompt, the oracle's next-token
distribution must stay within a mean KL of the bf16 model's, and every token
the device generates must be the oracle's greedy pick.
"""

import re

import numpy as np
import pytest
import test_chain_persist_grow_mh as chain
from accuracy_vs_hf import RefModel, ref_forward_logits
from gen_llama_data import WEIGHTS_DIR, SafetensorsReader
from llama_config import HEAD_D, N_LAYERS
from llama_decode import Decoder, DecodeShape
from llama_oracle import Layer, greedy_verdict, teacher_forced_logits
from numpy_llama_ref import load_model, load_tokenizer, rope_table

SYNTHETIC = ["--synthetic-weights"]

CASES = {
    "empty-cache": (["--start-pos", "0"], 4, []),
    "carry": (
        ["--start-pos", "62", "--dispatches", "2", "--idle-s", "6"],
        8,
        ["carry across 2 dispatches: PASS"],
    ),
    "cache-end": (["--start-pos", "124"], 4, []),
    "split": (
        ["--start-pos", "62", "--passes", "3", "--split"],
        3,
        ["3 tokens vs 3 x 1: PASS"],
    ),
    "sampled": (
        ["--start-pos", "5", "--sample-temp", "0.7", "--sample-topk", "4"]
        + ["--sample-seed", "1234", "--dispatches", "2"],
        8,
        ["carry across 2 dispatches: PASS"],
    ),
    "prompt": (
        ["--start-pos", "60", "--prompt-tokens", "7", "--vs-batch", "1"],
        5,
        ["carry across 3 dispatches: PASS", "prompt in batches of 4 vs 1: PASS"],
    ),
}


@pytest.mark.parametrize("case", list(CASES))
def test_decode_matches_the_oracle(case, capsys):
    args, tokens, lines = CASES[case]
    assert chain.main(SYNTHETIC + args) == 0
    out = capsys.readouterr().out
    assert f"chain_persist_grow_mh: {tokens}/{tokens} tokens PASS" in out
    assert "KV cache: PASS" in out
    for line in lines:
        assert line in out
    if case == "sampled":
        states = re.findall(r"PASS .*\[sampled, state ([0-9a-f]{8})\]", out)
        assert len(states) == tokens
        assert len(set(states)) == tokens, "a draw reused the sampler's state"
        assert out.count("sampler state: PASS") == 2


DATA_DIR = chain.DATA_DIR
SAFETENSORS = WEIGHTS_DIR / "model.safetensors"
GENERATED = 8

# Lines of the King Lear opening in IRON's iron/lm/prompt.txt: 3, 7 and 12
# lines, 24, 60 and 104 tokens.
LEAR = (
    "SCENE I. King Lear's palace.\nEnter KENT, GLOUCESTER, and EDMUND\nKENT\n"
    "I thought the king had more affected the Duke of\nAlbany than Cornwall.\n"
    "GLOUCESTER\nIt did always seem so to us: but now, in the\n"
    "division of the kingdom, it appears not which of\n"
    "the dukes he values most; for equalities are so\n"
    "weighed, that curiosity in neither can make choice\n"
    "of either's moiety.\nKENT"
).split("\n")
# Measured mean KL 0.087 / 0.065 / 0.054 (the INT8 recipe's 0.041 / 0.030 /
# 0.023); the limits leave about 15%.
PROMPTS = {
    "small": ("\n".join(LEAR[:3]), 0.10),
    "medium": ("\n".join(LEAR[:7]), 0.075),
    "large": ("\n".join(LEAR[:12]), 0.065),
}


@pytest.fixture(scope="module")
def real_model():
    if not (DATA_DIR / "embed.i8.bin").exists() or not SAFETENSORS.exists():
        pytest.skip(
            f"needs {DATA_DIR} (gen_llama_data.py) and {SAFETENSORS} "
            "(IRON_EXAMPLE_WEIGHTS_DIR)"
        )
    model = load_model(DATA_DIR)
    lm = (model.embed_i8, model.embed_sc, model.final_norm)
    rope = rope_table(128, HEAD_D)
    layers = [Layer.real(L, DATA_DIR, 128) for L in range(N_LAYERS)]
    decoder = Decoder(DecodeShape(N_LAYERS, 128), layers, lm, rope)
    return {
        "enc": load_tokenizer(DATA_DIR / "tokenizer.model"),
        "lm": lm,
        "rope": rope,
        "decoder": decoder,
        "ref": RefModel(SafetensorsReader(SAFETENSORS)),
    }


@pytest.mark.extensive
@pytest.mark.parametrize("size", list(PROMPTS))
def test_real_weights_track_the_model_and_the_oracle(size, real_model):
    text, kl_limit = PROMPTS[size]
    prompt = [128000] + real_model["enc"].encode(text)
    out, _, _ = real_model["decoder"].generate(prompt, GENERATED, (0.0, 0, 0))
    layers = [Layer.real(L, DATA_DIR, 128) for L in range(N_LAYERS)]
    oracle = teacher_forced_logits(
        layers, real_model["lm"], real_model["rope"], prompt + out[:-1]
    )

    ref = ref_forward_logits(real_model["ref"], np.array(prompt))
    p = np.exp(ref - ref.max(axis=1, keepdims=True))
    p /= p.sum(axis=1, keepdims=True)
    q_log = oracle[: len(prompt)].astype(np.float64)
    q_log -= q_log.max(axis=1, keepdims=True)
    q_log -= np.log(np.exp(q_log).sum(axis=1, keepdims=True))
    kl = float(np.mean(np.sum(p * (np.log(np.maximum(p, 1e-300)) - q_log), axis=1)))
    print(f"{size}: {len(prompt)} tokens, mean KL(bf16 || oracle) = {kl:.4f}")

    tokens = prompt + out
    verdicts = [
        greedy_verdict(oracle[pos], tokens[pos + 1])
        for pos in range(len(prompt) - 1, len(tokens) - 1)
    ]
    assert all(ok for ok, _, _ in verdicts), verdicts
    assert kl <= kl_limit, f"{size}: mean KL {kl:.4f} over {kl_limit}"
