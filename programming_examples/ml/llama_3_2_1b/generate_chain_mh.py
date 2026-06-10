"""Phase 8a end-to-end: real-weight + real-K/V host-decode loop around
the N=16 multi-head GQA chain xclbin. Per token:
  - host-numpy: compute k_new/v_new from real wk/wv per layer, apply
    rope_k at the current position, append to per-layer K/V cache
  - host-numpy: per-layer dynamic-scale calibration (existing)
  - pack wblob + kvblob (cs reflects current position)
  - NPU dispatch: 16 mh layers
  - lm_head + argmax -> next token

The xclbin (aie2_chain_dynscale_mh.py) is unchanged. Prefill processes
the prompt one token at a time via NUMPY ONLY (no NPU dispatch needed;
we only care about building the K/V cache for those positions). Then
the decode loop dispatches NPU per generated token.

Note: chain_mh's flowkv_mh attends to ALL T=128 cache positions. Slots
past the current position contain zeros; the attention math is well-
defined but not the textbook "attend over current positions only".
Real causal masking comes in Phase 8c. Expect outputs to be more
sensible than Phase 7b's random-K/V garbage but not fully coherent
text yet.

Run:
  make chain_mh CHAIN_MH_N=16
  python generate_chain_mh.py -x build/final_chain_mh_N16_T128.xclbin \\
                              -i build/insts_chain_mh_N16_T128.bin \\
                              -k MLIR_AIE \\
                              --prompt "The capital of France is" --new-tokens 4
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

import aie.iron as iron
import aie.utils.test as test_utils
from aie.utils import DefaultNPURuntime

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
from test_chain_mh import pack_blobs
from test_flowkv import EXP_QUANT_SCALE
from gen_exp_lut import exp_lut
from gen_silu_lut import silu_lut
from numpy_llama_ref import load_model, lm_head_logits, rope_cos_sin
from generate import load_tokenizer


def embed_token_to_int8(model, token_id: int) -> np.ndarray:
    row_f32 = model.embed_i8[token_id].astype(np.float32) * model.embed_sc[token_id]
    return requant(row_f32, INV_ACT_SCALE)


def update_layer_cos_sin(layers, cos_lut, sin_lut, pos):
    """Set per-layer cos/sin for the current decode position. All layers
    share the same (cos, sin) for a given position — Q rope and K rope
    both use this token's position vector."""
    c = cos_lut[pos].astype(bfloat16)
    s = sin_lut[pos].astype(bfloat16)
    for layer in layers:
        layer["cos"] = c
        layer["sin"] = s


def numpy_forward_all_layers(x, layers, pos):
    """Numpy forward through all N_LAYERS, with K/V append at `pos`.
    Sets layer["scales"] per layer (used by pack_blobs) and per-layer
    ["t_used"] = pos + 1 so the device's flowkv_mh only attends over
    valid cache positions (Phase 8c causal mask)."""
    x_cur = x.copy()
    for L in range(N_LAYERS):
        x_cur, scales = numpy_layer_mh_forward(x_cur, layers[L], position=pos)
        layers[L]["scales"] = scales
        layers[L]["t_used"] = pos + 1
    return x_cur


def main():
    p = test_utils.create_default_argparser()
    p.add_argument("--prompt", type=str, default="The capital of France is")
    p.add_argument("--new-tokens", type=int, default=4)
    p.add_argument("--data-dir", type=str, default=str(Path(__file__).parent / "data"))
    p.add_argument(
        "--rng-seed",
        type=int,
        default=0,
        help="Seeds the empty-cache fixture randomness (gammas, "
        "cos/sin are overwritten per token).",
    )
    opts = p.parse_args()

    data_dir = Path(opts.data_dir)
    npu_kernel = test_utils.create_npu_kernel(opts).npu_kernel

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
    print(f"loaded {N_LAYERS} mh layers (real wk/wv; empty K/V caches)", flush=True)

    # Pre-compute rope (cos, sin) for every position in the cache window.
    cos_lut, sin_lut = rope_cos_sin(np.arange(T), HEAD_D)  # (T, HEAD_D) fp32

    prompt_ids = [128000] + enc.encode(opts.prompt)
    tokens = list(prompt_ids)
    print(f"prompt: {opts.prompt!r} -> {len(prompt_ids)} tokens", flush=True)
    if len(prompt_ids) + opts.new_tokens > T:
        print(
            f"WARNING: prompt+new_tokens ({len(prompt_ids)+opts.new_tokens}) "
            f"exceeds cache size T={T}; later positions will overwrite earlier.",
            file=sys.stderr,
        )

    # --- Prompt prefill (numpy only, no NPU dispatch) ----------------------
    # Build per-layer K/V cache up to position len(prompt_ids)-2. The last
    # prompt token's K/V is appended during the first decode iteration.
    for pos in range(len(prompt_ids) - 1):
        x_in = embed_token_to_int8(model, tokens[pos])
        update_layer_cos_sin(layers, cos_lut, sin_lut, pos)
        _ = numpy_forward_all_layers(x_in, layers, pos)
    print(
        f"prefill done: K/V cache populated for positions " f"0..{len(prompt_ids) - 2}",
        flush=True,
    )

    # --- Decode loop -------------------------------------------------------
    for step in range(opts.new_tokens):
        pos = len(tokens) - 1  # position we're now processing
        x_in = embed_token_to_int8(model, tokens[pos])
        update_layer_cos_sin(layers, cos_lut, sin_lut, pos)

        # Numpy forward (calibrates dynamic scales AND appends this pos's K/V).
        _ = numpy_forward_all_layers(x_in, layers, pos)

        # Pack + dispatch.
        wblob, kvblob = pack_blobs(layers)
        x_t = iron.tensor(x_in, dtype=np.int8)
        w_t = iron.tensor(wblob, dtype=np.int8)
        kv_t = iron.tensor(kvblob, dtype=np.int8)
        o_t = iron.zeros([D], dtype=np.int8)
        rc = DefaultNPURuntime.run_test(
            npu_kernel,
            [x_t, w_t, kv_t, o_t],
            {},
            verify=False,
            verbosity=opts.verbosity,
        )
        if rc != 0:
            print(f"NPU dispatch returned {rc}", file=sys.stderr)
            return rc
        o_t.to("cpu")
        hidden_i8 = o_t.numpy()

        hidden_fp = hidden_i8.astype(np.float32) * ACT_SCALE
        logits = lm_head_logits(model, hidden_fp[None, :])[0]
        next_tok = int(np.argmax(logits))
        tokens.append(next_tok)
        decoded = enc.decode([next_tok])
        print(f"  step {step}: pos={pos} token={next_tok} {decoded!r}", flush=True)

    full = enc.decode(tokens[1:])
    print(f"\nfinal: {full!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
