"""Phase 6c.7: end-to-end generate.py wrapper around the N=16 chain xclbin.

Tokenize prompt -> embed last token -> NPU chain (single-head, real
weight slice) -> host runs final_norm + lm_head + argmax -> emit token
-> loop. No on-device KV append; each step runs the chain with random
KV cache (the kernel reads the cache as opaque bytes). Output won't
be meaningful text -- single-head attention with no real history can't
approximate full multi-head GQA Llama -- but it exercises every layer
of the host/NPU pipeline.

Run:
  python generate.py -x build/final_chain_dynscale_N16_T128.xclbin \\
                     -i build/insts_chain_dynscale_N16_T128.bin \\
                     -k MLIR_AIE \\
                     --prompt "The capital of France is" --new-tokens 4
"""

from __future__ import annotations

import base64
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

import aie.iron as iron
import aie.utils.test as test_utils
from aie.utils import DefaultNPURuntime

from aie2_chain_dynscale import (
    D,
    HD,
    HEAD_D,
    QD,
    KVD,
    T,
    N_TILE,
    N_LAYERS,
    WQ_SLOT,
    WO_SLOT,
    WG_SLOT,
    WU_SLOT,
    WD_SLOT,
    N_TILES_Q,
    N_TILES_O,
    N_TILES_G,
    N_TILES_U,
    N_TILES_D,
    OFF_GAMMA_IN,
    OFF_WQ,
    OFF_CS,
    OFF_WO,
    OFF_GAMMA_POST,
    OFF_WG,
    OFF_WU,
    OFF_WD,
    GAMMA_BYTES,
    CS_BYTES,
    KCACHE_BYTES,
    VCACHE_BYTES,
    KV_HEADER,
    PER_LAYER_KV,
    OFF_K,
    OFF_V,
    KV_BYTES,
    WEIGHTS_BYTES,
    ACT_SCALE,
    INV_ACT_SCALE,
)
from test_chain_dynscale import (
    gen_real_layer,
    numpy_layer_forward,
    requant,
)
from test_ffn_half import pack_perchan_slots, fp32_bytes
from test_flowkv import EXP_QUANT_SCALE
from gen_exp_lut import exp_lut
from gen_silu_lut import silu_lut
from aie2_chain_dynscale import SILU_GATE_SCALE
from numpy_llama_ref import load_model, lm_head_logits, VOCAB_SIZE


# --- Llama 3 tiktoken setup (matches test_quant_quality.py) ---
def load_tokenizer(tokenizer_path: Path):
    import tiktoken

    mergeable = {}
    with open(tokenizer_path) as f:
        for line in f:
            tok_b64, rank = line.strip().split(" ")
            mergeable[base64.b64decode(tok_b64)] = int(rank)
    PAT = (
        r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}"
        r"| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
    )
    SPECIAL = {
        "<|begin_of_text|>": 128000,
        "<|end_of_text|>": 128001,
        "<|eot_id|>": 128009,
    }
    return tiktoken.Encoding(
        name="llama3", pat_str=PAT, mergeable_ranks=mergeable, special_tokens=SPECIAL
    )


# --- Pack the wblob/kvblob for one dispatch (mirrors test_chain_dynscale) ---
def pack_blobs(layers):
    wblob = np.zeros(WEIGHTS_BYTES, dtype=np.int8)
    for L in range(N_LAYERS):
        off = OFF_GAMMA_IN + L * GAMMA_BYTES
        wblob[off : off + GAMMA_BYTES] = layers[L]["gamma_in"].view(np.int8)
        off = OFF_GAMMA_POST + L * GAMMA_BYTES
        wblob[off : off + GAMMA_BYTES] = layers[L]["gamma_post"].view(np.int8)
        off = OFF_CS + L * CS_BYTES
        cs_packed = np.concatenate([layers[L]["cos"], layers[L]["sin"]])
        wblob[off : off + CS_BYTES] = cs_packed.view(np.int8)

    def pack_slot(off, w_i8, w_sc, bias, n_tiles_per_layer, prefix_bytes):
        slot_bytes = (
            len(prefix_bytes) + N_TILE * w_i8.shape[1] + N_TILE * 4 + N_TILE * 4
        )
        per_layer_total = n_tiles_per_layer * slot_bytes
        packed = pack_perchan_slots(w_i8, w_sc, bias, N_TILE, prefix_bytes=prefix_bytes)
        assert packed.size == per_layer_total
        wblob[off : off + per_layer_total] = packed

    for L in range(N_LAYERS):
        sc = layers[L]["scales"]
        wq_prefix = (
            fp32_bytes(ACT_SCALE, sc["q_inv_out"], sc["q_out_scale"], 0.0)
            + b"\x00" * 48
        )
        pack_slot(
            OFF_WQ + L * N_TILES_Q * WQ_SLOT,
            layers[L]["wq_i8"],
            layers[L]["wq_sc"],
            layers[L]["bq"],
            N_TILES_Q,
            wq_prefix,
        )
        wo_prefix = fp32_bytes(sc["sv_out_scale"], INV_ACT_SCALE) + b"\x00" * 56
        pack_slot(
            OFF_WO + L * N_TILES_O * WO_SLOT,
            layers[L]["wo_i8"],
            layers[L]["wo_sc"],
            layers[L]["bo"],
            N_TILES_O,
            wo_prefix,
        )
        pack_slot(
            OFF_WG + L * N_TILES_G * WG_SLOT,
            layers[L]["wg_i8"],
            layers[L]["wg_sc"],
            layers[L]["bg"],
            N_TILES_G,
            b"",
        )
        wu_prefix = (
            fp32_bytes(
                ACT_SCALE,
                sc["up_inv_out"],
                sc["silu_up_scale"],
                sc["silu_inv_out_scale"],
            )
            + b"\x00" * 48
        )
        pack_slot(
            OFF_WU + L * N_TILES_U * WU_SLOT,
            layers[L]["wu_i8"],
            layers[L]["wu_sc"],
            layers[L]["bu"],
            N_TILES_U,
            wu_prefix,
        )
        wd_prefix = fp32_bytes(sc["down_act_scale"], INV_ACT_SCALE) + b"\x00" * 56
        pack_slot(
            OFF_WD + L * N_TILES_D * WD_SLOT,
            layers[L]["wd_i8"],
            layers[L]["wd_sc"],
            layers[L]["bd"],
            N_TILES_D,
            wd_prefix,
        )

    kvblob = np.zeros(KV_BYTES, dtype=np.int8)
    for L in range(N_LAYERS):
        sc = layers[L]["scales"]
        k_off = L * PER_LAYER_KV + OFF_K
        v_off = L * PER_LAYER_KV + OFF_V
        kvblob[k_off : k_off + KV_HEADER] = np.frombuffer(
            fp32_bytes(sc["k_scale"], 0.0), dtype=np.int8
        )
        kvblob[k_off + KV_HEADER : k_off + KV_HEADER + KCACHE_BYTES] = layers[L][
            "kcache"
        ]
        kvblob[v_off : v_off + KV_HEADER] = np.frombuffer(
            fp32_bytes(sc["v_scale"], sc["sv_inv_out_scale"]), dtype=np.int8
        )
        kvblob[v_off + KV_HEADER : v_off + KV_HEADER + VCACHE_BYTES] = layers[L][
            "vcache"
        ]
    return wblob, kvblob


def embed_token_to_int8(model, token_id: int) -> np.ndarray:
    """Embed table is INT8 + per-row fp32 scale. Dequant then requant
    to chain's activation scale (INV_ACT_SCALE)."""
    row_f32 = model.embed_i8[token_id].astype(np.float32) * model.embed_sc[token_id]
    return requant(row_f32, INV_ACT_SCALE)


def main():
    p = test_utils.create_default_argparser()
    p.add_argument("--prompt", type=str, default="The capital of France is")
    p.add_argument("--new-tokens", type=int, default=4)
    p.add_argument("--data-dir", type=str, default=str(Path(__file__).parent / "data"))
    p.add_argument(
        "--rng-seed",
        type=int,
        default=0,
        help="Seeds the random per-layer KV cache + cos/sin.",
    )
    opts = p.parse_args()

    data_dir = Path(opts.data_dir)
    npu_kernel = test_utils.create_npu_kernel(opts).npu_kernel

    print(f"loading model from {data_dir} ...", flush=True)
    model = load_model(data_dir)
    enc = load_tokenizer(data_dir / "tokenizer.model")

    # Build all 16 real-layer dicts once (weights are static).
    rng = np.random.default_rng(opts.rng_seed)
    lut_exp = exp_lut(EXP_QUANT_SCALE).astype(np.float32)
    lut_silu = silu_lut(SILU_GATE_SCALE)
    layers = []
    for L in range(N_LAYERS):
        layer = gen_real_layer(L, data_dir, rng)
        layer["lut_exp"] = lut_exp
        layer["lut_silu"] = lut_silu
        layers.append(layer)

    prompt_ids = [128000] + enc.encode(opts.prompt)
    tokens = list(prompt_ids)
    print(f"prompt: {opts.prompt!r} -> {len(prompt_ids)} tokens", flush=True)

    for step in range(opts.new_tokens):
        # Embed the last token to int8.
        x_in = embed_token_to_int8(model, tokens[-1])

        # Numpy chain forward for per-layer dynamic-scale calibration.
        x_after = x_in.copy()
        for L in range(N_LAYERS):
            x_after, scales = numpy_layer_forward(x_after, layers[L])
            layers[L]["scales"] = scales

        # Pack blobs + dispatch NPU.
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

        # Dequant -> fp32 -> final_norm + lm_head -> argmax.
        hidden_fp = hidden_i8.astype(np.float32) * ACT_SCALE
        logits = lm_head_logits(model, hidden_fp[None, :])[0]  # (V,)
        next_tok = int(np.argmax(logits))
        tokens.append(next_tok)
        decoded = enc.decode([next_tok])
        print(f"  step {step}: token={next_tok} {decoded!r}", flush=True)

    full = enc.decode(tokens[1:])  # drop BOS
    print(f"\nfinal: {full!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
