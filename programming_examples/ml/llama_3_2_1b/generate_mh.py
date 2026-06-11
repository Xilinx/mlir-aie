"""Phase 7a end-to-end: real-weight host-decode loop around the single-
layer multi-head GQA xclbin. Mirrors generate.py but uses ONE layer (the
mh xclbin handles only one layer per dispatch) with full Q_DIM=2048/
KV_DIM=512 weights. Output will not be coherent text (single layer +
random KV cache) but exercises every piece of the multi-head pipeline
through tokenize -> embed -> NPU -> final_norm + lm_head -> argmax.

Run:
  python generate_mh.py -x build/final_layer_mh_T128.xclbin \\
                        -i build/insts_layer_mh_T128.bin \\
                        -k MLIR_AIE \\
                        --prompt "The capital of France is" --new-tokens 4 \\
                        --layer 0
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import aie.iron as iron
import aie.utils.test as test_utils
from aie.utils import DefaultNPURuntime

from aie2_layer_mh import D, ACT_SCALE, INV_ACT_SCALE, SILU_GATE_SCALE
from numpy_layer_mh import gen_real_layer_mh, numpy_layer_mh_forward, requant
from test_layer_mh import pack_blobs
from test_flowkv import EXP_QUANT_SCALE
from gen_exp_lut import exp_lut
from gen_silu_lut import silu_lut
from numpy_llama_ref import load_model, lm_head_logits
from generate import load_tokenizer


def embed_token_to_int8(model, token_id: int) -> np.ndarray:
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
    p.add_argument(
        "--layer",
        type=int,
        default=0,
        help="Which Llama layer's weights to load (mh xclbin "
        "runs ONE layer per dispatch).",
    )
    opts = p.parse_args()

    data_dir = Path(opts.data_dir)
    npu_kernel = test_utils.create_npu_kernel(opts).npu_kernel

    print(f"loading model from {data_dir} ...", flush=True)
    model = load_model(data_dir)
    enc = load_tokenizer(data_dir / "tokenizer.model")

    rng = np.random.default_rng(opts.rng_seed)
    layer = gen_real_layer_mh(opts.layer, data_dir, rng)
    layer["lut_exp"] = exp_lut(EXP_QUANT_SCALE).astype(np.float32)
    layer["lut_silu"] = silu_lut(SILU_GATE_SCALE)

    prompt_ids = [128000] + enc.encode(opts.prompt)
    tokens = list(prompt_ids)
    print(f"prompt: {opts.prompt!r} -> {len(prompt_ids)} tokens", flush=True)

    for step in range(opts.new_tokens):
        x_in = embed_token_to_int8(model, tokens[-1])

        # Calibrate per-Q-head dynamic scales via numpy ref.
        _, scales = numpy_layer_mh_forward(x_in, layer)
        layer["scales"] = scales

        wblob, kvblob = pack_blobs(layer)
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
        print(f"  step {step}: token={next_tok} {decoded!r}", flush=True)

    full = enc.decode(tokens[1:])
    print(f"\nfinal: {full!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
