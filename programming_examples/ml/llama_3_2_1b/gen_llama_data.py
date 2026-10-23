#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""BF16 -> per-channel-INT8 weight extractor for Llama 3.2 1B.

Loads model.safetensors (146 BF16 tensors, 2.4 GB; from WEIGHTS_DIR, or pass
--weights) and dumps INT8 weights + fp32 per-output-channel
scales into data/ as flat .bin files our NPU runtime can pack.

Per-channel symmetric absmax quant for every (out, in) linear weight:
  scale[n] = max(|W[n, :]|) / 127     # fp32, one per output channel
  w_i8[n,k] = round(W[n,k] / scale[n]).clip(-127, 127)

The same per-row INT8 for embed_tokens (a (V, D) linear serving both the
embedding lookup and the lm_head; Llama 3.2 1B has tied embeddings).

Layernorm gamma vectors stay as bf16 (tiny, no quant benefit).

Output layout:
  data/embed.i8.bin              (V*D int8, row-major (V, D))
  data/embed.scales.f32.bin      (V fp32; one scale per vocab entry)
  data/final_norm.bf16.bin       (D bf16)
  data/layer_{L:02d}/gamma_in.bf16.bin           (D bf16)
                    /gamma_post.bf16.bin         (D bf16)
                    /wq.i8.bin     wq.scales.f32.bin   (Q_DIM*D, Q_DIM)
                    /wk.i8.bin     wk.scales.f32.bin   (KV_DIM*D, KV_DIM)
                    /wv.i8.bin     wv.scales.f32.bin   (KV_DIM*D, KV_DIM)
                    /wo.i8.bin     wo.scales.f32.bin   (D*Q_DIM, D)
                    /wg.i8.bin     wg.scales.f32.bin   (HD*D, HD)
                    /wu.i8.bin     wu.scales.f32.bin   (HD*D, HD)
                    /wd.i8.bin     wd.scales.f32.bin   (D*HD, D)

A tokenizer file is copied into data/tokenizer.model so
numpy_llama_ref.load_tokenizer can read it without juggling another path.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import struct
import sys
from pathlib import Path

import ml_dtypes
import numpy as np
from llama_config import (
    HD,
    KV_DIM,
    N_LAYERS,
    Q_DIM,
    VOCAB,
    D,
)

# The gated checkpoint's directory, as amd/IRON's examples find it
# (iron/lm/testing.py): model.safetensors and tokenizer.model.
WEIGHTS_DIR = Path(os.environ.get("IRON_EXAMPLE_WEIGHTS_DIR", "/srv")) / "llama3.2-1b"


_ST_DTYPE = {
    "BF16": ml_dtypes.bfloat16,
    "F16": np.float16,
    "F32": np.float32,
    "I8": np.int8,
    "U8": np.uint8,
}


class SafetensorsReader:
    """Manual safetensors reader — numpy backend can't load bf16 (numpy
    has no native bf16 dtype). Reinterpret raw bytes via ml_dtypes."""

    def __init__(self, path: Path):
        self.path = Path(path)
        with open(self.path, "rb") as fh:
            hlen = struct.unpack("<Q", fh.read(8))[0]
            self.header = json.loads(fh.read(hlen))
        self.data_start = 8 + hlen

    def keys(self) -> list[str]:
        return [k for k in self.header if k != "__metadata__"]

    def load_f32(self, name: str) -> np.ndarray:
        meta = self.header[name]
        dtype = _ST_DTYPE[meta["dtype"]]
        s, e = meta["data_offsets"]
        with open(self.path, "rb") as fh:
            fh.seek(self.data_start + s)
            raw = fh.read(e - s)
        arr = np.frombuffer(raw, dtype=dtype).reshape(meta["shape"])
        return arr.astype(np.float32)

    def load_bf16(self, name: str) -> np.ndarray:
        meta = self.header[name]
        assert meta["dtype"] == "BF16", f"{name} is {meta['dtype']}, want BF16"
        s, e = meta["data_offsets"]
        with open(self.path, "rb") as fh:
            fh.seek(self.data_start + s)
            raw = fh.read(e - s)
        return (
            np.frombuffer(raw, dtype=ml_dtypes.bfloat16).reshape(meta["shape"]).copy()
        )


def quant_int8_perchan_absmax(w_f32: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric per-output-channel INT8 quantization.

    HF Linear weight shape is (out_dim, in_dim). One scale per output row.
    Returns (q_i8, scale_f32) where scale_f32 has shape (out_dim,) and
    q_i8 has the same shape as the input.
    """
    assert w_f32.ndim == 2, f"expected 2D weight, got {w_f32.shape}"
    absmax = np.maximum(np.abs(w_f32).max(axis=1, keepdims=True), 1e-12)
    scale = absmax / 127.0  # (out_dim, 1)
    q = np.round(w_f32 / scale).clip(-127, 127).astype(np.int8)
    return q, scale.squeeze(1).astype(np.float32)


def dump_quantized(
    out_dir: Path,
    name: str,
    w_f32: np.ndarray,
    expected_shape: tuple[int, int] | None = None,
) -> None:
    """Quantize and dump (i8 weights, fp32 scales) to out_dir/name.{i8,scales.f32}.bin."""
    if expected_shape is not None and w_f32.shape != expected_shape:
        raise ValueError(f"{name}: got {w_f32.shape}, expected {expected_shape}")
    q, scales = quant_int8_perchan_absmax(w_f32)
    (out_dir / f"{name}.i8.bin").write_bytes(q.tobytes())
    (out_dir / f"{name}.scales.f32.bin").write_bytes(scales.tobytes())


def dump_bf16(
    out_dir: Path, name: str, w_bf16: np.ndarray, expected_shape: tuple | None = None
) -> None:
    if expected_shape is not None and w_bf16.shape != expected_shape:
        raise ValueError(f"{name}: got {w_bf16.shape}, expected {expected_shape}")
    (out_dir / f"{name}.bf16.bin").write_bytes(w_bf16.tobytes())


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--weights",
        type=Path,
        default=WEIGHTS_DIR / "model.safetensors",
        help="path to the bf16 Llama 3.2 1B model.safetensors "
        "(default: $IRON_EXAMPLE_WEIGHTS_DIR/llama3.2-1b/, $IRON_EXAMPLE_WEIGHTS_DIR "
        "defaulting to /srv).",
    )
    p.add_argument(
        "--tokenizer",
        type=Path,
        default=WEIGHTS_DIR / "tokenizer.model",
        help="path to the Llama 3.2 1B tokenizer.model (default: beside the "
        "weights).",
    )
    p.add_argument("--out-dir", type=Path, default=Path(__file__).parent / "data")
    args = p.parse_args(sys.argv[1:])

    for path in (args.weights, args.tokenizer):
        if not path.exists():
            p.error(f"{path} is missing")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    reader = SafetensorsReader(args.weights)
    print(
        f"loaded safetensors header: {len(reader.keys())} tensors from {args.weights}"
    )

    # --- Embedding (also serves as lm_head; tied in Llama 3.2 1B) ---
    embed_f32 = reader.load_f32("model.embed_tokens.weight")
    print(f"  embed_tokens: shape={embed_f32.shape} dtype=fp32")
    dump_quantized(args.out_dir, "embed", embed_f32, expected_shape=(VOCAB, D))

    # --- Final RMSNorm gamma (bf16, no quant) ---
    final_norm_bf16 = reader.load_bf16("model.norm.weight")
    dump_bf16(args.out_dir, "final_norm", final_norm_bf16, expected_shape=(D,))

    # --- Per-layer weights ---
    for L in range(N_LAYERS):
        layer_dir = args.out_dir / f"layer_{L:02d}"
        layer_dir.mkdir(parents=True, exist_ok=True)
        prefix = f"model.layers.{L}."

        gamma_in = reader.load_bf16(prefix + "input_layernorm.weight")
        gamma_post = reader.load_bf16(prefix + "post_attention_layernorm.weight")
        dump_bf16(layer_dir, "gamma_in", gamma_in, expected_shape=(D,))
        dump_bf16(layer_dir, "gamma_post", gamma_post, expected_shape=(D,))

        # Attention projections
        wq = reader.load_f32(prefix + "self_attn.q_proj.weight")
        wk = reader.load_f32(prefix + "self_attn.k_proj.weight")
        wv = reader.load_f32(prefix + "self_attn.v_proj.weight")
        wo = reader.load_f32(prefix + "self_attn.o_proj.weight")
        dump_quantized(layer_dir, "wq", wq, expected_shape=(Q_DIM, D))
        dump_quantized(layer_dir, "wk", wk, expected_shape=(KV_DIM, D))
        dump_quantized(layer_dir, "wv", wv, expected_shape=(KV_DIM, D))
        dump_quantized(layer_dir, "wo", wo, expected_shape=(D, Q_DIM))

        # FFN projections (SwiGLU)
        wg = reader.load_f32(prefix + "mlp.gate_proj.weight")
        wu = reader.load_f32(prefix + "mlp.up_proj.weight")
        wd = reader.load_f32(prefix + "mlp.down_proj.weight")
        dump_quantized(layer_dir, "wg", wg, expected_shape=(HD, D))
        dump_quantized(layer_dir, "wu", wu, expected_shape=(HD, D))
        dump_quantized(layer_dir, "wd", wd, expected_shape=(D, HD))

        if L == 0 or L == N_LAYERS - 1:
            for k in ("wq", "wk", "wv", "wo", "wg", "wu", "wd"):
                f = layer_dir / f"{k}.i8.bin"
                print(f"  layer {L:02d} {k}: {f.stat().st_size} bytes")

    # --- Tokenizer (copy in for convenience) ---
    shutil.copy(args.tokenizer, args.out_dir / args.tokenizer.name)
    print(f"  copied tokenizer: {args.out_dir / args.tokenizer.name}")

    print(f"done. output: {args.out_dir}/")


if __name__ == "__main__":
    main()
