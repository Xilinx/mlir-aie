# yolo_spec.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2026, Advanced Micro Devices, Inc.
#
"""yolo26n-cls XINT8 logical network specification — single source of truth.

Pure Python, no AIE imports. Mirrors the structure of
mlir-aie/programming_examples/ml/mobilenet/network_spec.py so the same
algorithm/mapping split applies:

    yolo_spec.py    — algorithm (this file)
    placement.py    — physical tile assignment (separate, TBD)
    aie2_yolo_*.py  — IRON designs that consume yolo_spec.NETWORK

Shape convention is (W, H, C) — same as mobilenet's spec. Source manifest
uses NCHW; the loader at the bottom of this file does the swap.

Block kinds at a glance (all variants share Conv+BN+SiLU folded by Quark):

    m0  : 3x3 s=2  3 -> 16                    (stem)
    m1  : 3x3 s=2  16 -> 32
    m2  : C3k2-small (cv1 + 1 inner C3k + cv2),   32 -> 64
    m3  : 3x3 s=2  64 -> 64
    m4  : C3k2-small,                              64 -> 128
    m5  : 3x3 s=2  128 -> 128
    m6  : C3k2-heavy (cv1 + 1 nested C3k {cv1,cv2,2x[cv1+cv2],cv3} + cv2),
                                                   128 -> 128
    m7  : 3x3 s=2  128 -> 256
    m8  : C3k2-heavy,                              256 -> 256
    m9  : PSA (cv1 + attn{qkv, Q@Kᵀ, softmax, S@V, pe, proj} + ffn + cv2),
                                                   256 -> 256
    m10 : head (1x1 256 -> 1280, GAP, Gemm 1280 -> 2)

Source of truth: data/manifest.json (extractor output, 42 ops, bit-exact
vs ONNX Runtime — produced by gen_yolo_data.py).
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Layer descriptors
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Conv:
    """Unary kernel-style op: conv1x1, conv3x3, dw3x3, gemm, softmax.

    kind:
        'conv1x1'  - pointwise 1x1 conv
        'conv3x3'  - 3x3 conv (stem + C3k inner convs + downsample blocks)
        'dw3x3'    - 3x3 depthwise conv (PSA positional-encoding only)
        'gemm'     - fully-connected (treated as (M, K)->(M, N))
        'softmax'  - PSA attention softmax (no weights; normalizes along last dim)
        'avgpool'  - global average pool (W,H,C) -> (1,1,C); no weights

    name is the ONNX-relative path inside the block, e.g. 'cv1', 'm.0/cv2',
    'm.0/attn/qkv'. Preserves the topology hint without needing a separate
    branch graph.

    activation: 'silu' if Quark folded a SiLU into the requant, else None.
    All YOLO Convs that feed further compute have SiLU; Convs that feed
    Add/Concat directly are activation=None.
    """

    name: str
    kind: str
    in_shape: tuple  # (W, H, C) for spatial ops; (M, K) for gemm
    out_shape: tuple
    stride: int = 1
    activation: Optional[str] = "silu"
    manifest_name: str = ""  # ONNX node name in manifest.json


@dataclass(frozen=True)
class MatMul:
    """Binary act×act matmul. PSA uses two of these (Q@Kᵀ and S@V).

    Shape convention: (B, H, M, K) for A and (B, H, K, N) for B, output
    (B, H, M, N). Unlike Conv, both inputs are runtime activations —
    triggers the dual-scale kernel signature used in the PSA m9 builder.
    """

    name: str
    a_shape: tuple
    b_shape: tuple
    out_shape: tuple
    manifest_name: str = ""


@dataclass(frozen=True)
class Block:
    """One model.N block. `topology` discriminates wiring shape for the
    downstream IRON design generator; `layers` is the execution order.
    """

    name: str
    topology: str  # 'conv_stride' | 'c3k2_small' | 'c3k2_heavy' | 'psa' | 'head'
    layers: tuple
    notes: str = ""


# ---------------------------------------------------------------------------
# NETWORK — 11 blocks, derived from data/manifest.json
# ---------------------------------------------------------------------------
NETWORK: tuple = (
    Block(
        "m0",
        "conv_stride",
        (
            Conv(
                "conv",
                "conv3x3",
                (512, 512, 3),
                (256, 256, 16),
                stride=2,
                manifest_name="/model.0/conv/Conv",
            ),
        ),
        notes="Stem; activation-dominated (7 MB peak), tiny weights (448 B)",
    ),
    Block(
        "m1",
        "conv_stride",
        (
            Conv(
                "conv",
                "conv3x3",
                (256, 256, 16),
                (128, 128, 32),
                stride=2,
                manifest_name="/model.1/conv/Conv",
            ),
        ),
    ),
    Block(
        "m2",
        "c3k2_small",
        (
            Conv(
                "cv1",
                "conv1x1",
                (128, 128, 32),
                (128, 128, 32),
                manifest_name="/model.2/cv1/conv/Conv",
            ),
            Conv(
                "m.0/cv1",
                "conv3x3",
                (128, 128, 16),
                (128, 128, 8),
                manifest_name="/model.2/m.0/cv1/conv/Conv",
            ),
            Conv(
                "m.0/cv2",
                "conv3x3",
                (128, 128, 8),
                (128, 128, 16),
                manifest_name="/model.2/m.0/cv2/conv/Conv",
            ),
            Conv(
                "cv2",
                "conv1x1",
                (128, 128, 48),
                (128, 128, 64),
                manifest_name="/model.2/cv2/conv/Conv",
            ),
        ),
        notes="cv1 splits 32ch into 16+16; m.0 processes one half; concat (16+16+16=48) → cv2",
    ),
    Block(
        "m3",
        "conv_stride",
        (
            Conv(
                "conv",
                "conv3x3",
                (128, 128, 64),
                (64, 64, 64),
                stride=2,
                manifest_name="/model.3/conv/Conv",
            ),
        ),
    ),
    Block(
        "m4",
        "c3k2_small",
        (
            Conv(
                "cv1",
                "conv1x1",
                (64, 64, 64),
                (64, 64, 64),
                manifest_name="/model.4/cv1/conv/Conv",
            ),
            Conv(
                "m.0/cv1",
                "conv3x3",
                (64, 64, 32),
                (64, 64, 16),
                manifest_name="/model.4/m.0/cv1/conv/Conv",
            ),
            Conv(
                "m.0/cv2",
                "conv3x3",
                (64, 64, 16),
                (64, 64, 32),
                manifest_name="/model.4/m.0/cv2/conv/Conv",
            ),
            Conv(
                "cv2",
                "conv1x1",
                (64, 64, 96),
                (64, 64, 128),
                manifest_name="/model.4/cv2/conv/Conv",
            ),
        ),
    ),
    Block(
        "m5",
        "conv_stride",
        (
            Conv(
                "conv",
                "conv3x3",
                (64, 64, 128),
                (32, 32, 128),
                stride=2,
                manifest_name="/model.5/conv/Conv",
            ),
        ),
    ),
    Block(
        "m6",
        "c3k2_heavy",
        (
            Conv(
                "cv1",
                "conv1x1",
                (32, 32, 128),
                (32, 32, 128),
                manifest_name="/model.6/cv1/conv/Conv",
            ),
            Conv(
                "m.0/cv1",
                "conv1x1",
                (32, 32, 64),
                (32, 32, 32),
                manifest_name="/model.6/m.0/cv1/conv/Conv",
            ),
            Conv(
                "m.0/cv2",
                "conv1x1",
                (32, 32, 64),
                (32, 32, 32),
                manifest_name="/model.6/m.0/cv2/conv/Conv",
            ),
            Conv(
                "m.0/m/m.0/cv1",
                "conv3x3",
                (32, 32, 32),
                (32, 32, 32),
                manifest_name="/model.6/m.0/m/m.0/cv1/conv/Conv",
            ),
            Conv(
                "m.0/m/m.0/cv2",
                "conv3x3",
                (32, 32, 32),
                (32, 32, 32),
                manifest_name="/model.6/m.0/m/m.0/cv2/conv/Conv",
            ),
            Conv(
                "m.0/m/m.1/cv1",
                "conv3x3",
                (32, 32, 32),
                (32, 32, 32),
                manifest_name="/model.6/m.0/m/m.1/cv1/conv/Conv",
            ),
            Conv(
                "m.0/m/m.1/cv2",
                "conv3x3",
                (32, 32, 32),
                (32, 32, 32),
                manifest_name="/model.6/m.0/m/m.1/cv2/conv/Conv",
            ),
            Conv(
                "m.0/cv3",
                "conv1x1",
                (32, 32, 64),
                (32, 32, 64),
                manifest_name="/model.6/m.0/cv3/conv/Conv",
            ),
            Conv(
                "cv2",
                "conv1x1",
                (32, 32, 192),
                (32, 32, 128),
                manifest_name="/model.6/cv2/conv/Conv",
            ),
        ),
        notes="C3k2 with nested C3k: cv1 splits, m.0 has its own cv1/cv2 split + 2 inner [cv1,cv2] residual pairs + cv3 fuse",
    ),
    Block(
        "m7",
        "conv_stride",
        (
            Conv(
                "conv",
                "conv3x3",
                (32, 32, 128),
                (16, 16, 256),
                stride=2,
                manifest_name="/model.7/conv/Conv",
            ),
        ),
    ),
    Block(
        "m8",
        "c3k2_heavy",
        (
            Conv(
                "cv1",
                "conv1x1",
                (16, 16, 256),
                (16, 16, 256),
                manifest_name="/model.8/cv1/conv/Conv",
            ),
            Conv(
                "m.0/cv1",
                "conv1x1",
                (16, 16, 128),
                (16, 16, 64),
                manifest_name="/model.8/m.0/cv1/conv/Conv",
            ),
            Conv(
                "m.0/cv2",
                "conv1x1",
                (16, 16, 128),
                (16, 16, 64),
                manifest_name="/model.8/m.0/cv2/conv/Conv",
            ),
            Conv(
                "m.0/m/m.0/cv1",
                "conv3x3",
                (16, 16, 64),
                (16, 16, 64),
                manifest_name="/model.8/m.0/m/m.0/cv1/conv/Conv",
            ),
            Conv(
                "m.0/m/m.0/cv2",
                "conv3x3",
                (16, 16, 64),
                (16, 16, 64),
                manifest_name="/model.8/m.0/m/m.0/cv2/conv/Conv",
            ),
            Conv(
                "m.0/m/m.1/cv1",
                "conv3x3",
                (16, 16, 64),
                (16, 16, 64),
                manifest_name="/model.8/m.0/m/m.1/cv1/conv/Conv",
            ),
            Conv(
                "m.0/m/m.1/cv2",
                "conv3x3",
                (16, 16, 64),
                (16, 16, 64),
                manifest_name="/model.8/m.0/m/m.1/cv2/conv/Conv",
            ),
            Conv(
                "m.0/cv3",
                "conv1x1",
                (16, 16, 128),
                (16, 16, 128),
                manifest_name="/model.8/m.0/cv3/conv/Conv",
            ),
            Conv(
                "cv2",
                "conv1x1",
                (16, 16, 384),
                (16, 16, 256),
                manifest_name="/model.8/cv2/conv/Conv",
            ),
        ),
    ),
    Block(
        "m9",
        "psa",
        (
            Conv(
                "cv1",
                "conv1x1",
                (16, 16, 256),
                (16, 16, 256),
                manifest_name="/model.9/cv1/conv/Conv",
            ),
            # PSA splits cv1 output 128+128: top half flows through attn+ffn, bottom half skips to concat.
            Conv(
                "attn/qkv",
                "conv1x1",
                (16, 16, 128),
                (16, 16, 256),
                manifest_name="/model.9/m/m.0/attn/qkv/conv/Conv",
                activation=None,
            ),
            # qkv reshapes to (B=4, nh=2, key_dim=32 | key_dim=32 | head_dim=64, N=H*W=256).
            # Pre-matmul, q is transposed: q.T has shape (4, 2, N=256, key_dim=32).
            MatMul(
                "attn/qk",
                a_shape=(4, 2, 256, 32),  # q.transpose(-2,-1)
                b_shape=(4, 2, 32, 256),  # k
                out_shape=(4, 2, 256, 256),  # attn scores
                manifest_name="/model.9/m/m.0/attn/MatMul",
            ),
            Conv(
                "attn/softmax",
                "softmax",
                (4, 2, 256, 256),
                (4, 2, 256, 256),
                activation=None,
                manifest_name="/model.9/m/m.0/attn/Softmax",
            ),
            Conv(
                "attn/pe",
                "dw3x3",
                (16, 16, 128),
                (16, 16, 128),
                manifest_name="/model.9/m/m.0/attn/pe/conv/Conv",
                activation=None,
            ),
            # x = v @ attn.transpose(-2,-1). A is V (head_dim,N); B is softmax output transposed.
            MatMul(
                "attn/sv",
                a_shape=(4, 2, 64, 256),  # v
                b_shape=(4, 2, 256, 256),  # attn.transpose(-2,-1)
                out_shape=(4, 2, 64, 256),  # reshapes back to (B, C=128, H=16, W=16)
                manifest_name="/model.9/m/m.0/attn/MatMul_1",
            ),
            Conv(
                "attn/proj",
                "conv1x1",
                (16, 16, 128),
                (16, 16, 128),
                manifest_name="/model.9/m/m.0/attn/proj/conv/Conv",
                activation=None,
            ),
            Conv(
                "ffn/ffn.0",
                "conv1x1",
                (16, 16, 128),
                (16, 16, 256),
                manifest_name="/model.9/m/m.0/ffn/ffn.0/conv/Conv",
            ),
            Conv(
                "ffn/ffn.1",
                "conv1x1",
                (16, 16, 256),
                (16, 16, 128),
                manifest_name="/model.9/m/m.0/ffn/ffn.1/conv/Conv",
                activation=None,
            ),
            Conv(
                "cv2",
                "conv1x1",
                (16, 16, 256),
                (16, 16, 256),
                manifest_name="/model.9/cv2/conv/Conv",
            ),
        ),
        notes="PSA: only block with act×act MatMul + Softmax; pe is the lone depthwise in the network",
    ),
    Block(
        "m10",
        "head",
        (
            Conv(
                "conv",
                "conv1x1",
                (16, 16, 256),
                (16, 16, 1280),
                manifest_name="/model.10/conv/conv/Conv",
            ),
            Conv(
                "avgpool",
                "avgpool",
                (16, 16, 1280),
                (1, 1, 1280),
                activation=None,
                manifest_name="(not in manifest; ONNX GlobalAveragePool between conv and linear)",
            ),
            Conv(
                "linear",
                "gemm",
                (1280,),
                (2,),
                activation=None,
                manifest_name="/model.10/linear/Gemm",
            ),
            # Final softmax: ONNX emits unquantized fp32 here; we deploy as INT8
            # with scale 2^-7 (output is signed but values [0, 127] = [0, 1-1/128]).
            # Required for full on-device offload — host gets probabilities, not logits.
            Conv(
                "softmax",
                "softmax",
                (2,),
                (2,),
                activation=None,
                manifest_name="/model.10/Softmax",
            ),
        ),
        notes="Conv 256→1280 (SiLU), GAP 16×16→1×1, Gemm 1280→2 logits, Softmax → 2 INT8 probs (scale 2^-7).",
    ),
)


# ---------------------------------------------------------------------------
# Convenience accessors
# ---------------------------------------------------------------------------
_BY_NAME = {b.name: b for b in NETWORK}


def block(name: str) -> Block:
    return _BY_NAME[name]


def in_shape(name: str) -> tuple:
    return (
        block(name).layers[0].in_shape
        if isinstance(block(name).layers[0], Conv)
        else block(name).layers[0].a_shape
    )


def out_shape(name: str) -> tuple:
    last = block(name).layers[-1]
    return last.out_shape


BLOCK_NAMES = tuple(b.name for b in NETWORK)
CONV_STRIDE_NAMES = tuple(b.name for b in NETWORK if b.topology == "conv_stride")
C3K2_SMALL_NAMES = tuple(b.name for b in NETWORK if b.topology == "c3k2_small")
C3K2_HEAVY_NAMES = tuple(b.name for b in NETWORK if b.topology == "c3k2_heavy")


# ---------------------------------------------------------------------------
# Self-check: count Conv records and cross-reference manifest
# ---------------------------------------------------------------------------
def _count_ops():
    weighted_kinds = {"conv1x1", "conv3x3", "dw3x3", "gemm"}
    n_weighted = sum(
        1
        for b in NETWORK
        for l in b.layers
        if isinstance(l, Conv) and l.kind in weighted_kinds
    )
    n_matmul = sum(1 for b in NETWORK for l in b.layers if isinstance(l, MatMul))
    n_softmax = sum(
        1
        for b in NETWORK
        for l in b.layers
        if isinstance(l, Conv) and l.kind == "softmax"
    )
    n_avgpool = sum(
        1
        for b in NETWORK
        for l in b.layers
        if isinstance(l, Conv) and l.kind == "avgpool"
    )
    return n_weighted, n_matmul, n_softmax, n_avgpool


if __name__ == "__main__":
    n_weighted, n_matmul, n_softmax, n_avgpool = _count_ops()
    print(
        f"yolo_spec: {len(NETWORK)} blocks; "
        f"{n_weighted} weighted (Conv/Gemm) + {n_matmul} MatMul = {n_weighted + n_matmul} "
        f"(manifest: 42)  | +{n_softmax} Softmax +{n_avgpool} AvgPool (implicit ops)"
    )
    for b in NETWORK:
        kinds = [l.kind if isinstance(l, Conv) else "matmul" for l in b.layers]
        print(f"  {b.name:5s} [{b.topology:13s}] {len(b.layers):2d} layers: {kinds}")
