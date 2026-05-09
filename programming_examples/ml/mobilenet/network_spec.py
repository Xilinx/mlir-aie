#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2026, Advanced Micro Devices, Inc.
"""MobileNet V3 logical network specification — single source of truth.

Pure Python, no AIE imports. Describes the algorithm in one place so a reader
can grasp the network shape without reading any IRON / placement code:

    init    : 3x3 stride-2 conv     (224,224,8)  -> (112,112,16)
    bn0     : DW + 1x1-skip          (112,112,16) -> (112,112,16)
    bn1..9  : 1x1 -> DW -> 1x1 (+skip on 2,4,5,7,8,9)   ... -> (14,14,80)
    bn10..12: same shape, pipelined across tiles                -> (7,7,80)
    bn13,14 : same shape, cascade-split across 5 tiles          -> (7,7,80)
    post_l1 : 1x1 expand + avg-pool  (7,7,80)    -> (1,1,1280)   (padded)
    post_l2 : FC1 -> FC2             (1,1,1280)  -> (1,1,1280)

Shapes are (W, H, C). The IRON design and the numpy reference both consume
NETWORK; the per-block test generator builds standalone designs from it.

The PLACEMENT dict (which logical block runs on which physical tile) is
deliberately kept separate in aie2_mobilenet_iron.py — algorithm vs. mapping.
"""

from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Layer + Block descriptors
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Conv:
    """One convolution / pooling / FC layer.

    kind:
        'conv1x1'  - pointwise 1x1 conv
        'conv3x3'  - 3x3 conv (used by `init` only)
        'dw3x3'    - 3x3 depthwise conv (output channels = input channels)
        'avgpool'  - global average pool (W,H,C) -> (1,1,C); fused with the
                     following conv1x1 in post_l1
        'fc'       - fully-connected (treated as 1x1 conv on (1,1,C) input)

    sf_key: key into scale_factors_final.json for this block
            (e.g. 'conv1x1_1', 'conv3x3', 'conv1x1_2', 'FC1', 'FC2').
    """

    kind: str
    in_shape: tuple  # (W, H, C)
    out_shape: tuple
    stride: int = 1
    activation: Optional[str] = "relu"  # 'relu' | 'relu6' | None
    sf_key: str = ""
    # Kernel-internal expand width (out_shape's C reflects post-pad width).
    # Currently used only by post_l1's fused expand+avgpool: 960 -> padded 1280.
    expand_oc: Optional[int] = None


@dataclass(frozen=True)
class Block:
    """One bottleneck (or boundary block: init / post_l1 / post_l2)."""

    name: str
    layers: tuple
    skip: bool = False  # add residual from block input to last-layer output
    skip_sf_key: Optional[str] = None
    weight_files: tuple = field(default_factory=tuple)


# ---------------------------------------------------------------------------
# NETWORK — the canonical mobilenet v3 description
# ---------------------------------------------------------------------------
NETWORK: tuple = (
    # ---- Stem ----
    Block(
        "init",
        (
            Conv(
                "conv3x3",
                (224, 224, 8),
                (112, 112, 16),
                stride=2,
                activation="relu",
                sf_key="conv3x3",
            ),
        ),
        weight_files=("init_chain.txt",),
    ),
    # ---- Regular bottlenecks (bn0..bn9), single compute tile each ----
    Block(
        "bn0",
        (
            Conv("dw3x3", (112, 112, 16), (112, 112, 16), sf_key="conv3x3"),
            Conv(
                "conv1x1",
                (112, 112, 16),
                (112, 112, 16),
                activation=None,
                sf_key="conv1x1_2",
            ),
        ),
        skip=True,
        skip_sf_key="skip_add",
        weight_files=("bn0_chain.txt",),
    ),
    Block(
        "bn1",
        (
            Conv("conv1x1", (112, 112, 16), (112, 112, 64), sf_key="conv1x1_1"),
            Conv("dw3x3", (112, 112, 64), (56, 56, 64), stride=2, sf_key="conv3x3"),
            Conv(
                "conv1x1",
                (56, 56, 64),
                (56, 56, 24),
                activation=None,
                sf_key="conv1x1_2",
            ),
        ),
        weight_files=("bn1_chain.txt",),
    ),
    Block(
        "bn2",
        (
            Conv("conv1x1", (56, 56, 24), (56, 56, 72), sf_key="conv1x1_1"),
            Conv("dw3x3", (56, 56, 72), (56, 56, 72), sf_key="conv3x3"),
            Conv(
                "conv1x1",
                (56, 56, 72),
                (56, 56, 24),
                activation=None,
                sf_key="conv1x1_2",
            ),
        ),
        skip=True,
        skip_sf_key="skip_add",
        weight_files=("bn2_chain.txt",),
    ),
    Block(
        "bn3",
        (
            Conv("conv1x1", (56, 56, 24), (56, 56, 72), sf_key="conv1x1_1"),
            Conv("dw3x3", (56, 56, 72), (28, 28, 72), stride=2, sf_key="conv3x3"),
            Conv(
                "conv1x1",
                (28, 28, 72),
                (28, 28, 40),
                activation=None,
                sf_key="conv1x1_2",
            ),
        ),
        weight_files=("bn3_chain.txt",),
    ),
    Block(
        "bn4",
        (
            Conv("conv1x1", (28, 28, 40), (28, 28, 120), sf_key="conv1x1_1"),
            Conv("dw3x3", (28, 28, 120), (28, 28, 120), sf_key="conv3x3"),
            Conv(
                "conv1x1",
                (28, 28, 120),
                (28, 28, 40),
                activation=None,
                sf_key="conv1x1_2",
            ),
        ),
        skip=True,
        skip_sf_key="skip_add",
        weight_files=("bn4_chain.txt", "bn4_5_chain.txt"),  # bn4+bn5 share a chain
    ),
    Block(
        "bn5",
        (
            Conv("conv1x1", (28, 28, 40), (28, 28, 120), sf_key="conv1x1_1"),
            Conv("dw3x3", (28, 28, 120), (28, 28, 120), sf_key="conv3x3"),
            Conv(
                "conv1x1",
                (28, 28, 120),
                (28, 28, 40),
                activation=None,
                sf_key="conv1x1_2",
            ),
        ),
        skip=True,
        skip_sf_key="skip_add",
        weight_files=("bn5_chain.txt",),
    ),
    Block(
        "bn6",
        (
            Conv("conv1x1", (28, 28, 40), (28, 28, 240), sf_key="conv1x1_1"),
            Conv("dw3x3", (28, 28, 240), (14, 14, 240), stride=2, sf_key="conv3x3"),
            Conv(
                "conv1x1",
                (14, 14, 240),
                (14, 14, 80),
                activation=None,
                sf_key="conv1x1_2",
            ),
        ),
        weight_files=("bn6_chain.txt",),
    ),
    Block(
        "bn7",
        (
            Conv("conv1x1", (14, 14, 80), (14, 14, 200), sf_key="conv1x1_1"),
            Conv("dw3x3", (14, 14, 200), (14, 14, 200), sf_key="conv3x3"),
            Conv(
                "conv1x1",
                (14, 14, 200),
                (14, 14, 80),
                activation=None,
                sf_key="conv1x1_2",
            ),
        ),
        skip=True,
        skip_sf_key="skip_add",
        weight_files=("bn7_chain.txt",),
    ),
    Block(
        "bn8",
        (
            Conv("conv1x1", (14, 14, 80), (14, 14, 184), sf_key="conv1x1_1"),
            Conv("dw3x3", (14, 14, 184), (14, 14, 184), sf_key="conv3x3"),
            Conv(
                "conv1x1",
                (14, 14, 184),
                (14, 14, 80),
                activation=None,
                sf_key="conv1x1_2",
            ),
        ),
        skip=True,
        skip_sf_key="skip_add",
        weight_files=("bn8_chain.txt", "bn8_9_chain.txt"),  # bn8+bn9 share a chain
    ),
    Block(
        "bn9",
        (
            Conv("conv1x1", (14, 14, 80), (14, 14, 184), sf_key="conv1x1_1"),
            Conv("dw3x3", (14, 14, 184), (14, 14, 184), sf_key="conv3x3"),
            Conv(
                "conv1x1",
                (14, 14, 184),
                (14, 14, 80),
                activation=None,
                sf_key="conv1x1_2",
            ),
        ),
        skip=True,
        skip_sf_key="skip_add",
        weight_files=("bn9_chain.txt",),
    ),
    # ---- Pipeline bottlenecks (bn10..bn12), one tile per layer ----
    Block(
        "bn10",
        (
            Conv("conv1x1", (14, 14, 80), (14, 14, 480), sf_key="conv1x1_1"),
            Conv("dw3x3", (14, 14, 480), (14, 14, 480), sf_key="conv3x3"),
            Conv(
                "conv1x1",
                (14, 14, 480),
                (14, 14, 112),
                activation=None,
                sf_key="conv1x1_2",
            ),
        ),
        weight_files=("bn10_1_chain.txt", "bn10_2_chain.txt", "bn10_3_chain.txt"),
    ),
    Block(
        "bn11",
        (
            Conv("conv1x1", (14, 14, 112), (14, 14, 336), sf_key="conv1x1_1"),
            Conv("dw3x3", (14, 14, 336), (14, 14, 336), sf_key="conv3x3"),
            Conv(
                "conv1x1",
                (14, 14, 336),
                (14, 14, 112),
                activation=None,
                sf_key="conv1x1_2",
            ),
        ),
        skip=True,
        skip_sf_key="skip_add",
        weight_files=("bn11_1_chain.txt", "bn11_2_chain.txt", "bn11_3_chain.txt"),
    ),
    Block(
        "bn12",
        (
            Conv("conv1x1", (14, 14, 112), (14, 14, 336), sf_key="conv1x1_1"),
            Conv("dw3x3", (14, 14, 336), (7, 7, 336), stride=2, sf_key="conv3x3"),
            Conv(
                "conv1x1", (7, 7, 336), (7, 7, 80), activation=None, sf_key="conv1x1_2"
            ),
        ),
        # bn12 fuses L2+L3 into one combined weight buffer; the IRON design loads
        # bn12_2_3_chain.txt (preferred) or concats bn12_2 + bn12_3 as fallback.
        weight_files=("bn12_1_chain.txt", "bn12_2_3_chain.txt"),
    ),
    # ---- Cascade bottlenecks (bn13, bn14), 5 compute tiles each ----
    # Logically a normal 1x1 -> DW -> 1x1+skip block; physically each conv is
    # split across two cascade-connected compute tiles.
    Block(
        "bn13",
        (
            Conv("conv1x1", (7, 7, 80), (7, 7, 960), sf_key="conv1x1_1"),
            Conv("dw3x3", (7, 7, 960), (7, 7, 960), sf_key="conv3x3"),
            Conv(
                "conv1x1", (7, 7, 960), (7, 7, 80), activation=None, sf_key="conv1x1_2"
            ),
        ),
        skip=True,
        skip_sf_key="skip_add",
        weight_files=(
            "bn13_1_chain.txt",
            "bn13_2_chain.txt",
            "bn13_3_put_chain.txt",
            "bn13_3_get_chain.txt",
        ),
    ),
    Block(
        "bn14",
        (
            Conv("conv1x1", (7, 7, 80), (7, 7, 960), sf_key="conv1x1_1"),
            Conv("dw3x3", (7, 7, 960), (7, 7, 960), sf_key="conv3x3"),
            Conv(
                "conv1x1", (7, 7, 960), (7, 7, 80), activation=None, sf_key="conv1x1_2"
            ),
        ),
        skip=True,
        skip_sf_key="skip_add",
        weight_files=(
            "bn14_1_chain.txt",
            "bn14_2_chain.txt",
            "bn14_3_put_chain.txt",
            "bn14_3_get_chain.txt",
        ),
    ),
    # ---- Post-processing ----
    # post_l1: 1x1 expand to 960 fused with global average pool; output is then
    # padded to 1280 channels (next layer's input width). Spatial reduces to 1x1.
    Block(
        "post_l1",
        (
            Conv(
                "avgpool",
                (7, 7, 80),
                (1, 1, 1280),
                sf_key="conv1x1_1",
                expand_oc=960,
            ),
        ),
        weight_files=("post_conv_chain.txt",),
    ),
    # post_l2: two FC layers split across 4 compute tiles.
    Block(
        "post_l2",
        (
            Conv("fc", (1, 1, 1280), (1, 1, 1280), sf_key="FC1"),
            Conv("fc", (1, 1, 1280), (1, 1, 1280), activation=None, sf_key="FC2"),
        ),
        weight_files=(
            "FC1_0_chain.txt",
            "FC1_1_chain.txt",
            "FC1_2_chain.txt",
            "FC1_3_chain.txt",
            "FC2_0_chain.txt",
            "FC2_1_chain.txt",
            "FC2_2_chain.txt",
            "FC2_3_chain.txt",
        ),
    ),
)


# ---------------------------------------------------------------------------
# Convenience accessors — used by the IRON design and the numpy reference
# ---------------------------------------------------------------------------
_BY_NAME = {b.name: b for b in NETWORK}


def block(name: str) -> Block:
    """Look up a block by name (e.g. 'bn3', 'init', 'post_l2')."""
    return _BY_NAME[name]


def in_shape(name: str) -> tuple:
    """Block's input (W, H, C) — the input of its first layer."""
    return block(name).layers[0].in_shape


def out_shape(name: str) -> tuple:
    """Block's output (W, H, C) — the output of its last layer."""
    return block(name).layers[-1].out_shape


def dw_channels(name: str) -> int:
    """Width of the dw3x3 layer in this block (= expansion channels).

    Raises ValueError if the block has no dw3x3 layer.
    """
    for layer in block(name).layers:
        if layer.kind == "dw3x3":
            return layer.in_shape[2]
    raise ValueError(f"block {name!r} has no dw3x3 layer")


def stride(name: str) -> int:
    """Aggregate spatial stride across the block's layers."""
    s = 1
    for layer in block(name).layers:
        s *= layer.stride
    return s


# Convenience groupings used by the per-block test generator.
REGULAR_NAMES = tuple(f"bn{i}" for i in range(10))
PIPELINE_NAMES = ("bn10", "bn11", "bn12")
CASCADE_NAMES = ("bn13", "bn14")
