#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""yolo26n's integer parameters: their container, where they live, and seeded synthetic ones.

Numpy only, so running the design and its weight-free tests needs neither
the checkpoint nor ONNX; gen_yolo_data.py extracts the real ones.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
from yolo_spec import (
    Add,
    Attention,
    Concat,
    Conv,
    DetectPost,
    InputQuant,
    MaxPool,
    Network,
    Slice,
    Upsample,
)

WEIGHTS_DIR = Path(os.environ.get("IRON_EXAMPLE_WEIGHTS_DIR", "/srv")) / "yolo26n"
DATA_DIR = Path(__file__).resolve().parent / "data"
INT8_VALUES = np.arange(-128, 128, dtype=np.int8)


class Params(dict):
    """`{"<op name>|<field>": array}`, saved as one `.npz`."""

    def put(self, op_name: str, field: str, value) -> None:
        self[f"{op_name}|{field}"] = np.asarray(value)

    def of(self, op_name: str, field: str) -> np.ndarray:
        return self[f"{op_name}|{field}"]

    def scale(self, tensor_name: str) -> int:
        return int(self[f"{tensor_name}|scale"])

    @classmethod
    def load(cls, path: Path) -> Params:
        with np.load(path) as data:
            return cls({key: data[key] for key in data.files})


def exp_table(score_scale: int) -> np.ndarray:
    """`round(2**26 * exp(-d * 2**score_scale))` for every int8 score gap `d` in 0..255.

    26 bits is the fewest for which the rounded probabilities match exact
    arithmetic on coco128; a row's total then needs 64 bits.
    """
    gaps = np.arange(256, dtype=np.float64)
    return np.round(2.0**26 * np.exp(-gaps * 2.0**score_scale)).astype(np.int32)


def anchor_grid(op: DetectPost) -> tuple[np.ndarray, np.ndarray]:
    """Anchor centres `(2, A)` in grid units and per-anchor strides `(A,)`, as Ultralytics makes them."""
    centres, strides = [], []
    for tensor, stride in zip(op.scores, op.strides):
        ys, xs = np.meshgrid(
            np.arange(tensor.height, dtype=np.float32) + np.float32(0.5),
            np.arange(tensor.width, dtype=np.float32) + np.float32(0.5),
            indexing="ij",
        )
        centres.append(np.stack([xs.reshape(-1), ys.reshape(-1)]))
        strides.append(np.full(tensor.height * tensor.width, stride, dtype=np.float32))
    return np.concatenate(centres, axis=1), np.concatenate(strides)


def synthetic_params(seed: int, path: Path) -> Path:
    """Write seed `seed`'s synthetic parameters to `path` unless there and current; return `path`.

    The file is current when newer than the sources that make it. It is
    written beside `path` and renamed into place, so runs sharing it never
    read a partial file.
    """
    sources = [Path(__file__), Path(__file__).with_name("yolo_spec.py")]
    if not path.exists() or path.stat().st_mtime < max(
        s.stat().st_mtime for s in sources
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_name(f"{path.stem}.{os.getpid()}.partial.npz")
        np.savez(partial, **Synthesizer(seed).run(Network.yolo26n()))
        partial.replace(path)
    return path


class Synthesizer:
    """Seeded parameters with the real shapes and power-of-two scales, for weight-free tests.

    Weights are drawn so each conv's output spreads over a few dozen steps
    rather than saturating; SiLU and sigmoid tables are numpy's float
    functions on the chosen scales.
    """

    # Concatenated tensors must share a scale, so every tensor at one resolution shares one.
    SCALES = {320: -2, 160: -3, 80: -4, 40: -4, 20: -4}

    def __init__(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)
        self.params = Params()

    def run(self, network: Network) -> Params:
        for op in network.ops:
            match op:
                case InputQuant():
                    pixels = np.arange(256, dtype=np.float32) / np.float32(255)
                    table = np.clip(np.round(pixels * np.float32(128)), -128, 127)
                    self.params.put(op.name, "table", table.astype(np.int8))
                    self.params.put(op.dst.name, "scale", -7)
                case Conv():
                    self.conv(op)
                case Add():
                    self.params.put(op.dst.name, "scale", self.SCALES[op.dst.width])
                case Concat() | MaxPool() | Upsample() | Slice():
                    srcs = op.srcs if isinstance(op, Concat) else (op.src,)
                    scale = self.params.scale(srcs[0].name)
                    if isinstance(op, Concat):
                        for src in srcs:
                            if self.params.scale(src.name) != scale:
                                raise ValueError(
                                    f"{op.name}: synthetic concat scales differ"
                                )
                    self.params.put(op.dst.name, "scale", scale)
                case Attention():
                    s = self.params.scale(op.qkv.name)
                    for field, value in {
                        "exp_table": exp_table(s + 1),
                        "q_mul": 91,
                        "q_mul_scale": -9,
                        "q_scale": s - 2,
                        "score_scale": s + 1,
                        "prob_scale": -7,
                    }.items():
                        self.params.put(op.name, field, value)
                    self.params.put(op.values.name, "scale", s)
                    self.params.put(op.dst.name, "scale", s)
                case DetectPost():
                    tables = []
                    for tensor in op.scores:
                        x = INT8_VALUES.astype(np.float32) * np.float32(
                            2.0 ** self.params.scale(tensor.name)
                        )
                        tables.append((1 / (1 + np.exp(-x))).astype(np.float32))
                    self.params.put(op.name, "sigmoid", np.stack(tables))
        return self.params

    def conv(self, op: Conv) -> None:
        sx, sw = self.params.scale(op.src.name), -7
        sy = self.SCALES[op.dst.width]
        shift = sy - sx - sw
        fan_in = op.src.channels // op.groups * op.kernel * op.kernel
        # Inputs spread ~30 wide; aim the sum at ~40 output steps.
        reach = int(
            np.clip(40 * 2.0**shift / (30 * math.sqrt(fan_in) / math.sqrt(3)), 1, 127)
        )
        weight = self.rng.integers(
            -reach,
            reach + 1,
            (op.dst.channels, op.src.channels // op.groups, op.kernel, op.kernel),
        )
        bias = self.rng.integers(-(1 << (shift + 5)), 1 << (shift + 5), op.dst.channels)
        self.params.put(op.name, "weight", weight.astype(np.int8))
        self.params.put(op.name, "bias", bias.astype(np.int64))
        self.params.put(op.name, "shift", shift)
        if op.act:
            x = INT8_VALUES.astype(np.float32) * np.float32(2.0**sy)
            silu = x / (1 + np.exp(-x))
            table = np.clip(np.round(silu / np.float32(2.0**sy)), -128, 127)
            self.params.put(op.name, "table", table.astype(np.int8))
        self.params.put(op.dst.name, "scale", sy)
