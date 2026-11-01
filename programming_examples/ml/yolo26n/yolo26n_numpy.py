#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Integer numpy reference for yolo26n: the oracle every NPU run is checked against.

`Oracle(params).run(image)` evaluates `yolo_spec.Network.yolo26n()` on one
letterboxed uint8 RGB image (640, 640, 3) and returns every tensor by name:
int8 activations as `(C, H, W)` int64 arrays and the detections as float32
`(300, 6)` rows of x1, y1, x2, y2, score, class.

Requantization rounds half to even and saturates to int8, as ONNX
QuantizeLinear does; with power-of-two scales that is exact integer
arithmetic. The attention softmax is integer too: a Q26 exp table and an
exactly rounded division. Where it and ONNX Runtime's float32 softmax
disagree, `SoftmaxArbiter` computes the probability exactly to say which is
off (verify_oracle.py). The
postprocess runs in float32 on dequantized int8 values, as the ONNX graph
does, and is exact.
"""

from __future__ import annotations

import math
from decimal import Decimal, localcontext
from fractions import Fraction

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from yolo_params import Params, anchor_grid
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


def requantize(acc: np.ndarray, shift: int) -> np.ndarray:
    """`acc / 2**shift`, rounded half to even and saturated to int8."""
    scaled = np.round(acc / 2.0**shift) if shift >= 0 else acc * 2.0**-shift
    return np.clip(scaled, -128, 127).astype(np.int64)


def conv2d(x: np.ndarray, weight: np.ndarray, stride: int, groups: int) -> np.ndarray:
    """Exact integer convolution, `(C, H, W)` by `(O, C/groups, k, k)`, zero-padded by `k // 2`."""
    k = weight.shape[-1]
    padded = np.pad(x, ((0, 0), (k // 2, k // 2), (k // 2, k // 2))).astype(np.float64)
    windows = sliding_window_view(padded, (1, k, k))[:, ::stride, ::stride, 0]
    # float64 holds every partial sum exactly: |sum| < 128 * 128 * 9 * 512 < 2**53.
    if groups == 1:
        acc = np.tensordot(
            weight.astype(np.float64), windows, axes=([1, 2, 3], [0, 3, 4])
        )
    elif groups == x.shape[0] == weight.shape[0]:
        acc = np.einsum("chwij,cij->chw", windows, weight[:, 0].astype(np.float64))
    else:
        raise ValueError(
            f"unsupported grouped conv: groups={groups}, shape {weight.shape}"
        )
    return acc.astype(np.int64)


def topk(values: np.ndarray, k: int) -> np.ndarray:
    """Indices of the `k` largest values, ties to the lower index, as ONNX TopK orders them."""
    return np.lexsort((np.arange(values.size), -values))[:k]


class SoftmaxArbiter:
    """Decides an int8 attention probability exactly, to say who is off when two softmaxes disagree.

    Each `exp` is a correctly rounded `digits`-digit decimal, summed and
    divided as a `Fraction`, so the result is exact unless the true value
    lies within that error of a rounding boundary, which `exact` reports.

    Args:
        score_scale: The log2 scale of the int8 scores.
        prob_scale: The log2 scale of the int8 probabilities.
        digits: The decimal digits each exponential is rounded to.
    """

    def __init__(self, score_scale: int, prob_scale: int, digits: int = 60) -> None:
        self.score_scale = score_scale
        self.prob_scale = prob_scale
        self.digits = digits

    def exact(self, scores: np.ndarray, index: int) -> int | None:
        """The correctly rounded int8 probability of `scores[index]` in its softmax row.

        Args:
            scores: One row of int8 attention scores, at `2**score_scale`.
            index: The element whose probability to round.

        Returns:
            The probability on the `2**prob_scale` grid, rounded half to even
            and saturated to 127 as QuantizeLinear does, or None when the
            exponentials' error straddles a rounding boundary.
        """
        gaps = [int(g) for g in scores.max() - scores]
        step = Fraction(2) ** self.score_scale
        with localcontext() as ctx:
            ctx.prec = self.digits
            terms = [
                Fraction((-Decimal(g * step.numerator) / step.denominator).exp())
                for g in gaps
            ]
        value = terms[index] / sum(terms) / Fraction(2) ** self.prob_scale
        rounded = math.floor(value)
        excess = value - rounded - Fraction(1, 2)
        # Each exp is within half a unit in its last place, so the ratio is
        # within 10**(1 - digits) relatively; four times that is the margin.
        error = value * 4 * Fraction(1, 10 ** (self.digits - 1)) if any(gaps) else 0
        if error and abs(excess) <= error:
            return None
        if excess > 0 or excess == 0 and rounded % 2:
            rounded += 1
        return min(rounded, 127)

    def verdict(self, scores: np.ndarray, index: int, oracle: int, ort: int) -> str:
        """Who is off at one probability where the oracle and ONNX Runtime differ.

        Args:
            scores: The element's row of int8 attention scores.
            index: The element.
            oracle: The oracle's int8 probability.
            ort: ONNX Runtime's.

        Returns:
            "ORT off-boundary" when the oracle's is the exact one, "oracle
            wrong" when ONNX Runtime's is, "both wrong" when neither is, and
            "undecidable" when the exact value is too near a boundary to say.
        """
        exact = self.exact(scores, index)
        if exact is None:
            return "undecidable"
        if exact == oracle:
            return "ORT off-boundary"
        return "oracle wrong" if exact == ort else "both wrong"


class Oracle:
    def __init__(self, params: Params, network: Network | None = None) -> None:
        self.params = params
        self.network = network or Network.yolo26n()

    def run(
        self,
        image: np.ndarray,
        probabilities: dict[str, np.ndarray] | None = None,
    ) -> dict[str, np.ndarray]:
        """Every tensor of the network on one image, by name.

        Each attention also leaves its int8 scores and probabilities,
        `(heads, HW, HW)`, under `<op name>/scores` and `<op name>/probs`.

        Args:
            image: A letterboxed `(640, 640, 3)` uint8 RGB image.
            probabilities: Attention probabilities to use in place of the
                computed ones, by attention op name, to follow another
                softmax's rounding downstream.

        Returns:
            The tensors by name.

        Raises:
            ValueError: If `image` has another shape or dtype.
        """
        if image.shape != (640, 640, 3) or image.dtype != np.uint8:
            raise ValueError(
                f"expected a (640, 640, 3) uint8 image, got {image.shape} {image.dtype}"
            )
        p = self.params
        t: dict[str, np.ndarray] = {}
        for op in self.network.ops:
            match op:
                case InputQuant():
                    t[op.dst.name] = p.of(op.name, "table").astype(np.int64)[
                        image.transpose(2, 0, 1)
                    ]
                case Conv():
                    acc = conv2d(
                        t[op.src.name], p.of(op.name, "weight"), op.stride, op.groups
                    )
                    y = requantize(
                        acc + p.of(op.name, "bias")[:, None, None],
                        int(p.of(op.name, "shift")),
                    )
                    t[op.dst.name] = (
                        p.of(op.name, "table").astype(np.int64)[y + 128]
                        if op.act
                        else y
                    )
                case Add():
                    sa, sb = p.scale(op.a.name), p.scale(op.b.name)
                    base = min(sa, sb)
                    total = (t[op.a.name] << (sa - base)) + (
                        t[op.b.name] << (sb - base)
                    )
                    t[op.dst.name] = requantize(total, p.scale(op.dst.name) - base)
                case Concat():
                    t[op.dst.name] = np.concatenate([t[s.name] for s in op.srcs])
                case Slice():
                    t[op.dst.name] = t[op.src.name][
                        op.start : op.start + op.dst.channels
                    ]
                case MaxPool():
                    r = op.kernel // 2
                    padded = np.pad(
                        t[op.src.name], ((0, 0), (r, r), (r, r)), constant_values=-129
                    )
                    t[op.dst.name] = sliding_window_view(
                        padded, (1, op.kernel, op.kernel)
                    ).max(axis=(3, 4, 5))
                case Upsample():
                    t[op.dst.name] = t[op.src.name].repeat(2, axis=1).repeat(2, axis=2)
                case Attention():
                    self.attention(op, t, (probabilities or {}).get(op.name))
                case DetectPost():
                    t[op.dst.name] = self.detect(op, t)
        return t

    def attention(
        self,
        op: Attention,
        t: dict[str, np.ndarray],
        probabilities: np.ndarray | None,
    ) -> None:
        p = self.params
        s_qkv = p.scale(op.qkv.name)
        c, h, w = op.qkv.shape
        heads = t[op.qkv.name].reshape(op.heads, c // op.heads, h * w)
        q = heads[:, : op.key_dim]
        k = heads[:, op.key_dim : 2 * op.key_dim]
        v = heads[:, 2 * op.key_dim :]
        s_q = int(p.of(op.name, "q_scale"))
        q = requantize(
            q * int(p.of(op.name, "q_mul")),
            s_q - s_qkv - int(p.of(op.name, "q_mul_scale")),
        )
        s_s = int(p.of(op.name, "score_scale"))
        scores = requantize(np.einsum("hdi,hdj->hij", q, k), s_s - s_q - s_qkv)
        # Integer softmax: exp(-(max - s) * 2**s_s) from a Q26 table, then each
        # probability rounded exactly (half to even) to the output's int8 grid.
        e = p.of(op.name, "exp_table").astype(np.int64)[
            scores.max(axis=-1, keepdims=True) - scores
        ]
        s_p = int(p.of(op.name, "prob_scale"))
        numerator = e << -s_p
        total = e.sum(axis=-1, keepdims=True)
        quotient, remainder = np.divmod(2 * numerator, 2 * total)
        round_up = (remainder > total) | ((remainder == total) & (quotient % 2 == 1))
        probs = np.minimum(quotient + round_up, 127)
        t[f"{op.name}/scores"] = scores
        t[f"{op.name}/probs"] = probs
        if probabilities is not None:
            probs = probabilities.reshape(probs.shape).astype(np.int64)
        out = requantize(
            np.einsum("hdj,hij->hdi", v, probs), p.scale(op.dst.name) - s_qkv - s_p
        )
        t[op.values.name] = v.reshape(op.values.shape)
        t[op.dst.name] = out.reshape(op.dst.shape)

    def detect(self, op: DetectPost, t: dict[str, np.ndarray]) -> np.ndarray:
        p = self.params
        dist = np.concatenate(
            [
                (t[b.name] * np.float32(2.0 ** p.scale(b.name)))
                .reshape(4, -1)
                .astype(np.float32)
                for b in op.boxes
            ],
            axis=1,
        )
        sigmoid = p.of(op.name, "sigmoid")
        scores = np.concatenate(
            [
                sigmoid[i][t[s.name] + 128].reshape(s.channels, -1)
                for i, s in enumerate(op.scores)
            ],
            axis=1,
        ).T
        anchors, strides = anchor_grid(op)
        boxes = (np.concatenate([anchors - dist[:2], anchors + dist[2:]]) * strides).T
        k = op.dst.height
        first = topk(scores.max(axis=1), k)
        candidates = scores[first].reshape(-1)
        second = topk(candidates, k)
        classes = scores.shape[1]
        rows = first[second // classes]
        return np.concatenate(
            [
                boxes[rows],
                candidates[second][:, None],
                (second % classes).astype(np.float32)[:, None],
            ],
            axis=1,
        ).astype(np.float32)
