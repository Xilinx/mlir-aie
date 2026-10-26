#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""YOLO26n detection network at 640x640, as integer operators.

Pure Python, no AIE imports. `Network.yolo26n()` lays the model out the way
Ultralytics' `yolo26.yaml` does at scale n (depth 0.50, width 0.25), one
operator per quantized ONNX node group. Every operator's `name` is the ONNX
node it stands for, so the extractor binds parameters by name and checks the
exported graph against this layout:

    0  Conv 3x3 s2        3 -> 16   320x320
    1  Conv 3x3 s2       16 -> 32   160x160
    2  C3k2               32 -> 64   160x160
    3  Conv 3x3 s2       64 -> 64    80x80
    4  C3k2 (P3 skip)     64 -> 128   80x80
    5  Conv 3x3 s2      128 -> 128   40x40
    6  C3k2 c3k (P4 skip)      128   40x40
    7  Conv 3x3 s2      128 -> 256   20x20
    8  C3k2 c3k                256   20x20
    9  SPPF                    256   20x20
    10 C2PSA (P5 skip)         256   20x20
    11-22 neck: upsample, concat, C3k2, stride-2 conv; P3 64ch, P4 128ch, P5 256ch
    23 Detect one2one head and its end-to-end top-k, (300, 6)

Shapes are (C, H, W), batch 1.
"""

from __future__ import annotations

from dataclasses import dataclass

IMAGE_SIZE = 640
NUM_CLASSES = 80
MAX_DETECTIONS = 300
STRIDES = (8, 16, 32)


@dataclass(frozen=True)
class Tensor:
    name: str
    channels: int
    height: int
    width: int

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.channels, self.height, self.width)

    @property
    def channels8(self) -> int:
        """Channels as the NPU holds them: rounded up to a group of 8."""
        return -(-self.channels // 8) * 8

    @property
    def row_bytes(self) -> int:
        """One row on the NPU: `[channels8 / 8][width][8]` int8."""
        return self.channels8 * self.width


@dataclass(frozen=True)
class InputQuant:
    """uint8 RGB pixels to the int8 network input: `/255`, then QuantizeLinear."""

    name: str
    dst: Tensor


@dataclass(frozen=True)
class Conv:
    """Conv2d with BatchNorm folded into its bias; `act` adds the SiLU."""

    name: str
    src: Tensor
    dst: Tensor
    kernel: int
    stride: int
    groups: int
    act: bool

    @property
    def scope(self) -> str:
        return self.name.removesuffix("/conv/Conv").removesuffix("/Conv")


@dataclass(frozen=True)
class Add:
    name: str
    a: Tensor
    b: Tensor
    dst: Tensor


@dataclass(frozen=True)
class Concat:
    name: str
    srcs: tuple[Tensor, ...]
    dst: Tensor


@dataclass(frozen=True)
class Slice:
    """Channels `[start, start + dst.channels)` of `src`."""

    name: str
    src: Tensor
    dst: Tensor
    start: int


@dataclass(frozen=True)
class MaxPool:
    name: str
    src: Tensor
    dst: Tensor
    kernel: int


@dataclass(frozen=True)
class Upsample:
    """Nearest-neighbour 2x."""

    name: str
    src: Tensor
    dst: Tensor


@dataclass(frozen=True)
class Attention:
    """PSA multi-head attention over the H*W positions of a qkv tensor.

    `qkv` holds, per head, `key_dim` query, `key_dim` key and `head_dim`
    value channels. `values` is the value channels of every head, which the
    positional-encoding conv reads; `dst` is softmax(q^T k * scale) applied
    to the values, in the same layout.
    """

    name: str
    qkv: Tensor
    values: Tensor
    dst: Tensor
    heads: int
    key_dim: int
    head_dim: int


@dataclass(frozen=True)
class DetectPost:
    """Ultralytics' end-to-end postprocess: sigmoid, box decode, two top-k.

    `dst` is `(6, max_det, 1)`: per detection x1, y1, x2, y2, score, class.
    """

    name: str
    boxes: tuple[Tensor, ...]
    scores: tuple[Tensor, ...]
    dst: Tensor
    strides: tuple[int, ...]


Op = (
    InputQuant
    | Conv
    | Add
    | Concat
    | Slice
    | MaxPool
    | Upsample
    | Attention
    | DetectPost
)


class Network:
    """Builds the operator list module by module, mirroring Ultralytics."""

    def __init__(self) -> None:
        self.ops: list[Op] = []

    def _emit(self, op: Op) -> Tensor:
        self.ops.append(op)
        return op.dst

    def span(self, first: str, last: str) -> tuple[int, int]:
        """The index range `(start, stop)` of the ops from `first` through `last`.

        Args:
            first: The name of the range's first op.
            last: The name of its last op.

        Returns:
            `ops[start:stop]` is the range, `last` included.

        Raises:
            KeyError: If either name is no op's.
            ValueError: If two ops share a name, or `last` comes before `first`.
        """
        index = {op.name: i for i, op in enumerate(self.ops)}
        if len(index) != len(self.ops):
            raise ValueError("the network's op names are not unique")
        missing = [name for name in (first, last) if name not in index]
        if missing:
            raise KeyError(f"no op is named {', '.join(missing)}")
        start, stop = index[first], index[last] + 1
        if stop <= start:
            raise ValueError(f"{last} comes before {first}")
        return start, stop

    def conv(
        self,
        scope: str,
        src: Tensor,
        channels: int,
        kernel: int,
        stride: int = 1,
        groups: int = 1,
        act: bool = True,
        plain: bool = False,
    ) -> Tensor:
        """A Ultralytics `Conv` (conv + BN + optional SiLU), or a bare `nn.Conv2d` when `plain`."""
        name = f"{scope}/Conv" if plain else f"{scope}/conv/Conv"
        height = (src.height + 2 * (kernel // 2) - kernel) // stride + 1
        width = (src.width + 2 * (kernel // 2) - kernel) // stride + 1
        dst = Tensor(name, channels, height, width)
        return self._emit(
            Conv(name, src, dst, kernel, stride, groups, act and not plain)
        )

    def add(self, name: str, a: Tensor, b: Tensor) -> Tensor:
        return self._emit(Add(name, a, b, Tensor(name, *a.shape)))

    def concat(self, name: str, *srcs: Tensor) -> Tensor:
        channels = sum(s.channels for s in srcs)
        dst = Tensor(name, channels, srcs[0].height, srcs[0].width)
        return self._emit(Concat(name, srcs, dst))

    def split(self, scope: str, src: Tensor, sizes: tuple[int, ...]) -> list[Tensor]:
        outs, start = [], 0
        for index, size in enumerate(sizes):
            name = f"{scope}/Split_output_{index}_{index}"
            outs.append(
                self._emit(
                    Slice(name, src, Tensor(name, size, src.height, src.width), start)
                )
            )
            start += size
        return outs

    def bottleneck(
        self, scope: str, src: Tensor, channels: int, expansion: float
    ) -> Tensor:
        hidden = int(channels * expansion)
        y = self.conv(f"{scope}/cv1", src, hidden, 3)
        y = self.conv(f"{scope}/cv2", y, channels, 3)
        return self.add(f"{scope}/Add", src, y)

    def c3k(self, scope: str, src: Tensor, channels: int) -> Tensor:
        hidden = channels // 2
        a = self.conv(f"{scope}/cv1", src, hidden, 1)
        for index in range(2):
            a = self.bottleneck(f"{scope}/m/m.{index}", a, hidden, 1.0)
        b = self.conv(f"{scope}/cv2", src, hidden, 1)
        return self.conv(
            f"{scope}/cv3", self.concat(f"{scope}/Concat", a, b), channels, 1
        )

    def psa_block(self, scope: str, src: Tensor) -> Tensor:
        channels = src.channels
        heads = channels // 64
        head_dim = channels // heads
        key_dim = head_dim // 2
        qkv = self.conv(
            f"{scope}/attn/qkv", src, channels + 2 * heads * key_dim, 1, act=False
        )
        values = Tensor(f"{scope}/attn/Reshape_2", channels, src.height, src.width)
        attn = Tensor(f"{scope}/attn/Reshape_1", channels, src.height, src.width)
        self._emit(
            Attention(f"{scope}/attn", qkv, values, attn, heads, key_dim, head_dim)
        )
        pe = self.conv(
            f"{scope}/attn/pe", values, channels, 3, groups=channels, act=False
        )
        y = self.conv(
            f"{scope}/attn/proj",
            self.add(f"{scope}/attn/Add", attn, pe),
            channels,
            1,
            act=False,
        )
        x = self.add(f"{scope}/Add", src, y)
        y = self.conv(f"{scope}/ffn/ffn.0", x, 2 * channels, 1)
        y = self.conv(f"{scope}/ffn/ffn.1", y, channels, 1, act=False)
        return self.add(f"{scope}/Add_1", x, y)

    def c3k2(
        self,
        scope: str,
        src: Tensor,
        channels: int,
        c3k: bool,
        expansion: float = 0.5,
        attn: bool = False,
    ) -> Tensor:
        hidden = int(channels * expansion)
        a, b = self.split(
            scope, self.conv(f"{scope}/cv1", src, 2 * hidden, 1), (hidden, hidden)
        )
        if attn:
            m = self.bottleneck(f"{scope}/m.0/m.0.0", b, hidden, 0.5)
            m = self.psa_block(f"{scope}/m.0/m.0.1", m)
        elif c3k:
            m = self.c3k(f"{scope}/m.0", b, hidden)
        else:
            m = self.bottleneck(f"{scope}/m.0", b, hidden, 0.5)
        return self.conv(
            f"{scope}/cv2", self.concat(f"{scope}/Concat", a, b, m), channels, 1
        )

    def sppf(self, scope: str, src: Tensor) -> Tensor:
        hidden = src.channels // 2
        y = [self.conv(f"{scope}/cv1", src, hidden, 1, act=False)]
        for suffix in ("m", "m_1", "m_2"):
            name = f"{scope}/{suffix}/MaxPool"
            y.append(self._emit(MaxPool(name, y[-1], Tensor(name, *y[-1].shape), 5)))
        y = self.conv(
            f"{scope}/cv2", self.concat(f"{scope}/Concat", *y), src.channels, 1
        )
        return self.add(f"{scope}/Add", y, src)

    def c2psa(self, scope: str, src: Tensor) -> Tensor:
        hidden = src.channels // 2
        a, b = self.split(
            scope, self.conv(f"{scope}/cv1", src, 2 * hidden, 1), (hidden, hidden)
        )
        b = self.psa_block(f"{scope}/m/m.0", b)
        return self.conv(
            f"{scope}/cv2", self.concat(f"{scope}/Concat", a, b), src.channels, 1
        )

    def upsample(self, name: str, src: Tensor) -> Tensor:
        return self._emit(
            Upsample(
                name, src, Tensor(name, src.channels, 2 * src.height, 2 * src.width)
            )
        )

    def detect(self, scope: str, levels: tuple[Tensor, ...]) -> Tensor:
        box_hidden = max(16, levels[0].channels // 4)
        cls_hidden = max(levels[0].channels, min(NUM_CLASSES, 100))
        boxes, scores = [], []
        for i, x in enumerate(levels):
            box = f"{scope}/one2one_cv2.{i}/one2one_cv2.{i}"
            y = self.conv(f"{box}.0", x, box_hidden, 3)
            y = self.conv(f"{box}.1", y, box_hidden, 3)
            boxes.append(self.conv(f"{box}.2", y, 4, 1, plain=True))
            cls = f"{scope}/one2one_cv3.{i}/one2one_cv3.{i}"
            y = self.conv(
                f"{cls}.0/one2one_cv3.{i}.0.0", x, x.channels, 3, groups=x.channels
            )
            y = self.conv(f"{cls}.0/one2one_cv3.{i}.0.1", y, cls_hidden, 1)
            y = self.conv(
                f"{cls}.1/one2one_cv3.{i}.1.0", y, cls_hidden, 3, groups=cls_hidden
            )
            y = self.conv(f"{cls}.1/one2one_cv3.{i}.1.1", y, cls_hidden, 1)
            scores.append(self.conv(f"{cls}.2", y, NUM_CLASSES, 1, plain=True))
        dst = Tensor(scope, 6, MAX_DETECTIONS, 1)
        return self._emit(DetectPost(scope, tuple(boxes), tuple(scores), dst, STRIDES))

    @classmethod
    def yolo26n(cls) -> Network:
        net = cls()
        x = net._emit(
            InputQuant(
                "images", Tensor("images/QuantizeLinear", 3, IMAGE_SIZE, IMAGE_SIZE)
            )
        )
        x = net.conv("/model.0", x, 16, 3, 2)
        x = net.conv("/model.1", x, 32, 3, 2)
        x = net.c3k2("/model.2", x, 64, c3k=False, expansion=0.25)
        x = net.conv("/model.3", x, 64, 3, 2)
        p3 = net.c3k2("/model.4", x, 128, c3k=False, expansion=0.25)
        x = net.conv("/model.5", p3, 128, 3, 2)
        p4 = net.c3k2("/model.6", x, 128, c3k=True)
        x = net.conv("/model.7", p4, 256, 3, 2)
        x = net.c3k2("/model.8", x, 256, c3k=True)
        x = net.sppf("/model.9", x)
        p5 = net.c2psa("/model.10", x)
        x = net.concat("/model.12/Concat", net.upsample("/model.11/Resize", p5), p4)
        n13 = net.c3k2("/model.13", x, 128, c3k=True)
        x = net.concat("/model.15/Concat", net.upsample("/model.14/Resize", n13), p3)
        out3 = net.c3k2("/model.16", x, 64, c3k=True)
        x = net.concat("/model.18/Concat", net.conv("/model.17", out3, 64, 3, 2), n13)
        out4 = net.c3k2("/model.19", x, 128, c3k=True)
        x = net.concat("/model.21/Concat", net.conv("/model.20", out4, 128, 3, 2), p5)
        out5 = net.c3k2("/model.22", x, 256, c3k=True, attn=True)
        net.detect("/model.23", (out3, out4, out5))
        return net
