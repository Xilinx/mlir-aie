#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Extract the integer parameters of yolo26n from its XINT8 ONNX, or make synthetic ones.

Every operator of `yolo_spec.Network.yolo26n()` is bound to the ONNX nodes of
the same name; the extractor checks the graph's connectivity and shapes
against the spec as it goes. All scales are powers of two and all zero points
0, so each operator's parameters are int8 weights, int32 biases at the
accumulator's scale and log2 scales. Element-wise chains (SiLU, the input
quantizer, the score sigmoid) become 256-entry tables evaluated by ONNX
Runtime itself on every int8 input, so they match it by construction.

    python gen_yolo_data.py                  # from $IRON_EXAMPLE_WEIGHTS_DIR/yolo26n/yolo26n_xint8.onnx
    python gen_yolo_data.py --synthetic 7    # seeded synthetic weights, no checkpoint needed

Writes `data/yolo26n_params.npz` (or `--out`).
"""

from __future__ import annotations

import argparse
import math
import os
import tempfile
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper
from onnx.utils import extract_model

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


class QdqGraph:
    """A Quark QDQ graph, navigated through its Quantize/DequantizeLinear pairs."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.model = onnx.shape_inference.infer_shapes(onnx.load(path))
        self.nodes = {n.name: n for n in self.model.graph.node}
        self.consts = {
            t.name: numpy_helper.to_array(t) for t in self.model.graph.initializer
        }
        for n in self.model.graph.node:
            if n.op_type == "Constant":
                self.consts[n.output[0]] = numpy_helper.to_array(n.attribute[0].t)
        self.producer = {o: n for n in self.model.graph.node for o in n.output}
        self.consumers: dict[str, list] = {}
        for n in self.model.graph.node:
            for i in n.input:
                self.consumers.setdefault(i, []).append(n)

    def node(self, name: str, op_type: str):
        node = self.nodes.get(name)
        if node is None or node.op_type != op_type:
            raise ValueError(
                f"expected a {op_type} node named {name}, found {node and node.op_type}"
            )
        return node

    def _scale_log2(self, qdq) -> int:
        scale = self.consts[qdq.input[1]].reshape(-1)
        if scale.size != 1:
            raise ValueError(f"{qdq.name}: per-channel scale")
        exponent = math.log2(float(scale[0]))
        if exponent != round(exponent):
            raise ValueError(f"{qdq.name}: scale {scale[0]} is not a power of two")
        if len(qdq.input) > 2 and (self.consts[qdq.input[2]] != 0).any():
            raise ValueError(f"{qdq.name}: nonzero zero point")
        return int(exponent)

    def dequantized(self, tensor: str) -> tuple[str, int]:
        """The int8 tensor and log2 scale behind a DequantizeLinear output."""
        dq = self.producer[tensor]
        if dq.op_type != "DequantizeLinear":
            raise ValueError(
                f"{tensor} is produced by {dq.op_type}, not DequantizeLinear"
            )
        return dq.input[0], self._scale_log2(dq)

    def constant(self, tensor: str) -> tuple[np.ndarray, int]:
        q, scale = self.dequantized(tensor)
        return self.consts[q], scale

    def quantized(self, tensor: str) -> tuple[str, int]:
        """The int8 tensor and log2 scale of the QuantizeLinear reading a float tensor."""
        qls = [
            c for c in self.consumers.get(tensor, []) if c.op_type == "QuantizeLinear"
        ]
        if len(qls) != 1:
            raise ValueError(f"{tensor}: expected one QuantizeLinear, found {len(qls)}")
        return qls[0].output[0], self._scale_log2(qls[0])

    def evaluate(self, inputs: dict[str, np.ndarray], output: str) -> np.ndarray:
        """Run the subgraph from `inputs` to `output` in ONNX Runtime."""
        with tempfile.TemporaryDirectory() as tmp:
            whole, part = Path(tmp) / "whole.onnx", Path(tmp) / "part.onnx"
            onnx.save(self.model, whole)
            extract_model(str(whole), str(part), list(inputs), [output])
            session = ort.InferenceSession(
                str(part), providers=["CPUExecutionProvider"]
            )
            return session.run([output], inputs)[0]

    def shape(self, tensor: str) -> tuple[int, ...]:
        for v in list(self.model.graph.value_info) + list(self.model.graph.output):
            if v.name == tensor:
                return tuple(d.dim_value for d in v.type.tensor_type.shape.dim)
        raise KeyError(tensor)

    def table(self, src: str, dst: str) -> np.ndarray:
        """`dst` as a function of every int8 value of `src`, for an element-wise chain."""
        shape = self.shape(src)
        values = np.resize(INT8_VALUES, int(np.prod(shape))).reshape(shape)
        return self.evaluate({src: values}, dst).reshape(-1)[:256]


class Extractor:
    """Binds each spec operator to the ONNX graph and records its parameters."""

    def __init__(self, graph: QdqGraph) -> None:
        self.graph = graph
        self.params = Params()
        self.int8_name: dict[str, str] = {}

    def bind(self, tensor, onnx_tensor: str, scale: int) -> None:
        shape = self.graph.shape(onnx_tensor)
        if tuple(shape[-3:]) != tensor.shape:
            raise ValueError(f"{tensor.name}: spec shape {tensor.shape}, ONNX {shape}")
        self.int8_name[tensor.name] = onnx_tensor
        self.params.put(tensor.name, "scale", scale)
        self.params.put(tensor.name, "onnx", onnx_tensor)

    def source(self, tensor, onnx_float: str) -> int:
        """Check `onnx_float` dequantizes the int8 tensor bound to spec `tensor`; return its scale."""
        int8, scale = self.graph.dequantized(onnx_float)
        if self.int8_name.get(tensor.name) != int8:
            raise ValueError(
                f"{tensor.name} is bound to {self.int8_name.get(tensor.name)}, ONNX reads {int8}"
            )
        if scale != self.params.scale(tensor.name):
            raise ValueError(
                f"{tensor.name}: scale {scale} vs {self.params.scale(tensor.name)}"
            )
        return scale

    def run(self, network: Network) -> Params:
        for op in network.ops:
            match op:
                case InputQuant():
                    self.input_quant(op)
                case Conv():
                    self.conv(op)
                case Add():
                    self.add(op)
                case Concat():
                    self.same_scale(op, self.graph.node(op.name, "Concat"), op.srcs)
                case Slice():
                    self.slice(op)
                case MaxPool():
                    self.same_scale(op, self.graph.node(op.name, "MaxPool"), (op.src,))
                case Upsample():
                    self.same_scale(op, self.graph.node(op.name, "Resize"), (op.src,))
                case Attention():
                    self.attention(op)
                case DetectPost():
                    self.detect(op)
        return self.params

    def input_quant(self, op: InputQuant) -> None:
        q, scale = self.graph.quantized(op.name)
        pixels = np.arange(256, dtype=np.float32) / np.float32(255)
        shape = (1, 3, 640, 640)
        values = np.resize(pixels, int(np.prod(shape))).reshape(shape)
        table = self.graph.evaluate({op.name: values}, q).reshape(-1)[:256]
        self.params.put(op.name, "table", table.astype(np.int8))
        self.bind(op.dst, q, scale)

    def conv(self, op: Conv) -> None:
        node = self.graph.node(op.name, "Conv")
        attrs = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
        if (attrs["kernel_shape"][0], attrs["strides"][0], attrs.get("group", 1)) != (
            op.kernel,
            op.stride,
            op.groups,
        ):
            raise ValueError(
                f"{op.name}: {attrs} vs spec k={op.kernel} s={op.stride} g={op.groups}"
            )
        sx = self.source(op.src, node.input[0])
        weight, sw = self.graph.constant(node.input[1])
        bias, sb = self.graph.constant(node.input[2])
        if sb < sx + sw:
            raise ValueError(
                f"{op.name}: bias scale 2^{sb} is finer than the accumulator's 2^{sx + sw}"
            )
        conv_q, sy = self.graph.quantized(node.output[0])
        self.params.put(op.name, "weight", weight.astype(np.int8))
        self.params.put(op.name, "bias", bias.astype(np.int64) << (sb - sx - sw))
        self.params.put(op.name, "shift", sy - sx - sw)
        if not op.act:
            self.bind(op.dst, conv_q, sy)
            return
        act_q, sa = self.graph.quantized(
            self.graph.node(f"{op.scope}/act/Mul", "Mul").output[0]
        )
        self.params.put(
            op.name, "table", self.graph.table(conv_q, act_q).astype(np.int8)
        )
        self.bind(op.dst, act_q, sa)

    def add(self, op: Add) -> None:
        node = self.graph.node(op.name, "Add")
        operands = {self.graph.dequantized(i)[0] for i in node.input}
        if operands != {self.int8_name[op.a.name], self.int8_name[op.b.name]}:
            raise ValueError(f"{op.name}: operands {operands} do not match the spec")
        for src in (op.a, op.b):
            reader = next(
                i
                for i in node.input
                if self.graph.dequantized(i)[0] == self.int8_name[src.name]
            )
            self.source(src, reader)
        self.bind(op.dst, *self.graph.quantized(node.output[0]))

    def same_scale(self, op, node, srcs) -> None:
        scales = {self.source(s, i) for s, i in zip(srcs, node.input)}
        out, scale = self.graph.quantized(node.output[0])
        if scales != {scale}:
            raise ValueError(f"{op.name}: input scales {scales} vs output {scale}")
        self.bind(op.dst, out, scale)

    def slice(self, op: Slice) -> None:
        node = self.graph.node(op.name, "Slice")
        starts, axes = (
            self.graph.consts[node.input[1]],
            self.graph.consts[node.input[3]],
        )
        if list(axes) != [1] or int(starts[0]) != op.start:
            raise ValueError(f"{op.name}: slices axes {axes} from {starts}")
        self.same_scale(op, node, (op.src,))

    def attention(self, op: Attention) -> None:
        prefix = op.name
        reshape = self.graph.node(f"{prefix}/Reshape", "Reshape")
        s_qkv = self.source(op.qkv, reshape.input[0])
        split = [
            self.graph.node(f"{prefix}/Split_output_{i}_{i}", "Slice") for i in range(3)
        ]
        sizes = [
            int(self.graph.consts[s.input[2]][0] - self.graph.consts[s.input[1]][0])
            for s in split
        ]
        if sizes != [op.key_dim, op.key_dim, op.head_dim]:
            raise ValueError(f"{prefix}: q/k/v split {sizes}")
        for node in [
            reshape,
            *split,
            self.graph.node(f"{prefix}/Reshape_2", "Reshape"),
        ]:
            if self.graph.quantized(node.output[0])[1] != s_qkv:
                raise ValueError(f"{node.name} requantizes")
        mul = self.graph.node(f"{prefix}/Mul", "Mul")
        q_scale_value, s_c = self.graph.constant(mul.input[1])
        s_q = self.graph.quantized(mul.output[0])[1]
        transpose = self.graph.node(f"{prefix}/Transpose", "Transpose")
        if self.graph.quantized(transpose.output[0])[1] != s_q:
            raise ValueError(f"{transpose.name} requantizes")
        s_s = self.graph.quantized(
            self.graph.node(f"{prefix}/MatMul", "MatMul").output[0]
        )[1]
        s_p = self.graph.quantized(
            self.graph.node(f"{prefix}/Softmax", "Softmax").output[0]
        )[1]
        out_mm = self.graph.node(f"{prefix}/MatMul_1", "MatMul")
        s_o = self.graph.quantized(out_mm.output[0])[1]
        reshape_1 = self.graph.node(f"{prefix}/Reshape_1", "Reshape")
        if self.graph.quantized(reshape_1.output[0])[1] != s_o:
            raise ValueError(f"{reshape_1.name} requantizes")
        for field, value in {
            "exp_table": exp_table(s_s),
            "q_mul": int(q_scale_value.reshape(-1)[0]),
            "q_mul_scale": s_c,
            "q_scale": s_q,
            "score_scale": s_s,
            "prob_scale": s_p,
        }.items():
            self.params.put(prefix, field, value)
        self.bind(
            op.values,
            *self.graph.quantized(
                self.graph.node(f"{prefix}/Reshape_2", "Reshape").output[0]
            ),
        )
        self.bind(op.dst, *self.graph.quantized(reshape_1.output[0]))

    def detect(self, op: DetectPost) -> None:
        boxes = [self.int8_name[t.name] for t in op.boxes]
        scores = [self.int8_name[t.name] for t in op.scores]
        sigmoid = self.graph.node(f"{op.name}/Sigmoid", "Sigmoid").output[0]
        inputs = {}
        for name in scores:
            shape = self.graph.shape(name)
            inputs[name] = np.resize(INT8_VALUES, int(np.prod(shape))).reshape(shape)
        out = self.graph.evaluate(inputs, sigmoid).reshape(80, -1)
        starts = np.cumsum([0] + [t.height * t.width for t in op.scores[:-1]])
        self.params.put(
            op.name,
            "sigmoid",
            np.stack([out[0, s : s + 256] for s in starts]).astype(np.float32),
        )
        anchors = self.graph.consts[
            self.graph.node(f"{op.name}/Sub", "Sub").input[0]
        ].reshape(2, -1)
        strides = self.graph.consts[
            self.graph.node(f"{op.name}/Mul_2", "Mul").input[1]
        ].reshape(-1)
        expected_anchors, expected_strides = anchor_grid(op)
        if not (
            np.array_equal(anchors, expected_anchors)
            and np.array_equal(strides, expected_strides)
        ):
            raise ValueError(f"{op.name}: anchors or strides differ from the spec")
        for name in boxes + scores:
            if name not in self.int8_name.values():
                raise ValueError(f"{op.name}: {name} is unbound")


def exp_table(score_scale: int) -> np.ndarray:
    """`round(2**22 * exp(-d * 2**score_scale))` for every int8 score gap `d` in 0..255.

    400 keys of at most 2**22 sum below 2**31, so a row's total fits int32.
    """
    gaps = np.arange(256, dtype=np.float64)
    return np.round(2.0**22 * np.exp(-gaps * 2.0**score_scale)).astype(np.int32)


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--onnx", type=Path, default=WEIGHTS_DIR / "yolo26n_xint8.onnx")
    parser.add_argument(
        "--synthetic", type=int, metavar="SEED", help="seeded synthetic weights instead"
    )
    parser.add_argument("--out", type=Path, default=DATA_DIR / "yolo26n_params.npz")
    args = parser.parse_args()

    network = Network.yolo26n()
    if args.synthetic is not None:
        params = Synthesizer(args.synthetic).run(network)
    else:
        if not args.onnx.exists():
            raise SystemExit(f"{args.onnx} is missing; run export_yolo26n.py first")
        params = Extractor(QdqGraph(args.onnx)).run(network)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **params)
    print(f"wrote {args.out}: {len(params)} arrays")


if __name__ == "__main__":
    main()
