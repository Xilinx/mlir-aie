"""
gen_yolo_data.py — extract INT8 weights, biases, and quant params from the
Quark-quantized XINT8 ONNX into the layout AIE bottleneck kernels expect.

For each Conv / MatMul / Gemm node:
  - Pull INT8 weight initializer; reorder OIYX → OIYXI8O8 (MobileNet ref layout)
  - Pull INT8 bias initializer; promote to INT32 = (bias_i8 << pre_shift)
  - Compute right_shift = log2(scale_in) + log2(scale_w) - log2(scale_out)
  - Emit per-layer binary blobs to data/<block>/<layer>_{weights,bias}.bin
  - Emit data/manifest.json with shapes, scales, shifts, file paths

Bias is promoted to INT32 at extract time so the runtime kernel just uses it
as the accumulator init value when weight_index == 0 — no separate bias-add
stage. See README § "Int8 recipe" for the requantization math.

Usage:
    python gen_yolo_data.py [path/to/model.onnx]
"""

import json
import math
import sys
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper

_HERE = Path(__file__).resolve().parent
DEFAULT_MODEL = _HERE / "models" / "phase1_25k_xint8_acc0.8968.onnx"
DEFAULT_OUTDIR = _HERE / "data"


# --------------------------------------------------------------------------
# Quant helpers
# --------------------------------------------------------------------------


def block_of(name: str) -> str:
    """Group a node by 'model.N' prefix; matches inspect_xint8_graph.py."""
    s = name.lstrip("/")
    if not s.startswith("model."):
        return "global"
    parts = s.replace("/", ".").split(".")
    return f"model.{parts[1]}" if len(parts) >= 2 and parts[1].isdigit() else "global"


def trace_dq(nodes_by_output, initializers, tensor_name):
    """Follow tensor_name back through DequantizeLinear; return (quant_init, scale, zp) or (None, None, None)."""
    producer = nodes_by_output.get(tensor_name)
    if producer is None or producer.op_type != "DequantizeLinear":
        return None, None, None
    q_init = initializers.get(producer.input[0])
    scale_init = initializers.get(producer.input[1])
    zp_init = initializers.get(producer.input[2]) if len(producer.input) > 2 else None
    return q_init, scale_init, zp_init


def scalar_scale(scale_init):
    """Extract a single FP32 scale value (we audited 100% per-tensor pow-2)."""
    if scale_init is None:
        return None
    arr = numpy_helper.to_array(scale_init).flatten()
    return float(arr[0])


def log2_int(x: float, tol: float = 1e-6) -> int:
    """log2 of an exact power of 2, with a tight tolerance check."""
    lg = math.log2(x)
    lg_round = round(lg)
    if abs(lg - lg_round) > tol:
        raise ValueError(f"Scale {x} is not an exact power of 2 (log2={lg})")
    return int(lg_round)


def conv_attrs(node):
    attrs = {a.name: a for a in node.attribute}

    def _ints(a, default):
        return list(a.ints) if a is not None and a.ints else default

    return {
        "kernel": _ints(attrs.get("kernel_shape"), [1, 1]),
        "strides": _ints(attrs.get("strides"), [1, 1]),
        "pads": _ints(attrs.get("pads"), [0, 0, 0, 0]),
        "dilations": _ints(attrs.get("dilations"), [1, 1]),
        "group": attrs["group"].i if "group" in attrs else 1,
    }


# --------------------------------------------------------------------------
# Weight layout: OIYX → OIYXI8O8
# --------------------------------------------------------------------------


def reorder_oiyx_to_oiyxi8o8(w_oiyx: np.ndarray) -> np.ndarray:
    """Reorder a Conv weight tensor [O, I, Y, X] → [O//8, I//8, Y, X, 8 (I-inner), 8 (O-inner)].

    The AIE bottleneck kernels expect 8×8 vector tiles where the inner dims are
    I8 then O8. Requires O and I both divisible by 8 (caller must pad or special-case).
    """
    assert w_oiyx.ndim == 4, f"expected OIYX, got shape {w_oiyx.shape}"
    O, I, Y, X = w_oiyx.shape
    if O % 8 != 0 or I % 8 != 0:
        raise ValueError(f"OIYXI8O8 requires O,I divisible by 8; got O={O}, I={I}")

    # [O, I, Y, X] → [O//8, 8, I//8, 8, Y, X]
    w = w_oiyx.reshape(O // 8, 8, I // 8, 8, Y, X)
    # Permute to [O//8, I//8, Y, X, I_inner(8), O_inner(8)]
    #   axis ordering: O_outer=0, O_inner=1, I_outer=2, I_inner=3, Y=4, X=5
    #   target:        O_outer,    I_outer=2, Y=4,     X=5,       I_inner=3, O_inner=1
    return np.ascontiguousarray(w.transpose(0, 2, 4, 5, 3, 1))


# --------------------------------------------------------------------------
# Extractor
# --------------------------------------------------------------------------


@dataclass
class LayerData:
    name: str
    op: str
    block: str
    # Shape
    in_shape: list = field(default_factory=list)
    out_shape: list = field(default_factory=list)
    kernel_shape: list = field(default_factory=list)
    strides: list = field(default_factory=list)
    pads: list = field(default_factory=list)
    group: int = 1
    # Quant
    in_log2_scale: int | None = None
    in_b_log2_scale: int | None = None  # MatMul: scale of the second (dynamic) input
    wt_log2_scale: int | None = None
    out_log2_scale: int | None = None
    bias_log2_scale: int | None = None
    right_shift: int | None = None
    bias_pre_shift: int | None = None
    # File paths (relative to outdir)
    weights_file: str | None = None
    weights_layout: str | None = None  # "OIYXI8O8" or "OIYX_raw" for special cases
    weights_shape: list | None = None
    bias_file: str | None = None
    bias_int32_range: list | None = None
    # Flags
    has_bias: bool = False
    skipped_reason: str | None = None  # if extraction skipped (e.g., dynamic operands)


def extract_layer(node, graph_helpers, value_info, outdir: Path) -> LayerData:
    """Pull the quantized weights, bias, and scales for one Conv/MatMul/Gemm node."""
    nodes_by_output, initializers = graph_helpers
    blk = block_of(node.name)
    rec = LayerData(name=node.name, op=node.op_type, block=blk)

    # --- Shapes (from shape inference) ---
    def _shape(name):
        vi = value_info.get(name)
        if vi is None:
            return []
        return [
            d.dim_value if d.dim_value else (d.dim_param or None)
            for d in vi.type.tensor_type.shape.dim
        ]

    if len(node.input) >= 1:
        rec.in_shape = _shape(node.input[0])
    if len(node.output) >= 1:
        rec.out_shape = _shape(node.output[0])

    # --- Conv attributes ---
    if node.op_type == "Conv":
        a = conv_attrs(node)
        rec.kernel_shape = a["kernel"]
        rec.strides = a["strides"]
        rec.pads = a["pads"]
        rec.group = a["group"]

    # --- Trace input scale ---
    _, in_scale_init, _ = trace_dq(nodes_by_output, initializers, node.input[0])
    in_scale = scalar_scale(in_scale_init)
    if in_scale:
        try:
            rec.in_log2_scale = log2_int(in_scale)
        except ValueError:
            rec.in_log2_scale = None

    # --- Softmax: scales only (no weights, no bias). INT8 LUT kernel needs
    #     input + output scale to size the lookup table.
    if node.op_type == "Softmax":
        rec.weights_layout = "none"
        rec.weights_shape = []
        rec.has_bias = False
        rec.skipped_reason = "softmax: scales only, no weights (INT8 LUT kernel)"
        return rec

    # --- Trace weight: must be static initializer for Conv/Gemm. For MatMul
    #     with a dynamic B operand (PSA Q@K, S@V), trace input B as another
    #     quantized activation and record its scale instead.
    weight_init = scale_init = zp_init = None
    if len(node.input) >= 2:
        weight_init, scale_init, zp_init = trace_dq(
            nodes_by_output, initializers, node.input[1]
        )

    if weight_init is None:
        # Dynamic-operand MatMul: still extract scales so the GEMM kernel has
        # enough metadata for requant. Both inputs are quantized activations.
        _, in_b_scale_init, _ = trace_dq(nodes_by_output, initializers, node.input[1])
        in_b_scale = scalar_scale(in_b_scale_init)
        if in_b_scale:
            try:
                rec.in_b_log2_scale = log2_int(in_b_scale)
            except ValueError:
                pass
        rec.skipped_reason = (
            "dynamic weight operand (no static initializer); "
            "GEMM kernel uses in_log2_scale + in_b_log2_scale + out_log2_scale"
        )
        return rec

    wt_scale = scalar_scale(scale_init)
    if wt_scale:
        try:
            rec.wt_log2_scale = log2_int(wt_scale)
        except ValueError:
            rec.wt_log2_scale = None

    # --- Output quant: walk forward to find QuantizeLinear consuming this node's output ---
    # (find a QuantizeLinear whose input[0] is our output)
    out_scale = None
    for n in graph_helpers[
        0
    ].values():  # nodes_by_output is a single-output map; flatten manually below
        pass
    # Build a forward index lazily
    # (We'll attach this on the caller; for now do a linear search inside this helper.)
    rec.out_log2_scale = None  # filled in by caller

    # --- Trace bias ---
    bias_init = bias_scale_init = None
    if len(node.input) >= 3:
        bias_init, bias_scale_init, _ = trace_dq(
            nodes_by_output, initializers, node.input[2]
        )

    # --- Pull INT8 weights and write binary ---
    w_i8 = numpy_helper.to_array(weight_init).astype(np.int8)
    rec.weights_shape = list(w_i8.shape)
    block_dir = outdir / blk
    block_dir.mkdir(parents=True, exist_ok=True)
    safe = node.name.lstrip("/").replace("/", "_")

    # Try OIYXI8O8 reorder if applicable
    can_reorder = (
        node.op_type == "Conv"
        and w_i8.ndim == 4
        and w_i8.shape[0] % 8 == 0
        and w_i8.shape[1] % 8 == 0
        and rec.group == 1  # depthwise needs different layout (bn_conv2dk3_dw)
    )
    if can_reorder:
        w_reordered = reorder_oiyx_to_oiyxi8o8(w_i8)
        wt_path = block_dir / f"{safe}_weights_OIYXI8O8.bin"
        w_reordered.tofile(wt_path)
        rec.weights_layout = "OIYXI8O8"
    else:
        wt_path = block_dir / f"{safe}_weights_raw.bin"
        w_i8.tofile(wt_path)
        rec.weights_layout = (
            "OIYX_raw" if w_i8.ndim == 4 else f"shape_{'x'.join(map(str, w_i8.shape))}"
        )
    rec.weights_file = str(wt_path.relative_to(outdir.parent))

    # --- Bias: promote to INT32 ---
    if bias_init is not None:
        b_i8 = numpy_helper.to_array(bias_init).astype(np.int32)
        b_scale = scalar_scale(bias_scale_init)
        rec.has_bias = True
        if b_scale and in_scale and wt_scale:
            expected = in_scale * wt_scale
            ratio = b_scale / expected
            try:
                pre_shift = log2_int(ratio)
                rec.bias_pre_shift = pre_shift
                rec.bias_log2_scale = log2_int(b_scale)
                # Promote: bias_int32_native = bias_i8 << pre_shift
                # (in original accumulator units, ready to use as accum-init)
                b_int32 = (b_i8.astype(np.int64) << pre_shift).astype(np.int32)
                rec.bias_int32_range = [int(b_int32.min()), int(b_int32.max())]
            except ValueError:
                rec.skipped_reason = (
                    rec.skipped_reason or ""
                ) + f" | bias scale ratio not pow2 ({ratio})"
                b_int32 = b_i8.astype(np.int32)
        else:
            b_int32 = b_i8.astype(np.int32)

        bias_path = block_dir / f"{safe}_bias_int32.bin"
        b_int32.tofile(bias_path)
        rec.bias_file = str(bias_path.relative_to(outdir.parent))

    return rec


def build_output_scale_index(graph):
    """Map node_output_name → log2_scale via the QuantizeLinear that consumes it."""
    inits = {t.name: t for t in graph.initializer}
    out_scale_log2 = {}
    for n in graph.node:
        if n.op_type != "QuantizeLinear":
            continue
        src_name = n.input[0]
        scale_init = inits.get(n.input[1])
        s = scalar_scale(scale_init)
        if s is not None and s > 0:
            try:
                out_scale_log2[src_name] = log2_int(s)
            except ValueError:
                pass
    return out_scale_log2


def main():
    in_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_MODEL
    outdir = DEFAULT_OUTDIR
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Loading: {in_path}")
    model = onnx.load(str(in_path))
    graph = model.graph

    # Force a deployment-shaped input to enable shape inference for activations.
    inp = graph.input[0]
    inp.type.tensor_type.ClearField("shape")
    for d in [4, 3, 512, 512]:
        inp.type.tensor_type.shape.dim.add().dim_value = d
    model = onnx.shape_inference.infer_shapes(model, strict_mode=False, data_prop=True)
    graph = model.graph

    # Build indexes
    nodes_by_output = {}
    for n in graph.node:
        for o in n.output:
            nodes_by_output[o] = n
    initializers = {t.name: t for t in graph.initializer}
    value_info = {
        vi.name: vi
        for vi in list(graph.value_info) + list(graph.input) + list(graph.output)
    }
    out_scale_log2 = build_output_scale_index(graph)

    layers = []
    relevant = {"Conv", "MatMul", "Gemm", "Softmax"}
    for n in graph.node:
        if n.op_type not in relevant:
            continue
        rec = extract_layer(n, (nodes_by_output, initializers), value_info, outdir)
        # Fill in output scale from forward-walk index
        rec.out_log2_scale = out_scale_log2.get(n.output[0])
        # Right-shift derivation:
        #   accum_int = Σ x_i8 * w_i8  (in (scale_in * scale_w) units)
        #   y_q = round(accum_int * scale_in * scale_w / scale_out)
        #   With pow2 scales: M = 2^(in_log + wt_log - out_log), typically < 1
        #   So shift_right = -log2(M) = out_log - in_log - wt_log
        # The kernel does `(sum + round) >> shift_right`, so this must be positive
        # for the canonical INT8 case (accum compresses down to INT8 range).
        if rec.op == "Softmax":
            # No multiply-accumulate; scales drive LUT/normalization, not a shift.
            pass
        else:
            # Conv/Gemm use wt scale; dynamic-operand MatMul uses in_b scale.
            b_log = (
                rec.wt_log2_scale
                if rec.wt_log2_scale is not None
                else rec.in_b_log2_scale
            )
            if (
                rec.in_log2_scale is not None
                and b_log is not None
                and rec.out_log2_scale is not None
            ):
                rec.right_shift = rec.out_log2_scale - rec.in_log2_scale - b_log
        layers.append(rec)

    # Manifest
    manifest = {
        "source": str(in_path),
        "deployment_shape": [4, 3, 512, 512],
        "weight_layout_default": "OIYXI8O8 (O,I divisible by 8) or OIYX_raw (otherwise)",
        "bias_format": "INT32, pre-promoted as (bias_i8 << pre_shift); use as accumulator init when weight_index==0",
        "layers": [asdict(L) for L in layers],
    }
    manifest_path = outdir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    # Summary
    print(f"\nExtracted {len(layers)} layers to {outdir}/")
    print(f"  Manifest: {manifest_path}")
    op_count = Counter(L.op for L in layers)
    print(f"  Op breakdown: {dict(op_count)}")
    layouts = Counter(L.weights_layout for L in layers)
    print(f"  Weight layouts: {dict(layouts)}")
    with_bias = sum(1 for L in layers if L.has_bias)
    print(f"  Layers with bias: {with_bias} / {len(layers)}")
    skipped = [L for L in layers if L.skipped_reason]
    if skipped:
        print(f"  Skipped (dynamic operands or non-pow2): {len(skipped)}")
        for L in skipped:
            print(f"    - {L.name}: {L.skipped_reason}")

    # Spot-check: print the first 5 layers' key fields
    print(f"\nFirst 5 layers:")
    for L in layers[:5]:
        print(f"  [{L.block}] {L.op} {L.name}")
        print(f"     wt: {L.weights_layout} {L.weights_shape}  → {L.weights_file}")
        if L.has_bias:
            print(
                f"     bias: INT32 pre_shift={L.bias_pre_shift} range={L.bias_int32_range}  → {L.bias_file}"
            )
        print(
            f"     scales: in=2^{L.in_log2_scale} wt=2^{L.wt_log2_scale} "
            f"out=2^{L.out_log2_scale}  right_shift={L.right_shift}"
        )


if __name__ == "__main__":
    main()
