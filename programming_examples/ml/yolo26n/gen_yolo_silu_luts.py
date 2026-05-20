"""Build per-Conv INT8 SiLU LUTs (256 bytes each) from the Quark XINT8 ONNX.

Each Conv has a `Conv→QL→DQ→HardSigmoid→Mul(const)→QL→DQ→Mul(linear)→QL`
chain in the ONNX with a predictable `<layer>/{conv,act}/...` node naming.
We extract the three QL scales + HardSigmoid params + mul_const, then
sample `hard_silu_multistep` over all 256 int8 inputs to build the LUT.

  - conv_stride (m0/m1/m3/m5/m7): one LUT at data/model.N/silu_lut.bin
  - c3k2_small (m2/m4): four LUTs at data/model.N/<layer>/silu_lut.bin
                        for layer in {cv1, m.0/cv1, m.0/cv2, cv2}
  - c3k2_heavy (m6/m8): nine LUTs per block (see C3K2_HEAVY_LAYERS).
  - PSA       (m9):     three LUTs per block at cv1, m/m.0/ffn/ffn.0, cv2.

The head block (m10) is not handled here; its LUT path is emitted
directly by gen_yolo_data.py.

Usage:
    python gen_yolo_silu_luts.py [path/to/model.onnx]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx

HERE = Path(__file__).resolve().parent

DEFAULT_ONNX = HERE / "models" / "phase1_25k_xint8_acc0.8968.onnx"
DATA_DIR = HERE / "data"


# --------------------------------------------------------------------------
# INT8 HardSiLU reference (matches the ONNX Quark XINT8 SiLU chain bit-exact).
# --------------------------------------------------------------------------
def _banker_quant(fp: np.ndarray, scale: float, dtype) -> np.ndarray:
    """Banker's rounding QuantizeLinear: round(fp/scale) clipped to dtype."""
    info = np.iinfo(np.dtype(dtype))
    rounded = np.rint(fp / scale)
    return np.clip(rounded, info.min, info.max).astype(dtype)


def hard_silu_multistep(
    linear_int8: np.ndarray,
    conv_out_scale: float,
    scaled_hsig_scale: float,
    scaled_hsig_dtype,
    post_silu_scale: float,
    alpha: float = 1.0 / 6.0,
    beta: float = 0.5,
    mul_const: float = 1.0001220703125,
) -> np.ndarray:
    """Two-QL chain — matches ONNX semantics (HardSigmoid + const-Mul fused
    in fp32, then quantized once, then multiplied by the dequantized input
    and quantized to the post-SiLU output scale)."""
    fp = linear_int8.astype(np.float32) * conv_out_scale
    scaled_fp = np.clip(alpha * fp + beta, 0.0, 1.0) * mul_const
    scaled_q = _banker_quant(scaled_fp, scaled_hsig_scale, scaled_hsig_dtype)
    scaled_dq = scaled_q.astype(np.float32) * scaled_hsig_scale
    silu_fp = fp * scaled_dq
    return _banker_quant(silu_fp, post_silu_scale, np.int8)


# Single-conv stride blocks: each has /model.N/conv + /model.N/act SiLU chain.
CONV_STRIDE_BLOCKS = ("m0", "m1", "m3", "m5", "m7")

# c3k2_small blocks (m2, m4): four conv+act chains per block, nested under
# /model.N/<layer>/{conv,act}/...
C3K2_SMALL_BLOCKS = ("m2", "m4")
C3K2_SMALL_LAYERS = ("cv1", "m.0/cv1", "m.0/cv2", "cv2")

# c3k2_heavy blocks (m6, m8): nine conv+act chains per block (outer cv1/cv2,
# m.0 parallel cv1/cv2, two inner_pair [cv1, cv2] residuals, m.0/cv3 fuse).
C3K2_HEAVY_BLOCKS = ("m6", "m8")
C3K2_HEAVY_LAYERS = (
    "cv1",
    "m.0/cv1",
    "m.0/cv2",
    "m.0/m/m.0/cv1",
    "m.0/m/m.0/cv2",
    "m.0/m/m.1/cv1",
    "m.0/m/m.1/cv2",
    "m.0/cv3",
    "cv2",
)

PSA_BLOCKS = ("m9",)
PSA_LAYERS = ("cv1", "m/m.0/ffn/ffn.0", "cv2")


def _block_to_n(block_name: str) -> int:
    """'m1' → 1, 'm10' → 10."""
    assert block_name.startswith("m")
    return int(block_name[1:])


def extract_silu_scales_for(graph, prefix: str) -> dict:
    """Pull the three QL scales + HardSigmoid params + mul_const out of any
    `<prefix>/{conv,act}/...` chain. `prefix` is a full path slice up to but
    not including `/conv` or `/act` — e.g. `/model.1`, `/model.2/cv1`,
    `/model.2/m.0/cv1`."""
    nodes_by_output = {o: n for n in graph.node for o in n.output}
    inits = {t.name: t for t in graph.initializer}

    def ql_scale_dtype(ql_node):
        scale = onnx.numpy_helper.to_array(inits[ql_node.input[1]]).item()
        if len(ql_node.input) >= 3:
            dtype_id = inits[ql_node.input[2]].data_type
        else:
            dtype_id = onnx.TensorProto.INT8
        dtype = {
            onnx.TensorProto.INT8: np.int8,
            onnx.TensorProto.UINT8: np.uint8,
        }.get(dtype_id, np.int8)
        return scale, dtype

    conv_ql = next(
        n for n in graph.node if n.name == f"{prefix}/conv/Conv_output_0_QuantizeLinear"
    )
    scaled_hsig_ql = next(
        n
        for n in graph.node
        if n.name == f"{prefix}/act/Sigmoid_output_0_QuantizeLinear"
    )
    silu_ql = next(
        n for n in graph.node if n.name == f"{prefix}/act/Mul_output_0_QuantizeLinear"
    )

    conv_scale, _ = ql_scale_dtype(conv_ql)
    scaled_hsig_scale, scaled_hsig_dt = ql_scale_dtype(scaled_hsig_ql)
    silu_scale, _ = ql_scale_dtype(silu_ql)

    hardsig_node = next(
        n
        for n in graph.node
        if n.op_type == "HardSigmoid" and n.name == f"{prefix}/act/Sigmoid"
    )
    alpha = next(a.f for a in hardsig_node.attribute if a.name == "alpha")
    beta = next((a.f for a in hardsig_node.attribute if a.name == "beta"), 0.5)

    mul_const_node = next(
        n for n in graph.node if n.name == f"{prefix}/act/Sigmoid_output_0_Mul"
    )
    const_node = nodes_by_output[mul_const_node.input[1]]
    mul_const = onnx.numpy_helper.to_array(
        next(a.t for a in const_node.attribute if a.name == "value")
    ).item()

    return {
        "conv_out": conv_scale,
        "scaled_hsig_out": scaled_hsig_scale,
        "scaled_hsig_dt": scaled_hsig_dt,
        "silu_out": silu_scale,
        "alpha": alpha,
        "beta": beta,
        "mul_const": mul_const,
    }


def build_lut_for(graph, prefix: str) -> tuple[np.ndarray, dict]:
    scales = extract_silu_scales_for(graph, prefix)
    v = np.arange(-128, 128, dtype=np.int8)
    lut = hard_silu_multistep(
        v,
        conv_out_scale=scales["conv_out"],
        scaled_hsig_scale=scales["scaled_hsig_out"],
        scaled_hsig_dtype=scales["scaled_hsig_dt"],
        post_silu_scale=scales["silu_out"],
        alpha=scales["alpha"],
        beta=scales["beta"],
        mul_const=scales["mul_const"],
    )
    return lut, scales


def main():
    onnx_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_ONNX
    print(f"Loading: {onnx_path}")
    m = onnx.load(str(onnx_path))

    inp = m.graph.input[0]
    inp.type.tensor_type.ClearField("shape")
    for d in (4, 3, 512, 512):
        inp.type.tensor_type.shape.dim.add().dim_value = d
    m = onnx.shape_inference.infer_shapes(m, strict_mode=False, data_prop=True)

    for block in CONV_STRIDE_BLOCKS:
        n = _block_to_n(block)
        lut, scales = build_lut_for(m.graph, f"/model.{n}")
        out_path = DATA_DIR / f"model.{n}" / "silu_lut.bin"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(lut.astype(np.int8).tobytes())
        print(
            f"  {block}: conv_out={scales['conv_out']} silu_out={scales['silu_out']}  "
            f"LUT[0]={int(lut[0])} LUT[128]={int(lut[128])} LUT[255]={int(lut[255])}  "
            f"-> {out_path}"
        )

    for block in C3K2_SMALL_BLOCKS:
        n = _block_to_n(block)
        for layer in C3K2_SMALL_LAYERS:
            prefix = f"/model.{n}/{layer}"
            lut, scales = build_lut_for(m.graph, prefix)
            out_path = DATA_DIR / f"model.{n}" / layer / "silu_lut.bin"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(lut.astype(np.int8).tobytes())
            print(
                f"  {block}/{layer}: conv_out={scales['conv_out']} silu_out={scales['silu_out']}  "
                f"LUT[0]={int(lut[0])} LUT[128]={int(lut[128])} LUT[255]={int(lut[255])}  "
                f"-> {out_path}"
            )

    for block in C3K2_HEAVY_BLOCKS:
        n = _block_to_n(block)
        for layer in C3K2_HEAVY_LAYERS:
            prefix = f"/model.{n}/{layer}"
            lut, scales = build_lut_for(m.graph, prefix)
            out_path = DATA_DIR / f"model.{n}" / layer / "silu_lut.bin"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(lut.astype(np.int8).tobytes())
            print(
                f"  {block}/{layer}: conv_out={scales['conv_out']} silu_out={scales['silu_out']}  "
                f"-> {out_path}"
            )

    for block in PSA_BLOCKS:
        n = _block_to_n(block)
        for layer in PSA_LAYERS:
            prefix = f"/model.{n}/{layer}"
            lut, scales = build_lut_for(m.graph, prefix)
            out_path = DATA_DIR / f"model.{n}" / layer / "silu_lut.bin"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(lut.astype(np.int8).tobytes())
            print(
                f"  {block}/{layer}: conv_out={scales['conv_out']} silu_out={scales['silu_out']}  "
                f"-> {out_path}"
            )

    # m10 (head) — single SiLU on the 1x1 256→1280 conv. The act chain
    # naming differs slightly (no `/cv*` parent), so build manually here
    # using the same multistep algorithm but with m10's chain prefix
    # `/model.10/conv`.
    lut10, scales10 = build_lut_for(m.graph, "/model.10/conv")
    out_path = DATA_DIR / "model.10" / "silu_lut.bin"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(lut10.astype(np.int8).tobytes())
    print(
        f"  m10/conv: conv_out={scales10['conv_out']} silu_out={scales10['silu_out']}  "
        f"-> {out_path}"
    )


if __name__ == "__main__":
    main()
