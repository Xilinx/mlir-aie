"""NPU-vs-ORT bit-exact compare for any conv_stride / c3k2 / psa block.

For block m_N:
  - Seed-0 random INT8 image (3,512,512), dequantized via the model's input QL
    scale, fed to ORT at batch=4 (m9 PSA Reshape requires that).
  - ORT returns:
      - the int8 input tensor of /model.N/conv (== expected NPU input)
      - the post-SiLU int8 output of /model.N/act (== expected NPU output)
  - The NPU xclbin is run with the same input, then compared.

For m9 (psa), the M9_STAGE env var selects which intermediate ORT tensor to
compare:
  - M9_STAGE=1: cv1 post-SiLU (smallest possible test; isolates cv1).
  - M9_STAGE=10 (default): cv2 post-SiLU (full m9 block output).
Other m9 stages emit intermediate PSA tensors (qkv pack, scores, etc.) that
don't have clean ORT analogs at the same scale — those are exercised
end-to-end via `make run_chain`.

Bit-exact match means the NPU pipeline (conv + bias accum + banker requant +
SiLU LUT) matches the Quark XINT8 graph at this block to the last bit.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort

import aie.iron as iron
import aie.utils.test as test_utils
from aie.utils import DefaultNPURuntime

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import yolo_spec  # noqa: E402

ONNX_PATH = HERE / "models" / "phase1_25k_xint8_acc0.8968.onnx"


def _expose_quantize_outputs(model):
    """Add every QuantizeLinear output to the model's outputs so ORT returns
    them all — lets us pull any block's input/output int8 tensor directly."""
    existing = {o.name for o in model.graph.output}
    for n in model.graph.node:
        if n.op_type != "QuantizeLinear":
            continue
        out_name = n.output[0]
        if out_name in existing:
            continue
        vi = onnx.helper.make_tensor_value_info(out_name, onnx.TensorProto.INT8, None)
        model.graph.output.append(vi)
    return model


def get_input_qparams(graph):
    inits = {t.name: t for t in graph.initializer}
    in_name = graph.input[0].name
    ql = next(
        n for n in graph.node if n.op_type == "QuantizeLinear" and n.input[0] == in_name
    )
    scale = float(onnx.numpy_helper.to_array(inits[ql.input[1]]).item())
    zp = (
        int(onnx.numpy_helper.to_array(inits[ql.input[2]]).item())
        if len(ql.input) >= 3
        else 0
    )
    return scale, zp


def find_block_io_tensors(
    graph, model_n: int, topology: str, stage: int = 0
) -> tuple[str, str]:
    """Returns (input_int8_tensor, output_int8_tensor) names for block m_N.

    For conv_stride blocks (m0/m1/m3/m5/m7):
      Input: int8 feeding /model.N/conv/Conv.
      Output: /model.N/act/Mul_output_0_QuantizeLinear_Output (post-SiLU).

    For c3k2_small blocks (m2/m4):
      Input: int8 feeding /model.N/cv1/conv/Conv.
      Output: /model.N/cv2/act/Mul_output_0_QuantizeLinear_Output (post-SiLU,
              the block's effective output — what downstream blocks see).

    For psa blocks (m9), the `stage` arg selects the output tensor:
      stage=1  -> post-SiLU of cv1 (isolates the cv1 1x1+SiLU).
      stage=10 -> post-SiLU of cv2 (full m9 block output).
      Other stages have no clean single-tensor ORT analog (intermediate PSA
      tensors are emitted at different scales / layouts); use `make run_chain`
      to verify those end-to-end.
    """
    nodes_by_output = {o: n for n in graph.node for o in n.output}

    if topology == "conv_stride":
        first_conv_name = f"/model.{model_n}/conv/Conv"
        out_tensor = f"/model.{model_n}/act/Mul_output_0_QuantizeLinear_Output"
    elif topology in ("c3k2_small", "c3k2_heavy"):
        # Both c3k2 variants emit post-SiLU cv2 output as the block result.
        first_conv_name = f"/model.{model_n}/cv1/conv/Conv"
        out_tensor = f"/model.{model_n}/cv2/act/Mul_output_0_QuantizeLinear_Output"
    elif topology == "psa":
        first_conv_name = f"/model.{model_n}/cv1/conv/Conv"
        if stage == 1:
            out_tensor = f"/model.{model_n}/cv1/act/Mul_output_0_QuantizeLinear_Output"
        elif stage == 10:
            out_tensor = f"/model.{model_n}/cv2/act/Mul_output_0_QuantizeLinear_Output"
        else:
            raise ValueError(
                f"psa block: M9_STAGE={stage} has no single-tensor ORT analog "
                "(only stage 1 = cv1 act, stage 10 = cv2 act are supported here). "
                "Use `make run_chain` for end-to-end verification of other stages."
            )
    else:
        raise ValueError(f"unsupported topology {topology!r}")

    conv = next(n for n in graph.node if n.name == first_conv_name)
    dq = nodes_by_output[conv.input[0]]
    assert dq.op_type == "DequantizeLinear", f"unexpected upstream {dq.op_type}"
    ql = nodes_by_output[dq.input[0]]
    assert ql.op_type == "QuantizeLinear", f"unexpected upstream of DQ: {ql.op_type}"
    in_tensor = f"{ql.name}_Output"
    return in_tensor, out_tensor


def main():
    p = test_utils.create_default_argparser()
    p.add_argument(
        "--block",
        required=True,
        help="block name (m0/m1/m2/m3/m4/m5/m6/m7/m8/m9). m9 also reads M9_STAGE "
        "env var (1 = cv1 only, 10 = full block; default 10). m10 is verified "
        "via run_chain only -- its softmax output is 1D (2,) and not bit-exact "
        "comparable here.",
    )
    opts = p.parse_args(sys.argv[1:])
    model_n = int(opts.block[1:])

    blk = yolo_spec.block(opts.block)
    in_w, in_h, in_c_decl = blk.layers[0].in_shape

    # For m9 PSA, M9_STAGE selects which intermediate layer's output we compare
    # against ORT. Stage 1 = cv1 only (first layer), stage 10 = full block (cv2,
    # last layer). The same env var drives the build via per_block_iron().
    m9_stage = int(os.environ.get("M9_STAGE", "10"))

    # Block output shape: depends on topology + (for psa) which stage we're at.
    # m10 (head) softmax output is 1D (2,) -- bit-exact compare is out of scope
    # here (full-chain test_chain_ort.py covers it). Bail with a friendly note.
    if opts.block == "m10":
        print(
            f"Block m10 standalone: output is 1D softmax(2,) -- bit-exact "
            f"verification is via run_chain, not run_ort. Use `make BLOCK=m10 "
            f"time` for timing only."
        )
        return 0
    if blk.topology == "conv_stride":
        out_w, out_h, out_c = blk.layers[0].out_shape
    elif blk.topology == "psa":
        if m9_stage == 1:
            out_w, out_h, out_c = blk.layers[0].out_shape  # cv1
        else:
            out_w, out_h, out_c = blk.layers[-1].out_shape  # cv2 (stage 10)
    else:
        out_w, out_h, out_c = blk.layers[-1].out_shape
    in_c_pad = 8 if opts.block == "m0" else in_c_decl  # m0 pads RGB 3→8

    # ----- Load ONNX, find io tensors, run ORT -----
    m = onnx.load(str(ONNX_PATH))
    inp = m.graph.input[0]
    inp.type.tensor_type.ClearField("shape")
    for d in (4, 3, 512, 512):
        inp.type.tensor_type.shape.dim.add().dim_value = d
    m = onnx.shape_inference.infer_shapes(m, strict_mode=False, data_prop=True)
    in_scale, in_zp = get_input_qparams(m.graph)
    assert in_zp == 0

    in_tensor_name, out_tensor_name = find_block_io_tensors(
        m.graph, model_n, blk.topology, stage=m9_stage
    )
    print(
        f"Block {opts.block}: input tensor {in_tensor_name!r}, output tensor {out_tensor_name!r}"
    )

    _expose_quantize_outputs(m)
    sess = ort.InferenceSession(m.SerializeToString())

    rng = np.random.default_rng(seed=0)
    rgb_hwc = rng.integers(-128, 128, size=(512, 512, 3), dtype=np.int8)
    nchw_int8 = np.repeat(rgb_hwc.transpose(2, 0, 1)[None], 4, axis=0)
    in_fp = nchw_int8.astype(np.float32) * in_scale

    print("Running ORT...")
    ort_in_nchw, ort_out_nchw = sess.run(
        [in_tensor_name, out_tensor_name],
        {sess.get_inputs()[0].name: in_fp},
    )
    assert ort_out_nchw.shape == (4, out_c, out_h, out_w), ort_out_nchw.shape
    expected = (
        ort_out_nchw[0].transpose(1, 2, 0).astype(np.int8)
    )  # (out_h, out_w, out_c)

    # Build NPU input matching ORT's int8 input tensor for this block.
    # ORT input shape: (4, C_decl, H, W) → take sample 0, transpose to HWC,
    # pad channels if necessary (m0 only).
    ort_in_hwc = ort_in_nchw[0].transpose(1, 2, 0).astype(np.int8)
    assert ort_in_hwc.shape == (in_h, in_w, in_c_decl), ort_in_hwc.shape
    if in_c_pad != in_c_decl:
        in_padded = np.zeros((in_h, in_w, in_c_pad), dtype=np.int8)
        in_padded[:, :, :in_c_decl] = ort_in_hwc
    else:
        in_padded = ort_in_hwc

    in_tensor_iron = iron.tensor(in_padded.reshape(-1), dtype=np.int8)
    out_tensor_iron = iron.zeros([out_h * out_w * out_c], dtype=np.int8)

    npu_opts = test_utils.create_npu_kernel(opts)
    print("Running NPU...")
    rc = DefaultNPURuntime.run_test(
        npu_opts.npu_kernel,
        [in_tensor_iron, out_tensor_iron],
        {},
        verify=False,
        verbosity=opts.verbosity,
    )
    if rc != 0:
        return rc

    out_tensor_iron.to("cpu")
    actual = out_tensor_iron.numpy().reshape(out_h, out_w, out_c)

    diff = actual.astype(np.int16) - expected.astype(np.int16)
    n_diff = int((diff != 0).sum())
    print(
        f"NPU vs ORT post-SiLU ({opts.block}): mismatches={n_diff}/{expected.size}, "
        f"max|diff|={int(np.abs(diff).max()) if n_diff else 0}"
    )
    if n_diff == 0:
        print(f"BIT-EXACT NPU == ORT for {out_tensor_name}")
        return 0
    print("First 8 diffs:")
    for y, x, c in np.argwhere(diff != 0)[:8]:
        print(f"  [{y},{x},{c}]: npu={actual[y, x, c]}, ort={expected[y, x, c]}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
