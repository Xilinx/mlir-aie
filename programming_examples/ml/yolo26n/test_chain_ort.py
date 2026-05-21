"""End-to-end chain NPU-vs-ORT bit-exact check for the full m0..m10 chain.

Feeds the same (4, 3, 512, 512) RGB int8 input that test_block_ort.py uses,
runs ORT to get the chain's final tensor (post-SiLU int8 for non-head tails,
fp32 softmax for the m10 head), then runs the chain xclbin on the NPU and
bit-exact compares.

For an N>1 chain (built with CHAIN_N_SAMPLES=N at MLIR-generation time), set
the same CHAIN_N_SAMPLES env var here and the script feeds N identical copies
of the test image, then verifies every output slice matches the single ORT
reference.

Run:
    make run_chain                          # N=1
    CHAIN_N_SAMPLES=15 make run_chain       # N=15 (also rebuilds xclbin)
    python3 test_chain_ort.py -x build/final_chain.xclbin \\
                              -i build/insts_chain.bin -k MLIR_AIE
"""

from __future__ import annotations

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


# Chain endpoints — must match aie2_yolo_iron_partial.CHAIN_BLOCKS.
LAST_BLOCK = "m10"
LAST_N = int(LAST_BLOCK[1:])


def get_input_qparams(graph):
    inits = {t.name: t for t in graph.initializer}
    in_name = graph.input[0].name
    ql = next(
        n for n in graph.node if n.op_type == "QuantizeLinear" and n.input[0] == in_name
    )
    scale = float(onnx.numpy_helper.to_array(inits[ql.input[1]]).item())
    return scale


def main():
    p = test_utils.create_default_argparser()
    opts = p.parse_args(sys.argv[1:])

    m0 = yolo_spec.block("m0")
    in_w, in_h, in_c_decl = m0.layers[0].in_shape  # 512x512x3
    last = yolo_spec.block(LAST_BLOCK)
    last_out_shape = last.layers[-1].out_shape
    if last.topology == "head":
        # m10 emits (out_c,) flat probs padded to 4 bytes for shim alignment.
        out_c = last_out_shape[0]
        OUT_PAD = 4
        out_total = OUT_PAD
    else:
        out_w, out_h, out_c = last_out_shape
        out_total = out_h * out_w * out_c
    in_c_pad = 8  # m0 pads RGB to 8

    m = onnx.load(str(ONNX_PATH))
    inp = m.graph.input[0]
    inp.type.tensor_type.ClearField("shape")
    for d in (4, 3, 512, 512):
        inp.type.tensor_type.shape.dim.add().dim_value = d
    m = onnx.shape_inference.infer_shapes(m, strict_mode=False, data_prop=True)
    in_scale = get_input_qparams(m.graph)

    last_topo = yolo_spec.block(LAST_BLOCK).topology
    if last_topo == "conv_stride":
        out_tensor_name = f"/model.{LAST_N}/act/Mul_output_0_QuantizeLinear_Output"
    elif last_topo in ("c3k2_small", "c3k2_heavy"):
        out_tensor_name = f"/model.{LAST_N}/cv2/act/Mul_output_0_QuantizeLinear_Output"
    elif last_topo == "psa":
        # m9's final SiLU is also on cv2/act.
        out_tensor_name = f"/model.{LAST_N}/cv2/act/Mul_output_0_QuantizeLinear_Output"
    elif last_topo == "head":
        # m10's softmax in ONNX is fp32 (not quantized). The graph's
        # `output0` IS the fp softmax. We dequantize the NPU's i8 @ 2^-7
        # for the comparison.
        out_tensor_name = "output0"
    else:
        raise ValueError(f"unsupported chain tail topology {last_topo!r}")
    print(f"Chain output tensor: {out_tensor_name!r}")

    _expose_quantize_outputs(m)
    sess = ort.InferenceSession(m.SerializeToString())

    rng = np.random.default_rng(seed=0)
    rgb_hwc = rng.integers(-128, 128, size=(in_h, in_w, in_c_decl), dtype=np.int8)
    nchw_int8 = np.repeat(rgb_hwc.transpose(2, 0, 1)[None], 4, axis=0)
    in_fp = nchw_int8.astype(np.float32) * in_scale

    print("Running ORT...")
    (ort_out,) = sess.run(
        [out_tensor_name],
        {sess.get_inputs()[0].name: in_fp},
    )
    if last_topo == "head":
        # ORT softmax output (4, out_c) fp32 → batch 0 → quantize to
        # i8 at 2^-7 to match NPU's quantized probs.
        out_scale_inv = 2.0**7  # 128
        expected = np.clip(np.rint(ort_out[0] * out_scale_inv), 0, 127).astype(np.int8)
    else:
        assert ort_out.shape == (4, out_c, out_h, out_w), ort_out.shape
        expected = ort_out[0].transpose(1, 2, 0).astype(np.int8)

    # CHAIN_N_SAMPLES must match what the xclbin was built with.
    n_samples = int(os.environ.get("CHAIN_N_SAMPLES", "1"))

    # NPU input: HWC with channels padded 3→8, then tiled N_SAMPLES times.
    in_padded = np.zeros((in_h, in_w, in_c_pad), dtype=np.int8)
    in_padded[:, :, :in_c_decl] = rgb_hwc
    in_flat = np.tile(in_padded.reshape(-1), n_samples)

    in_tensor_iron = iron.tensor(in_flat, dtype=np.int8)
    out_tensor_iron = iron.zeros([n_samples * out_total], dtype=np.int8)

    npu_opts = test_utils.create_npu_kernel(opts)
    print(f"Running NPU chain (N={n_samples})...")
    # Call rt.load+run directly (instead of run_test) so we get the
    # XRTKernelResult and can read its NPU-reported execution time.
    # NB: re-running rt.run in the same process currently hangs (XRT
    # context appears wedged after the first invocation) — one timing
    # per process; benchmark via scripts/time_chain.py.
    rt = DefaultNPURuntime
    handle = rt.load(npu_opts.npu_kernel)
    result = rt.run(handle, [in_tensor_iron, out_tensor_iron])
    print(
        f"  NPU compute time: {result.npu_time / 1e6:.3f} ms total "
        f"= {result.npu_time / 1e6 / n_samples:.3f} ms/sample "
        f"-> {1e9 / (result.npu_time / n_samples):.2f} fps"
    )
    rc = 0 if result.ret.name == "ERT_CMD_STATE_COMPLETED" else 1
    if rc != 0:
        return rc

    # Verify every output slice matches the single ORT reference (all input
    # samples are identical, so all outputs should be identical too).
    out_tensor_iron.to("cpu")
    out_np = out_tensor_iron.numpy()
    total_diff = 0
    max_abs = 0
    first_bad_sample = None
    for i in range(n_samples):
        sample_bytes = out_np[i * out_total : (i + 1) * out_total]
        if last_topo == "head":
            actual = sample_bytes[:out_c]
        else:
            actual = sample_bytes.reshape(out_h, out_w, out_c)
        diff = actual.astype(np.int16) - expected.astype(np.int16)
        n_diff = int((diff != 0).sum())
        if n_diff:
            total_diff += n_diff
            max_abs = max(max_abs, int(np.abs(diff).max()))
            if first_bad_sample is None:
                first_bad_sample = (i, actual, diff)
    print(
        f"chain m0..{LAST_BLOCK} N={n_samples} vs ORT: mismatches={total_diff}/"
        f"{n_samples * expected.size}, max|diff|={max_abs}"
    )
    if total_diff == 0:
        print(
            f"BIT-EXACT NPU chain == ORT for {out_tensor_name} "
            f"(all {n_samples} sample{'s' if n_samples != 1 else ''})"
        )
        return 0
    i, actual, diff = first_bad_sample
    print(f"First diff is in sample {i}:")
    if last_topo == "head":
        print(f"  npu={actual.tolist()}  ort={expected.tolist()}")
    else:
        print("First 8 diffs:")
        for y, x, c in np.argwhere(diff != 0)[:8]:
            print(f"  [{y},{x},{c}]: npu={actual[y, x, c]}, ort={expected[y, x, c]}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
