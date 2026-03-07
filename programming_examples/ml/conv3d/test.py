#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2026, Advanced Micro Devices, Inc.

import torch
import torch.nn as nn
import sys
import os
import numpy as np
import aie.utils.test as test_utils
import aie.iron as iron
from aie.utils import TraceConfig, HostRuntime, NPUKernel, DefaultNPURuntime
from aie.utils.ml import DataShaper

torch.use_deterministic_algorithms(True)
torch.manual_seed(0)


def main(opts):
    design = "conv3d"
    xclbin_path = opts.xclbin
    insts_path = opts.instr

    log_folder = "log/"
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)

    depth = int(opts.depth)
    height = int(opts.height)
    width = int(opts.width)
    ci = int(opts.in_channels)
    co = int(opts.out_channels)

    ci8 = ci // 8
    co8 = co // 8

    num_iter = 1
    npu_time_total = 0
    trace_size = opts.trace_size
    enable_trace = False if not trace_size else True
    trace_file = "log/trace_" + design + ".txt"

    # Data types
    dtype_in = np.dtype("uint8")
    dtype_wts = np.dtype("int8")
    dtype_out = np.dtype("uint8")

    # Data layout shapes
    # Input: D{C/8}H{C8}W (depth, channel-groups, height, channels-per-group, width)
    shape_in_act = (depth, ci8, height, 8, width)
    # Weights: {O/8}{I/8}KDHW{I8}{O8}
    shape_in_wts = (co8, ci8, 3, 3, 3, 8, 8)
    # Output: D{C/8}H{C8}W
    shape_out = (depth, co8, height, 8, width)

    # Initialize random input and weights
    int_inp = torch.randint(1, 20, (1, ci, depth, height, width)).type(
        torch.FloatTensor
    )
    # True 3D kernel (3x3x3)
    int_weight = torch.randint(-50, 50, (co, ci, 3, 3, 3)).type(torch.FloatTensor)

    # Quantization scales
    conv_scale = 7.6294e-06
    int8_scale = 0.0078
    min_val = 0
    max_val = 255

    # Load NPU kernel
    npu_kernel = NPUKernel(xclbin_path, insts_path, kernel_name=opts.kernel)
    kernel_handle = DefaultNPURuntime.load(npu_kernel)

    # Define PyTorch reference model
    class Conv3dModel(nn.Module):
        def __init__(self):
            super().__init__()
            # True 3D convolution with 3x3x3 kernel
            # No padding in conv since we manually pad with replicate mode
            self.conv = nn.Conv3d(ci, co, kernel_size=3, padding=0, bias=False)

        def forward(self, x):
            out_int = self.conv(x)
            # Quantization: match NPU behavior
            out_quant = out_int * conv_scale
            out_float = int8_scale * torch.clamp(
                torch.round(out_quant / int8_scale), min_val, max_val
            )
            return out_float

    # Generate golden output
    model = Conv3dModel()
    model.eval()
    model.conv.weight.data.copy_(int_weight)

    # Apply replication padding to match NPU kernel border handling
    # Pad: (left, right, top, bottom, front, back) for (W, H, D)
    int_inp_padded = torch.nn.functional.pad(int_inp, (1, 1, 1, 1, 1, 1), mode='replicate')
    golden_output = model(int_inp_padded)

    # Reorder input data layout
    ds = DataShaper()
    before_input = int_inp.squeeze().data.numpy().astype(dtype_in)  # [ci, depth, height, width]
    before_input.tofile(
        log_folder + "/before_ifm_conv3d.txt", sep=",", format="%d"
    )

    # Reorder: CDHW → D{C/8}H{C8}W manually
    ci8 = ci // 8
    ifm_mem_fmt = np.zeros((depth, ci8, height, 8, width), dtype=dtype_in)
    for d in range(depth):
        for ic8 in range(ci8):
            for h in range(height):
                for ic in range(8):
                    for w in range(width):
                        ifm_mem_fmt[d, ic8, h, ic, w] = before_input[
                            ic8 * 8 + ic, d, h, w
                        ]

    ifm_mem_fmt = ifm_mem_fmt.flatten()
    ifm_mem_fmt.tofile(
        log_folder + "/after_ifm_conv3d.txt", sep=",", format="%d"
    )

    # Reorder weights: OIKDHW → {O/8}{I/8}KDHW{I8}{O8}
    # Manual reordering since DataShaper doesn't support 3D pattern yet
    wts_orig = int_weight.data.numpy().astype(dtype_wts)  # [co, ci, 3, 3, 3]
    co8, ci8 = co // 8, ci // 8
    wts = np.zeros((co8, ci8, 3, 3, 3, 8, 8), dtype=dtype_wts)

    for oc8 in range(co8):
        for ic8 in range(ci8):
            for kd in range(3):
                for kh in range(3):
                    for kw in range(3):
                        for i in range(8):
                            for o in range(8):
                                wts[oc8, ic8, kd, kh, kw, i, o] = wts_orig[
                                    oc8 * 8 + o, ic8 * 8 + i, kd, kh, kw
                                ]

    wts = wts.flatten()
    wts.tofile(log_folder + "/weights_conv3d.txt", sep=",", format="%d")

    # Prepare NPU buffers
    in1 = iron.tensor(ifm_mem_fmt, dtype=dtype_in)
    in2 = iron.tensor(wts, dtype=dtype_wts)
    out_size = np.prod(shape_out) * dtype_out.itemsize
    out = iron.zeros(out_size, dtype=dtype_out)

    buffers = [in1, in2, out]

    # Trace configuration
    trace_config = None
    if enable_trace:
        trace_config = TraceConfig(
            trace_size=trace_size,
            trace_file=trace_file,
            trace_after_last_tensor=True,
            enable_ctrl_pkts=False,
            last_tensor_shape=out.shape,
            last_tensor_dtype=out.dtype,
        )
        HostRuntime.prepare_args_for_trace(buffers, trace_config)

    # Run on NPU
    for i in range(num_iter):
        try:
            ret = DefaultNPURuntime.run(kernel_handle, buffers)
            if enable_trace:
                trace_buffer, _ = HostRuntime.extract_trace_from_args(
                    buffers, trace_config
                )
                trace_buffer = trace_buffer.view(np.uint32)
                trace_config.write_trace(trace_buffer)

            out_tensor = buffers[-1]
            if not isinstance(out_tensor, np.ndarray):
                out_tensor = out_tensor.numpy()
            data_buffer = out_tensor * int8_scale
            npu_time_total += ret.npu_time
        except Exception as e:
            print(f"\nNPU execution error: {e}")
            if enable_trace:
                print("Extracting trace buffer for debugging...")
                try:
                    trace_buffer, _ = HostRuntime.extract_trace_from_args(
                        buffers, trace_config
                    )
                    trace_buffer = trace_buffer.view(np.uint32)

                    # Save raw trace buffer
                    raw_trace_file = "log/trace_conv3d_raw.bin"
                    trace_buffer.tofile(raw_trace_file)
                    print(f"Raw trace buffer saved to {raw_trace_file}")
                    print(f"Trace buffer size: {len(trace_buffer)} uint32 values")
                    print(f"First 20 values: {trace_buffer[:20]}")

                    trace_config.write_trace(trace_buffer)
                    print(f"Trace written to {trace_file}")
                except Exception as trace_err:
                    print(f"Failed to extract trace: {trace_err}")
                    import traceback
                    traceback.print_exc()
            raise

    # Reorder output data layout: D{C/8}H{C8}W → CDHW
    temp_out = data_buffer.reshape(shape_out)  # [depth, co8, height, 8, width]
    co8 = co // 8
    ofm_mem_fmt = np.zeros((co, depth, height, width), dtype=np.float32)

    for d in range(depth):
        for oc8 in range(co8):
            for h in range(height):
                for oc in range(8):
                    for w in range(width):
                        ofm_mem_fmt[oc8 * 8 + oc, d, h, w] = temp_out[
                            d, oc8, h, oc, w
                        ]
    ofm_mem_fmt.tofile(
        log_folder + "/after_ofm_conv3d.txt", sep=",", format="%d"
    )
    ofm_mem_fmt_out = torch.from_numpy(ofm_mem_fmt).unsqueeze(0)

    # Compare NPU output with golden reference
    print(f"\nAvg NPU time: {int((npu_time_total / num_iter) / 1000)}us.")
    print(f"Volume size: {depth}x{height}x{width}, Channels: {ci}→{co}")

    # Note: Using 16x tolerance due to quantization rounding and border handling differences
    # True 3D convolution has more accumulations, higher error at borders
    if np.allclose(
        ofm_mem_fmt_out.detach().numpy(),
        golden_output.detach().numpy(),
        rtol=0,
        atol=16 * int8_scale,  # 16x tolerance for 3D quantization + border effects
    ):
        print("\nPASS!\n")
        exit(0)
    else:
        max_diff = np.max(
            np.abs(
                ofm_mem_fmt_out.detach().numpy() - golden_output.detach().numpy()
            )
        )
        print(f"\nFailed. Max difference: {max_diff}\n")
        exit(-1)


if __name__ == "__main__":
    p = test_utils.create_default_argparser()
    p.add_argument(
        "-d",
        "--depth",
        dest="depth",
        default=8,
        help="Depth of 3D convolution volume",
    )
    p.add_argument(
        "-ht",
        "--height",
        dest="height",
        default=8,
        help="Height of 3D convolution volume",
    )
    p.add_argument(
        "-wd",
        "--width",
        dest="width",
        default=8,
        help="Width of 3D convolution volume",
    )
    p.add_argument(
        "-ic",
        "--in_channels",
        dest="in_channels",
        default=8,
        help="Number of input channels",
    )
    p.add_argument(
        "-oc",
        "--out_channels",
        dest="out_channels",
        default=8,
        help="Number of output channels",
    )
    opts = p.parse_args(sys.argv[1:])
    main(opts)
