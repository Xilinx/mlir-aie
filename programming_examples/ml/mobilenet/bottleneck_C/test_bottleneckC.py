#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2024, Advanced Micro Devices, Inc.

import torch
import torch.nn as nn
import sys
import math
from aie.utils.ml import DataShaper
import time
import os
import numpy as np
import aie.iron as iron
from aie.utils import DefaultNPURuntime
from aie.utils import TraceConfig, HostRuntime, NPUKernel, DefaultNPURuntime
import aie.utils.test as test_utils
import json


def convert_to_numpy(array):
    if isinstance(array, np.ndarray):
        return array
    elif isinstance(array, torch.Tensor):
        return array.cpu().numpy()
    else:
        raise TypeError("Unsupported array type")


# Function to read scale factors from JSON file
def read_scale_factors(file_path):
    with open(file_path, "r") as file:
        return json.load(file)


log_dir = "log/"
data_dir = "data/"

# Read the existing scale factors
scale_factor_file = "scale_factors.json"
scale_factors = read_scale_factors(data_dir + scale_factor_file)

torch.use_deterministic_algorithms(True)
torch.manual_seed(0)
vectorSize = 8

bneck_13_InW1 = 7
bneck_13_InH1 = 7
bneck_13_InC1 = 80
bneck_13_OutC1 = 960
WeightChunks = 2  # 2 splits for input channel and then output

bneck_13_InW2 = bneck_13_InW1
bneck_13_InH2 = bneck_13_InH1
bneck_13_OutC2 = bneck_13_OutC1

bneck_13_InW3 = bneck_13_InW1
bneck_13_InH3 = bneck_13_InH1
bneck_13_OutC3 = 80

bneck_13_InC1_vec = math.floor(bneck_13_InC1 / vectorSize)
bneck_13_OutC3_vec = math.floor(bneck_13_OutC3 / vectorSize)

tensorInH = bneck_13_InH1
tensorInW = bneck_13_InW1
tensorInC = bneck_13_InC1


def chunk_weights_depth_cascade(int_weight, InC, WeightChunks):
    chunk_size = InC // WeightChunks
    chunks = []
    input_channels = int_weight.shape[1]
    output_channels = int_weight.shape[0]

    for i in range(WeightChunks):
        start_index = i * chunk_size
        end_index = input_channels if i == WeightChunks - 1 else (i + 1) * chunk_size
        for out_c_start in range(0, output_channels, 8):
            out_c_end = min(out_c_start + 8, output_channels)
            chunk = int_weight[out_c_start:out_c_end, start_index:end_index, :, :]
            # print("oc={}:{},ic={}:{}".format(out_c_start,out_c_end,start_index,end_index))
            chunks.append(chunk)
    return chunks


def reorder_and_concatenate_chunks(int_weight, InC, WeightChunks, ds, dtype_wts):
    # Chunk the weights
    chunks = chunk_weights_depth_cascade(int_weight, InC, WeightChunks)

    # Reorder each chunk
    reordered_chunks = []
    for idx, chunk in enumerate(chunks):
        reordered_chunk = ds.reorder_mat(
            chunk.data.numpy().astype(dtype_wts), "OIYXI8O8", "OIYX"
        )
        reordered_chunks.append(reordered_chunk)

    # Concatenate the reordered chunks
    total_wts = np.concatenate(reordered_chunks, axis=None)
    print(int_weight.shape)
    print(total_wts.shape)

    return total_wts


# wts_size=(bneck_13_OutC1*bneck_13_InC1)
wts_size = 2 * ((bneck_13_OutC1 * bneck_13_InC1) + (bneck_13_OutC2 * bneck_13_OutC3))


def main(opts):
    design = "mobilenet_bottleneck_C"
    xclbin_path = opts.xclbin
    insts_path = opts.instr

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    num_iter = 1
    npu_time_total = 0
    npu_time_min = 9999999
    npu_time_max = 0
    enable_trace = True if opts.trace_size > 0 else False
    print("trace_size: ", opts.trace_size)
    trace_file = "log/trace_" + design + ".txt"
    # ------------------------------------------------------
    # Configure this to match your design's buffer size
    # ------------------------------------------------------
    dtype_in = np.dtype("int8")
    dtype_wts = np.dtype("int8")
    dtype_out = np.dtype("int8")
    print(wts_size)

    shape_total_wts = (wts_size, 1)
    shape_in_act = (
        tensorInH,
        bneck_13_InC1_vec,
        tensorInW,
        vectorSize,
    )  #'YCXC8' , 'CYX'
    shape_out = (
        bneck_13_InH1,
        bneck_13_OutC3_vec,
        bneck_13_InW1,
        vectorSize,
    )  # bneck_12_OutC3/8
    size_out = bneck_13_InH1 * bneck_13_OutC3_vec * bneck_13_InW1 * vectorSize
    shape_out_final = (
        bneck_13_OutC3_vec * vectorSize,
        bneck_13_InH1,
        bneck_13_InW1,
    )  # bneck_12_OutC3/8

    # ------------------------------------------------------
    # Initialize activation, weights, scaling factor for int8 model
    # ------------------------------------------------------
    input = torch.randn(1, bneck_13_InC1_vec * vectorSize, bneck_13_InH1, bneck_13_InW1)
    # ------------------------------------------------------
    # Get device, load the xclbin & kernel and register them
    # ------------------------------------------------------
    npu_kernel = NPUKernel(xclbin_path, insts_path)
    kernel_handle = DefaultNPURuntime.load(npu_kernel)

    golden_output = np.loadtxt(
        data_dir + "golden_output.txt", delimiter=",", dtype="int8"
    )
    golden_output = golden_output.reshape(
        1, bneck_13_InC1, bneck_13_InH1, bneck_13_InW1
    )

    ds = DataShaper()

    before_input = np.loadtxt(
        data_dir + "before_ifm_mem_fmt_1x1.txt", delimiter=",", dtype="int8"
    )
    before_input = before_input.reshape(bneck_13_InC1, bneck_13_InH1, bneck_13_InW1)
    if bneck_13_InW1 > 1 and bneck_13_InH1 == 1:
        ifm_mem_fmt = ds.reorder_mat(before_input, "CXC8", "CX")
    elif bneck_13_InW1 > 1 and bneck_13_InH1 > 1:
        ifm_mem_fmt = ds.reorder_mat(before_input, "YCXC8", "CYX")

    else:
        ifm_mem_fmt = ds.reorder_mat(before_input, "CC8", "C")

    bn13_wts1 = np.loadtxt(data_dir + "bn13_1_chain.txt", delimiter=",", dtype="int32")
    bn13_wts2 = np.loadtxt(data_dir + "bn13_2_chain.txt", delimiter=",", dtype="int32")
    bn13_wts3_put = np.loadtxt(
        data_dir + "bn13_3_put_chain.txt", delimiter=",", dtype="int32"
    )
    bn13_wts3_get = np.loadtxt(
        data_dir + "bn13_3_get_chain.txt", delimiter=",", dtype="int32"
    )

    bn14_wts1 = np.loadtxt(data_dir + "bn14_1_chain.txt", delimiter=",", dtype="int32")
    bn14_wts2 = np.loadtxt(data_dir + "bn14_2_chain.txt", delimiter=",", dtype="int32")
    bn14_wts3_put = np.loadtxt(
        data_dir + "bn14_3_put_chain.txt", delimiter=",", dtype="int32"
    )
    bn14_wts3_get = np.loadtxt(
        data_dir + "bn14_3_get_chain.txt", delimiter=",", dtype="int32"
    )

    total_wts = np.concatenate(
        (
            bn13_wts1,
            bn13_wts3_put,
            bn13_wts3_get,
            bn14_wts1,
            bn14_wts3_put,
            bn14_wts3_get,
        ),
        axis=None,
    )

    total_wts.tofile(log_dir + "after_weights_mem_fmt_final.txt", sep=",", format="%d")
    print(total_wts.shape)

    # ------------------------------------------------------
    # Setup buffers run loop
    # ------------------------------------------------------
    in1 = iron.tensor(ifm_mem_fmt, dtype=dtype_in)
    in2 = iron.tensor(total_wts, dtype=dtype_wts)
    out = iron.zeros(size_out, dtype=dtype_out)
    buffers = [in1, in2, out]

    print("orig shape_out: ", shape_out)
    print("orig shape_size: ", size_out)
    print("in1 size: ", ifm_mem_fmt.shape)
    print("in2 size: ", total_wts.shape)

    trace_config = None
    if enable_trace:
        trace_config = TraceConfig(
            trace_size=opts.trace_size,
            trace_file=trace_file,
            trace_after_last_tensor=True,
            enable_ctrl_pkts=False,
            last_tensor_shape=out.shape,
            last_tensor_dtype=out.dtype,
        )
        HostRuntime.prepare_args_for_trace(buffers, trace_config)

    # ------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------
    for i in range(num_iter):
        start = time.time_ns()
        # aie_output = execute(app, ifm_mem_fmt, total_wts)
        DefaultNPURuntime.run(kernel_handle, buffers)
        full_output = out.numpy()
        stop = time.time_ns()
        npu_time = stop - start
        npu_time_total = npu_time_total + npu_time

        print("full_output shape: ", full_output.shape)
        aie_output = full_output[:size_out].view(np.int8)
        print("aie_output shape: ", aie_output.shape)
        # if i == 0:
        if enable_trace:
            # if i == 9:
            if i == 0:
                trace_buffer = full_output[size_out:].view(np.uint32)
    # ------------------------------------------------------
    # Reorder output data-layout
    # ------------------------------------------------------
    temp_out = aie_output.reshape(shape_out)
    temp_out = ds.reorder_mat(temp_out, "CDYX", "YCXD")
    ofm_mem_fmt = temp_out.reshape(shape_out_final)
    ofm_mem_fmt.tofile(log_dir + "after_ofm_mem_fmt_final.txt", sep=",", format="%d")
    ofm_mem_fmt_out = torch.from_numpy(ofm_mem_fmt).unsqueeze(0)
    print("Golden::Brevitas::", golden_output)
    print("AIE::", ofm_mem_fmt_out)
    # ------------------------------------------------------
    # Compare the AIE output and the golden reference
    # ------------------------------------------------------
    print("\nAvg NPU time: {}us.".format(int((npu_time_total / num_iter) / 1000)))
    golden = convert_to_numpy(golden_output)
    ofm_mem_fmt_out = convert_to_numpy(ofm_mem_fmt_out)
    max_difference = np.max((golden) - (ofm_mem_fmt_out))
    print("max_difference:", max_difference)
    if enable_trace:
        # trace_buffer = full_output[3920:]
        print("trace_buffer shape: ", trace_buffer.shape)
        print("trace_buffer dtype: ", trace_buffer.dtype)
        # write_out_trace(trace_buffer, str(opts.trace_file))
        write_out_trace(trace_buffer, "trace.txt")

    if np.allclose(
        ofm_mem_fmt_out,
        golden_output,
        rtol=0,
        atol=1,
    ):
        print("\nPASS!\n")
        exit(0)
    else:
        print("\nFailed.\n")
        exit(-1)


if __name__ == "__main__":
    p = test_utils.create_default_argparser()
    opts = p.parse_args(sys.argv[1:])
    main(opts)
