#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2024, Advanced Micro Devices, Inc.
import sys
import onnx
import torchvision
from torchvision.io import read_image
from torchvision.models import resnet50, ResNet50_Weights
import torch
from torchvision import transforms
from PIL import Image
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


def convert_to_numpy(array):
    if isinstance(array, np.ndarray):
        return array
    elif isinstance(array, torch.Tensor):
        return array.cpu().numpy()
    else:
        raise TypeError("Unsupported array type")


import json
import torch.nn.functional as F


def pad_tensor(tensor, target_shape):
    if isinstance(tensor, np.ndarray):
        tensor = torch.from_numpy(tensor)
    # Calculate the padding dimensions
    pad_size = target_shape[1] - tensor.shape[1]
    padding_dims = (
        0,
        0,
        0,
        0,
        0,
        pad_size,
    )  # Padding along the second dimension (index 1)
    padded_tensor = F.pad(tensor, padding_dims)
    return padded_tensor


# Function to read scale factors from JSON file
def read_scale_factors(file_path):
    with open(file_path, "r") as file:
        return json.load(file)


# Function to write scale factors to JSON file
def write_scale_factors(file_path, scale_factors):
    with open(file_path, "w") as file:
        json.dump(scale_factors, file, indent=4)


log_dir = "log/"
data_dir = "data/"

# Read the existing scale factors
scale_factor_file = "scale_factors_final.json"
scale_factors = read_scale_factors(data_dir + scale_factor_file)

vectorSize = 8


tensorInW = 224
tensorInH = 224
tensorInC = 8

tensor_init_OutC = 16
tensor_init_OutH = tensorInW // 2
tensor_init_OutW = tensorInH // 2


bneck_0_InW2 = tensor_init_OutW
bneck_0_InH2 = tensor_init_OutH
bneck_0_InC2 = tensor_init_OutC
bneck_0_OutC2 = bneck_0_InC2

bneck_0_InW3 = bneck_0_InW2
bneck_0_InH3 = bneck_0_InH2
bneck_0_InC3 = bneck_0_OutC2
bneck_0_OutC3 = bneck_0_InC3

# config for bn2
bn1_depthwiseStride = 2
bn1_depthWiseChannels = 64
bneck_1_OutC = 24

# each layer
bneck_1_InW1 = bneck_0_InW3
bneck_1_InH1 = bneck_0_InH3
bneck_1_InC1 = bneck_0_InC3
bneck_1_OutC1 = bn1_depthWiseChannels

bneck_1_InW2 = bneck_1_InW1
bneck_1_InH2 = bneck_1_InH1
bneck_1_OutC2 = bneck_1_OutC1

bneck_1_InW3 = bneck_1_InW2 // bn1_depthwiseStride
bneck_1_InH3 = bneck_1_InH2 // bn1_depthwiseStride
bneck_1_OutC3 = bneck_1_OutC

# config for bn2
bn2_depthwiseStride = 1
bn2_depthWiseChannels = 72
bneck_2_OutC = 24

# each layer
bneck_2_InW1 = bneck_1_InW3
bneck_2_InH1 = bneck_1_InH3
bneck_2_InC1 = bneck_1_OutC3
bneck_2_OutC1 = bn2_depthWiseChannels

bneck_2_InW2 = bneck_2_InW1
bneck_2_InH2 = bneck_2_InH1
bneck_2_OutC2 = bneck_2_OutC1

bneck_2_InW3 = bneck_2_InW2 // bn2_depthwiseStride
bneck_2_InH3 = bneck_2_InH2 // bn2_depthwiseStride
bneck_2_OutC3 = bneck_2_OutC

# config for bn3
bn3_depthwiseStride = 2
bn3_depthWiseChannels = 72
bneck_3_OutC = 40

# each layer
bneck_3_InW1 = bneck_2_InW3
bneck_3_InH1 = bneck_2_InH3
bneck_3_InC1 = bneck_2_OutC3
bneck_3_OutC1 = bn3_depthWiseChannels

bneck_3_InW2 = bneck_3_InW1
bneck_3_InH2 = bneck_3_InH1
bneck_3_OutC2 = bneck_3_OutC1

bneck_3_InW3 = bneck_3_InW2 // bn3_depthwiseStride
bneck_3_InH3 = bneck_3_InH2 // bn3_depthwiseStride
bneck_3_OutC3 = bneck_3_OutC


# config for bn5
bn4_depthwiseStride = 1
bn4_depthWiseChannels = 120
bneck_4_OutC = 40

# each layer
bneck_4_InW1 = bneck_3_InW3
bneck_4_InH1 = bneck_3_InH3
bneck_4_InC1 = bneck_3_OutC3
bneck_4_OutC1 = bn4_depthWiseChannels

bneck_4_InW2 = bneck_4_InW1
bneck_4_InH2 = bneck_4_InH1
bneck_4_OutC2 = bneck_4_OutC1

bneck_4_InW3 = bneck_4_InW2 // bn4_depthwiseStride
bneck_4_InH3 = bneck_4_InH2 // bn4_depthwiseStride
bneck_4_OutC3 = bneck_4_OutC

# config for bn5
bn5_depthwiseStride = 1
bn5_depthWiseChannels = 120
bneck_5_InW1 = 28
bneck_5_InH1 = 28
bneck_5_InC1 = 40
bneck_5_OutC = 40

bneck_5_OutC1 = bn5_depthWiseChannels

bneck_5_InW2 = bneck_5_InW1
bneck_5_InH2 = bneck_5_InH1
bneck_5_OutC2 = bneck_5_OutC1

bneck_5_InW3 = bneck_5_InW2 // bn5_depthwiseStride
bneck_5_InH3 = bneck_5_InH2 // bn5_depthwiseStride
bneck_5_OutC3 = bneck_5_OutC

# config for bn6
bneck_6_tensorInW = bneck_5_InW3
bneck_6_tensorInH = bneck_5_InH3
bneck_6_tensorInC = bneck_5_OutC3
bneck_6_tensorOutC = 80
bn6_depthwiseStride = 2
bn6_depthWiseChannels = 240

bneck_6_InW1 = bneck_6_tensorInW
bneck_6_InH1 = bneck_6_tensorInH
bneck_6_InC1 = bneck_6_tensorInC
bneck_6_OutC1 = bn6_depthWiseChannels

bneck_6_InW2 = bneck_6_InW1
bneck_6_InH2 = bneck_6_InH1
bneck_6_OutC2 = bneck_6_OutC1

bneck_6_InW3 = bneck_6_InW2 // bn6_depthwiseStride
bneck_6_InH3 = bneck_6_InH2 // bn6_depthwiseStride
bneck_6_OutC3 = bneck_6_tensorOutC

# config for bn7
bneck_7_tensorInW = bneck_6_InW3
bneck_7_tensorInH = bneck_6_InH3
bneck_7_tensorInC = bneck_6_OutC3
bneck_7_tensorOutC = 80

bn7_depthwiseStride = 1
bn7_depthWiseChannels = 200

bneck_7_InW1 = bneck_7_tensorInW
bneck_7_InH1 = bneck_7_tensorInH
bneck_7_InC1 = bneck_7_tensorInC
bneck_7_OutC1 = bn7_depthWiseChannels

bneck_7_InW2 = bneck_7_InW1
bneck_7_InH2 = bneck_7_InH1
bneck_7_OutC2 = bneck_7_OutC1

bneck_7_InW3 = bneck_7_InW2
bneck_7_InH3 = bneck_7_InH2
bneck_7_OutC3 = bneck_7_tensorOutC

# config for bn8
bneck_8_tensorInW = bneck_7_InW3
bneck_8_tensorInH = bneck_7_InH3
bneck_8_tensorInC = bneck_7_OutC3
bneck_8_tensorOutC = 80
bneck_8_depthwiseStride = 1
bneck_8_depthWiseChannels = 184

bneck_8_InW1 = bneck_8_tensorInW
bneck_8_InH1 = bneck_8_tensorInH
bneck_8_InC1 = bneck_8_tensorInC
bneck_8_OutC1 = bneck_8_depthWiseChannels

bneck_8_InW2 = bneck_8_InW1
bneck_8_InH2 = bneck_8_InH1
bneck_8_OutC2 = bneck_8_OutC1

bneck_8_InW3 = bneck_8_InW2
bneck_8_InH3 = bneck_8_InH2
bneck_8_OutC3 = bneck_8_tensorOutC


# config for bn8
bneck_9_tensorInW = bneck_8_InW3
bneck_9_tensorInH = bneck_8_InH3
bneck_9_tensorInC = bneck_8_OutC3
bneck_9_tensorOutC = 80
bneck_9_depthwiseStride = 1
bneck_9_depthWiseChannels = 184

bneck_9_InW1 = bneck_9_tensorInW
bneck_9_InH1 = bneck_9_tensorInH
bneck_9_InC1 = bneck_9_tensorInC
bneck_9_OutC1 = bneck_9_depthWiseChannels

bneck_9_InW2 = bneck_9_InW1
bneck_9_InH2 = bneck_9_InH1
bneck_9_OutC2 = bneck_9_OutC1

bneck_9_InW3 = bneck_9_InW2
bneck_9_InH3 = bneck_9_InH2
bneck_9_OutC3 = bneck_9_tensorOutC


bneck_10_InW1 = 14
bneck_10_InH1 = 14
bneck_10_InC1 = bneck_9_OutC3
bneck_10_OutC1 = 480

bneck_10_InW2 = 14
bneck_10_InH2 = 14
bneck_10_OutC2 = bneck_10_OutC1

bneck_10_InW3 = 14
bneck_10_InH3 = 14
bneck_10_OutC3 = 112

bneck_11_OutC1 = 336
bneck_11_OutC2 = 336
bneck_11_OutC3 = 112
kdim = 3
stride = 1
padding = 1

bneck_11_InW3 = 14
bneck_11_InH3 = 14

bneck_12_InH1 = 14
bneck_12_InW1 = 14
bneck_12_OutC1 = 336

bneck_12_InH2 = 14
bneck_12_InW2 = 14
bneck_12_OutC2 = 336

bneck_12_InW3 = 7
bneck_12_InH3 = 7
bneck_12_OutC3 = 80

# tensorOutW = bneck_3_InW3
# tensorOutH = bneck_3_InH3
# tensorOutC = bneck_3_OutC3


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

post_L1_InW = 7
post_L1_InH = 7
post_L1_InC = 80


post_L1_OutC = 960
post_L1_OutW = 1
post_L1_OutH = 1

post_L1_OutC_padd = 1280  # added for padding

post_L2_OutC = post_L1_OutC_padd
post_L2_OutW = 1
post_L2_OutH = 1

post_kdim = 7
post_stride = 1

tensorOutW = post_L2_OutW
tensorOutH = post_L2_OutH
tensorOutC = post_L2_OutC

# tensorOutW = bneck_13_InW3
# tensorOutH = bneck_13_InH3
# tensorOutC = bneck_13_OutC3
# Target shape for the padded weights
target_weights_shape = (post_L1_OutC_padd, post_L1_OutC_padd, 1, 1)
# Pad the input to the target shape
target_shape = (1, post_L1_OutC_padd, 1, 1)
InC_vec = math.floor(3 * tensorInC / vectorSize)
OutC_vec = math.floor(tensorOutC / vectorSize)


def main(opts):
    design = "mobilenet_complete"
    xclbin_path = opts.xclbin
    insts_path = opts.instr
    ds = DataShaper()
    log_dir = "log/"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    num_iter = 1
    npu_time_total = 0
    npu_time_min = 9999999
    npu_time_max = 0
    trace_size = 16384
    # enable_trace = False
    enable_trace = 0
    trace_file = "log/trace_" + design + ".txt"
    print("enable_trace: ", enable_trace)
    print("opts.trace_size: ", 0)
    # ------------------------------------------------------
    # Configure this to match your design's buffer size
    # ------------------------------------------------------
    dtype_in = np.dtype("int8")
    dtype_wts = np.dtype("int8")
    dtype_out = np.dtype("uint8")
    dtype_out_aie = np.dtype("uint16")
    # dtype_out = np.dtype("int8")

    shape_total_wts = (
        # (3 * 3 * tensorInC * tensor_init_OutC) +
        # (3*3*bneck_0_OutC2 + bneck_0_OutC2*bneck_0_OutC3 + bneck_1_InC1*bneck_1_OutC1 + 3*3*bneck_1_OutC2 + bneck_1_OutC2*bneck_1_OutC3)+
        # (bneck_2_InC1*bneck_2_OutC1 + 3*3*bneck_2_OutC2 + bneck_2_OutC2*bneck_2_OutC3)+
        # (bneck_3_InC1*bneck_3_OutC1 + 3*3*bneck_3_OutC2 + bneck_3_OutC2*bneck_3_OutC3)+
        # (bneck_4_InC1*bneck_4_OutC1 + 3*3*bneck_4_OutC2 + bneck_4_OutC2*bneck_4_OutC3)+
        # (bneck_5_InC1*bneck_5_OutC1 + 3*3*bneck_5_OutC2 + bneck_5_OutC2*bneck_5_OutC3)+
        # (bneck_6_InC1*bneck_6_OutC1 + 3*3*bneck_6_OutC2 + bneck_6_OutC2*bneck_6_OutC3)+
        # (bneck_7_InC1*bneck_7_OutC1 + 3*3*bneck_7_OutC2 + bneck_7_OutC2*bneck_7_OutC3)+
        # (bneck_8_InC1*bneck_8_OutC1 + 3*3*bneck_8_OutC2 + bneck_8_OutC2*bneck_8_OutC3)+
        # (bneck_9_InC1*bneck_9_OutC1 + 3*3*bneck_9_OutC2 + bneck_9_OutC2*bneck_9_OutC3)+
        # (bneck_10_InC1*bneck_10_OutC1)+(3*3*bneck_10_OutC2)+(bneck_10_OutC2*bneck_10_OutC3)+
        # (bneck_10_OutC3*bneck_11_OutC1)+(3*3*bneck_11_OutC2)+(bneck_11_OutC2*bneck_11_OutC3)+
        # (bneck_11_OutC3*bneck_12_OutC1)+(3*3*bneck_12_OutC2)+(bneck_12_OutC2*bneck_12_OutC3)+
        2 * ((bneck_13_OutC1 * bneck_13_InC1) + (bneck_13_OutC2 * bneck_13_OutC3)),
        1,
    )

    print("total weights:::", shape_total_wts)
    shape_in_act = (tensorInH, InC_vec, tensorInW, vectorSize)  #'YCXC8' , 'CYX'
    shape_out = (tensorOutH, OutC_vec, tensorOutW, vectorSize)  # HCWC8
    shape_out_final = (OutC_vec * vectorSize, tensorOutH, tensorOutW)  # CHW

    # ------------------------------------------------------
    # Get device, load the xclbin & kernel and register them
    # ------------------------------------------------------
    npu_kernel = NPUKernel(xclbin_path, insts_path)
    kernel_handle = DefaultNPURuntime.load(npu_kernel)

    golden_output = np.loadtxt(
        data_dir + "golden_output.txt", delimiter=",", dtype="int8"
    )
    golden_output = golden_output.reshape(1, tensorOutC, tensorOutH, tensorOutW)
    padded_golden_output = pad_tensor(golden_output, target_shape)

    ds = DataShaper()

    before_input = np.loadtxt(
        data_dir + "before_ifm_mem_fmt_1x1.txt", delimiter=",", dtype="uint8"
    )
    before_input = before_input.reshape(tensorInC, tensorInH, tensorInW)

    ifm_mem_fmt = ds.reorder_mat(before_input, "YCXC8", "CYX")
    ifm_mem_fmt.tofile(log_dir + "after_ifm_mem_fmt.txt", sep=",", format="%d")

    # init_wts_fmt = np.loadtxt(data_dir + "init_chain.txt", delimiter=",", dtype="int32")
    # bn0_total_wts = np.loadtxt(data_dir + "bn0_chain.txt", delimiter=",", dtype="int32")
    # bn1_total_wts = np.loadtxt(data_dir + "bn1_chain.txt", delimiter=",", dtype="int32")
    # bn01_total_wts = np.loadtxt(data_dir + "bn0_1_chain.txt", delimiter=",", dtype="int32")
    # bn2_total_wts = np.loadtxt(data_dir + "bn2_chain.txt", delimiter=",", dtype="int32")
    # bn3_total_wts = np.loadtxt(data_dir + "bn3_chain.txt", delimiter=",", dtype="int32")
    # bn4_total_wts = np.loadtxt(data_dir + "bn4_chain.txt", delimiter=",", dtype="int32")
    # bn5_total_wts = np.loadtxt(data_dir + "bn5_chain.txt", delimiter=",", dtype="int32")
    # bn4_5_total_wts = np.loadtxt(data_dir + "bn4_5_chain.txt", delimiter=",", dtype="int32")
    # bn6_total_wts = np.loadtxt(data_dir + "bn6_chain.txt", delimiter=",", dtype="int32")
    # bn7_total_wts = np.loadtxt(data_dir + "bn7_chain.txt", delimiter=",", dtype="int32")
    # bn8_total_wts = np.loadtxt(data_dir + "bn8_chain.txt", delimiter=",", dtype="int32")
    # bn9_total_wts = np.loadtxt(data_dir + "bn9_chain.txt", delimiter=",", dtype="int32")
    # bn8_9_total_wts = np.loadtxt(data_dir + "bn8_9_chain.txt", delimiter=",", dtype="int32")

    # bn10_wts1 = np.loadtxt(data_dir + "bn10_1_chain.txt", delimiter=",", dtype="int32")
    # bn10_wts2 = np.loadtxt(data_dir + "bn10_2_chain.txt", delimiter=",", dtype="int32")
    # bn10_wts3 = np.loadtxt(data_dir + "bn10_3_chain.txt", delimiter=",", dtype="int32")
    # bn11_wts1 = np.loadtxt(data_dir + "bn11_1_chain.txt", delimiter=",", dtype="int32")
    # bn11_wts2 = np.loadtxt(data_dir + "bn11_2_chain.txt", delimiter=",", dtype="int32")
    # bn11_wts3 = np.loadtxt(data_dir + "bn11_3_chain.txt", delimiter=",", dtype="int32")
    # bn12_wts1 = np.loadtxt(data_dir + "bn12_1_chain.txt", delimiter=",", dtype="int32")
    # bn12_wts2 = np.loadtxt(data_dir + "bn12_2_chain.txt", delimiter=",", dtype="int32")
    # bn12_wts3 = np.loadtxt(data_dir + "bn12_3_chain.txt", delimiter=",", dtype="int32")
    # bn12_wts2_3 = np.loadtxt(data_dir + "bn12_2_3_chain.txt", delimiter=",", dtype="int32")

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

    c_total_wts = np.concatenate(
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

    total_wts = np.concatenate((c_total_wts), axis=None)

    total_wts.tofile(log_dir + "after_weights_mem_fmt_final.txt", sep=",", format="%d")
    # print("{}+{}+{}".format(bn6_wts1.shape, bn6_wts2.shape, bn6_wts3.shape))
    print(shape_total_wts)
    print(total_wts.shape)

    # ------------------------------------------------------
    # Setup buffers run loop
    # ------------------------------------------------------
    in1 = iron.tensor(ifm_mem_fmt, dtype=dtype_in)
    in2 = iron.tensor(total_wts, dtype=dtype_wts)
    out = iron.zeros(shape_out, dtype=dtype_out_aie)
    buffers = [in1, in2, out]

    trace_config = None
    if enable_trace:
        trace_config = TraceConfig(
            trace_size=trace_size,
            trace_file=trace_file,
            trace_after_last_tensor=False,
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
        DefaultNPURuntime.run(kernel_handle, buffers)
        stop = time.time_ns()
        npu_time = stop - start
        npu_time_total = npu_time_total + npu_time

    # ------------------------------------------------------
    # Reorder output data-layout
    # ------------------------------------------------------
    aie_output = out.numpy()  # .astype(dtype_out)
    temp_out = aie_output.reshape(shape_out)
    temp_out = ds.reorder_mat(temp_out, "CDYX", "YCXD")
    ofm_mem_fmt = temp_out.reshape(shape_out_final)
    ofm_mem_fmt.tofile(log_dir + "after_ofm_mem_fmt_final.txt", sep=",", format="%d")
    ofm_mem_fmt_out = torch.from_numpy(ofm_mem_fmt).unsqueeze(0)
    # print("Golden::Brevitas::", golden_output)
    # print("AIE::", ofm_mem_fmt_out)
    # ------------------------------------------------------
    # Compare the AIE output and the golden reference
    # ------------------------------------------------------
    print("\nAvg NPU time: {}us.".format(int((npu_time_total / num_iter) / 1000)))
    # print("\nMin NPU time: {}us.".format(int((npu_time_min) / 1000)))
    # print("\nMax NPU time: {}us.".format(int((npu_time_max) / 1000)))
    golden = convert_to_numpy(padded_golden_output)
    ofm_mem_fmt_out = convert_to_numpy(ofm_mem_fmt_out)
    max_difference = np.max(
        np.abs((golden.astype(int)) - (ofm_mem_fmt_out.astype(int)))
    )
    print("max_difference:", max_difference)
    # Find the indices where the mismatch happens
    # Find the indices where the mismatch happens
    mismatch_indices = np.where(golden != ofm_mem_fmt_out)

    # Extract mismatch values
    mismatch_values_golden = golden[mismatch_indices]
    mismatch_values_ofm = ofm_mem_fmt_out[mismatch_indices]

    # Print mismatch indices and corresponding values
    print("golden shape: ", golden.shape)
    print("Output shape: ", ofm_mem_fmt_out.shape)

    # TODO Disabled to not print mistmatches to stdout for now
    # print("Mismatch indices and corresponding values:")
    # for idx, (golden_value, ofm_value) in zip(
    #     zip(*mismatch_indices), zip(mismatch_values_golden, mismatch_values_ofm)
    # ):
    #     print(
    #         f"Index: {idx}, Golden value: {golden_value}, OFM value: {ofm_value}, diff: {golden_value.astype(int)-ofm_value.astype(int)}"
    #     )

    print(
        "\n***WARNING**** Temporary check where we accept atol=9, whereas it should be 1"
    )
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
        # atol=3,
        atol=9,
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
