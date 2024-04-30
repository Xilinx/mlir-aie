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
import model as res

from aie.utils.xrt import setup_aie, extract_trace, write_out_trace, execute
import aie.utils.test as test_utils

torch.use_deterministic_algorithms(True)
torch.manual_seed(0)
from utils import  unpickle,load_class_label
import torchvision
from torchvision import transforms
from PIL import Image
from brevitas.nn import QuantConv2d, QuantIdentity, QuantReLU
from brevitas.quant.fixed_point import (
    Int8ActPerTensorFixedPoint,
    Int8WeightPerTensorFixedPoint,
    Uint8ActPerTensorFixedPoint,
)
from brevitas.graph.target.flexml import preprocess_for_flexml_quantize
from brevitas_examples.imagenet_classification.ptq.ptq_common import quantize_model
import torch.utils.data as data_utils
from brevitas_examples.imagenet_classification.ptq.ptq_common import calibrate
from brevitas_examples.imagenet_classification.ptq.ptq_common import calibrate_bn
from brevitas_examples.imagenet_classification.utils import generate_dataloader
from brevitas_examples.imagenet_classification.utils import SEED
from brevitas_examples.imagenet_classification.utils import validate

def main(opts):
    design = "resnet_conv2_x_int8"
    xclbin_path = opts.xclbin
    insts_path = opts.instr

    log_folder = "log/"
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)

    num_iter = 1
    npu_time_total = 0
    npu_time_min = 9999999
    npu_time_max = 0
    trace_size = 16384
    enable_trace = False
    trace_file = "log/trace_" + design + ".txt"
    # ------------------------------------------------------
    # Configure this to match your design's buffer size
    # ------------------------------------------------------
    dtype_in = np.dtype("int8")
    dtype_wts = np.dtype("int8")
    dtype_out = np.dtype("uint8")

    shape_in_act = (32, 8, 32, 8)
    shape_total_wts = (212992, 1)
    shape_out = (32, 32, 32, 8)

    # ------------------------------------------------------
    # Post training quantization to get int8 weights and activation for AIE
    # ------------------------------------------------------
    num_classes = 10
    model = res.Resnet50_conv2x_offload(num_classes)
    weights = "trained_resnet50/weight.tar" #trained FP model
    saved_model_dict = torch.load(weights, map_location=torch.device("cpu"))
    model.load_state_dict(saved_model_dict)

    data_dir = "data"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose(
        [
            transforms.Pad(4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32),
            transforms.ToTensor(),
        ]
    )
    transform_train = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    # CIFAR-10 dataset
    train_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, transform=transform_train, download=True
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, transform=transform_test, download=True
    )

    # Data loader
    indices = torch.arange(256)
    tr_sub = data_utils.Subset(train_dataset, indices)
    val_sub = data_utils.Subset(test_dataset, indices)
    calib_loader = torch.utils.data.DataLoader(dataset=tr_sub, batch_size=64, shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset=val_sub, batch_size=64, shuffle=False)
    img_shape = 32
    model_aie = preprocess_for_flexml_quantize(
        model.aie,
        torch.ones(1, 64, img_shape, img_shape),
        equalize_iters=1000,
        equalize_merge_bias=True,
        merge_bn=True,
    )

    quant_model = quantize_model(
        model_aie,
        backend="flexml",
        scale_factor_type="po2_scale",
        bias_bit_width=32,
        weight_bit_width=8,
        weight_narrow_range=False,
        weight_param_method="stats",
        weight_quant_granularity="per_tensor",
        weight_quant_type="sym",
        layerwise_first_last_bit_width=8,
        act_bit_width=8,
        act_param_method="stats",
        act_quant_percentile=99.999,
        act_quant_type="sym",
        quant_format="int",
        layerwise_first_last_mantissa_bit_width=4,
        layerwise_first_last_exponent_bit_width=3,
        weight_mantissa_bit_width=4,
        weight_exponent_bit_width=3,
        act_mantissa_bit_width=4,
        act_exponent_bit_width=3,
    )

    model.aie = quant_model
    model.eval()
    print("Starting post training quantization:")
    calibrate(calib_loader, model)
    model.eval()
    device, dtype = (
    next(model.parameters()).device,
    next(model.parameters()).dtype,
    )
    # -----------------------


    from numpy import load

    params = {}
    weights = {}
    for name, module in model.named_modules():
        if isinstance(module, QuantConv2d):
            # print(name)
            # print(module.quant_weight().scale)
            weights[name + ".int_weight"] = module.quant_weight().int(float_datatype=False)
            params[name + "_scale"] = module.quant_weight().scale.detach().numpy()
        if isinstance(module, QuantIdentity):
            # print(name)
            # print(module.quant_act_scale())
            params[name + "_scale"] = module.quant_act_scale()
        if isinstance(module, QuantReLU):
            # print(name)
            # print(module.quant_act_scale())
            params[name + "_scale"] = module.quant_act_scale()
    np.savez(os.path.join(os.getcwd(), "int_weights.npz"), **weights)
    np.savez(os.path.join(os.getcwd(), "int_conv_scale.npz"), **params)
    int_wts_data = load("int_weights.npz", allow_pickle=True)
    int_scale_data = load("int_conv_scale.npz", allow_pickle=True)

    int_wts_data_lst = int_wts_data.files    
    block_0_int_weight_1 = torch.from_numpy(int_wts_data["aie.layer1.conv1.int_weight"])
    block_0_int_weight_2 = torch.from_numpy(int_wts_data["aie.layer1.conv2.int_weight"])
    block_0_int_weight_3 = torch.from_numpy(int_wts_data["aie.layer1.conv3.int_weight"])
    block_0_int_weight_skip = torch.from_numpy(int_wts_data["aie.layer1.shortcut.0.int_weight"])

    block_1_int_weight_1 = torch.from_numpy(int_wts_data["aie.layer2.conv1.int_weight"])
    block_1_int_weight_2 = torch.from_numpy(int_wts_data["aie.layer2.conv2.int_weight"])
    block_1_int_weight_3 = torch.from_numpy(int_wts_data["aie.layer2.conv3.int_weight"])

    block_2_int_weight_1 = torch.from_numpy(int_wts_data["aie.layer3.conv1.int_weight"])
    block_2_int_weight_2 = torch.from_numpy(int_wts_data["aie.layer3.conv2.int_weight"])
    block_2_int_weight_3 = torch.from_numpy(int_wts_data["aie.layer3.conv3.int_weight"])

    int_scale_data_lst = int_scale_data.files

    init_scale = int_scale_data["aie.x_quant_scale"]
    block_0_relu_1 = int_scale_data["aie.layer1.relu1_scale"]
    block_0_relu_2 = int_scale_data["aie.layer1.relu2_scale"]
    block_0_relu_3 = int_scale_data["aie.layer1.relu3_scale"]
    block_0_add_scale = int_scale_data["aie.add_quant_scale"]

    block_0_weight_scale_1 = int_scale_data["aie.layer1.conv1_scale"]
    block_0_weight_scale_2 = int_scale_data["aie.layer1.conv2_scale"]
    block_0_weight_scale_3 = int_scale_data["aie.layer1.conv3_scale"]
    block_0_weight_scale_skip = int_scale_data["aie.layer1.shortcut.0_scale"]

    block_1_relu_1 = int_scale_data["aie.layer2.relu1_scale"]
    block_1_relu_2 = int_scale_data["aie.layer2.relu2_scale"]
    block_1_relu_3 = int_scale_data["aie.layer2.relu3_scale"]
    block_1_add_scale = int_scale_data["aie.add_1_quant_scale"]

    block_1_weight_scale_1 = int_scale_data["aie.layer2.conv1_scale"]
    block_1_weight_scale_2 = int_scale_data["aie.layer2.conv2_scale"]
    block_1_weight_scale_3 = int_scale_data["aie.layer2.conv3_scale"]

    block_2_relu_1 = int_scale_data["aie.layer3.relu1_scale"]
    block_2_relu_2 = int_scale_data["aie.layer3.relu2_scale"]
    block_2_relu_3 = int_scale_data["aie.layer3.relu3_scale"]
    block_2_add_scale = int_scale_data["aie.add_2_quant_scale"]

    block_2_weight_scale_1 = int_scale_data["aie.layer3.conv1_scale"]
    block_2_weight_scale_2 = int_scale_data["aie.layer3.conv2_scale"]
    block_2_weight_scale_3 = int_scale_data["aie.layer3.conv3_scale"]

    for name, param in model.named_parameters():
        if name.endswith(".bias"):
            param.data.fill_(0)

    block_0_combined_scale1 = -math.log(
    init_scale * block_0_weight_scale_1 / block_0_relu_1, 2
    )  # after conv1x1
    block_0_combined_scale2 = -math.log(
    block_0_relu_1 * block_0_weight_scale_2 / block_0_relu_2, 2
    )  # after conv3x3
    block_0_combined_scale3 = -math.log(
    block_0_relu_2 * block_0_weight_scale_3 / block_0_add_scale, 2
    )  # after conv1x1
    block_0_combined_scale4 = -math.log(
    block_0_add_scale / block_0_relu_3, 2
    )  # after skip addition using init scale
    # combined_scale4=-math.log(inp_scale1/inp_scale4)
    block_0_combined_scale_skip = -math.log(
    init_scale * block_0_weight_scale_skip / block_0_add_scale, 2
    )  # after LHS conv1x1

    block_1_combined_scale1 = -math.log(
        block_0_relu_3 * block_1_weight_scale_1 / block_1_relu_1, 2
    )  # after conv1x1
    block_1_combined_scale2 = -math.log(
        block_1_relu_1 * block_1_weight_scale_2 / block_1_relu_2, 2
    )  # after conv3x3
    block_1_combined_scale3 = -math.log(
        block_1_relu_2 * block_1_weight_scale_3 / block_1_add_scale, 2
    )  # after conv1x1
    block_1_combined_scale4 = -math.log(
        block_1_add_scale / block_1_relu_3, 2
    )  # after skip addition using init scale

    block_2_combined_scale1 = -math.log(
        block_1_relu_3 * block_2_weight_scale_1 / block_2_relu_1, 2
    )  # RHS after first conv1x1 | clip 0-->255
    block_2_combined_scale2 = -math.log(
        block_2_relu_1 * block_2_weight_scale_2 / block_2_relu_2, 2
    )  # RHS after second conv3x3 | clip 0-->255
    block_2_combined_scale3 = -math.log(
        block_2_relu_2 * block_2_weight_scale_3 / block_2_add_scale, 2
    )  # RHS after third conv1x1 | clip -128-->+127
    block_2_combined_scale4 = -math.log(
        block_2_add_scale / block_2_relu_3, 2
    )  # After addition | clip 0-->255

    print("--------------------------------------------------------------")
    print("Block0 combined_scale after first conv1x1:", block_0_combined_scale1)
    print("Block0 combined_scale after second conv3x3:", block_0_combined_scale2)
    print("Block0 combined_scale after third conv1x1:", block_0_combined_scale3)
    print("Block0 combined_scale after adding skip connection:", (block_0_combined_scale4))
    print("Block0 combined_scale after skip conv1x1:", block_0_combined_scale_skip)

    print("--------------------------------------------------------------")
    print("Block1 combined_scale after first conv1x1:", block_1_combined_scale1)
    print("Block1 combined_scale after second conv3x3:", block_1_combined_scale2)
    print("Block1 combined_scale after third conv1x1:", block_1_combined_scale3)
    print("Block1 combined_scale after adding skip connection:", (block_1_combined_scale4))
    print("--------------------------------------------------------------")
    print("Block2 combined_scale block2 after first conv1x1:", block_2_combined_scale1)
    print("Block2 combined_scale block2 after second conv3x3:", block_2_combined_scale2)
    print("Block2 combined_scale block2 after third conv1x1:", block_2_combined_scale3)
    print(
        "Block2 combined_scale block2 after adding skip connection:",
        (block_2_combined_scale4),
    )
    print("------------------------------------------------------------------")
        # ------------------------------------------------------
    # Get device, load the xclbin & kernel and register them
    # ------------------------------------------------------
    app = setup_aie(
        xclbin_path,
        insts_path,
        shape_in_act,
        dtype_in,
        shape_total_wts,
        dtype_wts,
        shape_out,
        dtype_out,
        enable_trace=enable_trace,
        trace_size=trace_size,
    )
    # ------------------------------------------------------
    # Reorder input data-layout
    # ------------------------------------------------------
    ds = DataShaper()

    block0_wts1 = ds.reorder_mat(
        block_0_int_weight_1.data.numpy().astype(dtype_wts), "OIYXI8O8", "OIYX"
    )
    block0_wts2 = ds.reorder_mat(
        block_0_int_weight_2.data.numpy().astype(dtype_wts), "OIYXI8O8", "OIYX"
    )
    block0_wts3 = ds.reorder_mat(
        block_0_int_weight_3.data.numpy().astype(dtype_wts), "OIYXI8O8", "OIYX"
    )
    block0_wts_skip = ds.reorder_mat(
        block_0_int_weight_skip.data.numpy().astype(dtype_wts), "OIYXI8O8", "OIYX"
    )

    total_wts = np.concatenate(
        (block0_wts1, block0_wts2, block0_wts3, block0_wts_skip), axis=None
    )

    block1_wts1 = ds.reorder_mat(
        block_1_int_weight_1.data.numpy().astype(dtype_wts), "OIYXI8O8", "OIYX"
    )
    block1_wts2 = ds.reorder_mat(
        block_1_int_weight_2.data.numpy().astype(dtype_wts), "OIYXI8O8", "OIYX"
    )
    block1_wts3 = ds.reorder_mat(
        block_1_int_weight_3.data.numpy().astype(dtype_wts), "OIYXI8O8", "OIYX"
    )

    total_wts2 = np.concatenate(
        (total_wts, block1_wts1, block1_wts2, block1_wts3), axis=None
    )

    block2_wts1 = ds.reorder_mat(
        block_2_int_weight_1.data.numpy().astype(dtype_wts), "OIYXI8O8", "OIYX"
    )
    block2_wts2 = ds.reorder_mat(
        block_2_int_weight_2.data.numpy().astype(dtype_wts), "OIYXI8O8", "OIYX"
    )
    block2_wts3 = ds.reorder_mat(
        block_2_int_weight_3.data.numpy().astype(dtype_wts), "OIYXI8O8", "OIYX"
    )

    total_wts3 = np.concatenate(
        (total_wts2, block2_wts1, block2_wts2, block2_wts3), axis=None
    )

    total_wts3.tofile(log_folder + "/weights_mem_fmt_final.txt", sep=",", format="%d")

    import time
    import cv2

    predicted_label = [None] * 64
    cpu_predicted_label = [None] * 64
    aie_time = [None] * 64
    metafile = r"./data/cifar-10-batches-py/batches.meta"
    datafile = r"./data/cifar-10-batches-py/test_batch"
    data_batch_1 = unpickle(datafile)
    metadata = unpickle(metafile)
    images = data_batch_1["data"]
    labels = data_batch_1["labels"]
    images = np.reshape(images, (10000, 3, 32, 32))
    dirname = "cifar_images"
    if not os.path.exists(dirname):
        os.mkdir(dirname)

    # Extract and dump first 10 images
    for i in range(0, 100):
        im = images[i]
        im = im.transpose(1, 2, 0)
        im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
        im_name = f"./cifar_images/image_{i}.png"
        cv2.imwrite(im_name, im)


    label_path = "data/cifar10_label_map.txt"
    model_num_classes = 10
    class_label_map = load_class_label(label_path, model_num_classes)
    quant_id_1 = QuantIdentity(
        act_quant=Uint8ActPerTensorFixedPoint, bit_width=8, return_quant_tensor=True
    )
    quant_id_1.eval()


    # ------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------
   
    for i in range(0, 64):
        print("____________________________________IMAGE {}____________________________________________".format(i))
        image_name = f"./cifar_images/image_{i}.png"
        img = Image.open(image_name)
        input_tensor = transform_test(img)
        input_batch = input_tensor.unsqueeze(0)
        with torch.no_grad():
            # print(input_batch.shape
            start = time.time() * 1000
            output1 = model.first(input_batch)

            # AIE OFFLOAD
            qnt_inp = model.aie.x_quant(output1)
            int_inp = model.aie.x_quant(output1).int(float_datatype=True)
            before_input = int_inp.squeeze().data.numpy().astype(dtype_in)
            ifm_mem_fmt = ds.reorder_mat(before_input, "YCXC8", "CYX")
            start = time.time_ns()
            aie_output = execute(app, ifm_mem_fmt, total_wts3) * block_2_relu_3
            stop = time.time_ns()
            temp_out = aie_output.reshape(32, 32, 32, 8)
            temp2_out = ds.reorder_mat(temp_out, "CDYX", "YCXD")
            ofm_mem_fmt = temp2_out.reshape(256, 32, 32)
            ofm_mem_fmt = torch.from_numpy(ofm_mem_fmt).unsqueeze(0)
            final_output_aie = model.post(ofm_mem_fmt)

            # ------------------------------------------------------------------------------
            # Baseline output for functional correctness
            output_golden = model.aie(output1)
            max_error = torch.max(torch.abs(ofm_mem_fmt - output_golden))
            # print(max_error)
            final_output_base = model.post(output_golden)
            predicted_class = np.argmax(final_output_aie)
            predicted_label[i] = metadata["label_names"][predicted_class]
            cpu_predicted_class = np.argmax(final_output_base)
            cpu_predicted_label[i] = metadata["label_names"][cpu_predicted_class]
            label = metadata["label_names"][labels[i]]
            print(
                f" Predicted AIE: {predicted_label[i]}, Predicted CPU: {predicted_label[i]}"
            )

            # Calculate the five categories with the highest classification probability
            prediction_class_index = (
                torch.topk(final_output_aie, k=5, sorted=True).indices.squeeze(0).tolist()
            )
            golden_prediction_class_index = (
                torch.topk(final_output_base, k=5, sorted=True).indices.squeeze(0).tolist()
            )
            npu_time = stop - start
    npu_time_total = npu_time_total + npu_time

    # ------------------------------------------------------
    # Compare the AIE output and the golden reference
    # ------------------------------------------------------
    print("\nAvg NPU time: {}us.".format(int((npu_time_total / 64) / 1000)))
    for x, y in zip(predicted_label, predicted_label):
       if x != y:
           print("\nFailed.\n")
           exit(-1)
    print("\nPASS!\n")
    exit(0)

if __name__ == "__main__":
    p = test_utils.create_default_argparser()
    opts = p.parse_args(sys.argv[1:])
    main(opts)
