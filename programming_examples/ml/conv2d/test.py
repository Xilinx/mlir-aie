import torchvision
import torch
import torch.nn as nn
import sys

sys.path.append("../../utils")
from mlutils import DataShaper
import os
import math
import numpy as np
from brevitas.nn import QuantConv2d, QuantIdentity, QuantReLU
from brevitas.quant.fixed_point import (
    Int8ActPerTensorFixedPoint,
    Int8WeightPerTensorFixedPoint,
    Uint8ActPerTensorFixedPoint,
)

torch.manual_seed(0)

design = "conv1x1_cifar_scalar"
# design="conv1x1_cifar_vector"
# aie_teardown()

import xrtutils

xclbin_path = os.path.abspath("/build/final.xclbin")
insts_path = os.path.abspath("/build/insts.txt")

log_folder = "log/log_" + design

enable_aie = True
aie_is_setup = False
enable_trace = True
trace_file = "traces/" + design + ".txt"

app = None
in_buf = None
arg1_buf = None
out_buf = None

dtype_in = np.dtype("int8")
dtype_wts = np.dtype("int8")
dtype_out = np.dtype("uint8")

# shape_in_act = (256, 32,32)
# shape_in_wts1 = (256,64,1,1) #output, input, kx,ky
# shape_in_wts2 = (64,64,3,3) #output, input, kx,ky
# shape_in_wts3 = (64,256,1,1) #output, input, kx,ky
# shape_total_wts=(69632,1)
# shape_out  = (256, 32,32)

shape_in_act = (32, 8, 32, 8)  #'YCXC8' , 'CYX'
shape_in_wts1 = (8, 8, 1, 1, 8, 8)  # out,in,ky,kx,in8,out8

shape_total_wts = (4096, 1)
shape_out = (32, 8, 32, 8)

trace_size = 16384


def setup_aie(
    xclbin_path,
    insts_path,
    in_0_shape,
    in_0_dtype,
    in_1_shape,
    in_1_dtype,
    out_buf_shape,
    out_buf_dtype,
    enable_trace=False,
    kernel_name="MLIR_AIE",
):
    app = xrtutils.AIE_Application(xclbin_path, insts_path, kernel_name)
    app.register_buffer(2, shape=in_0_shape, dtype=in_0_dtype)
    app.register_buffer(3, shape=in_1_shape, dtype=in_1_dtype)
    if enable_trace:
        out_buf_len_bytes = np.prod(out_buf_shape) * np.dtype(out_buf_dtype).itemsize
        out_buf_shape = (out_buf_len_bytes + trace_size,)
        out_buf_dtype = np.uint8
    app.register_buffer(4, shape=out_buf_shape, dtype=out_buf_dtype)
    return app


def extract_trace(out_buf, out_buf_shape, out_buf_dtype):
    trace_size_words = trace_size // 4
    out_buf_flat = out_buf.reshape((-1,)).view(np.uint32)
    output_prefix = (
        out_buf_flat[:-trace_size_words].view(out_buf_dtype).reshape(out_buf_shape)
    )
    trace_suffix = out_buf_flat[-trace_size_words:]
    return output_prefix, trace_suffix


def write_out_trace(trace, file_name):
    out_str = "\n".join(f"{i:0{8}x}" for i in trace if i != 0)
    with open(file_name, "w") as f:
        f.write(out_str)


app = setup_aie(
    xclbin_path,
    insts_path,
    shape_in_act,
    dtype_in,
    shape_total_wts,
    dtype_wts,
    shape_out,
    dtype_out,
    enable_trace,
)


import numpy as np
from brevitas.nn import QuantConv2d, QuantIdentity, QuantReLU
from brevitas.quant.fixed_point import (
    Int8ActPerTensorFixedPoint,
    Int8WeightPerTensorFixedPoint,
    Uint8ActPerTensorFixedPoint,
)

torch.manual_seed(0)
torch.use_deterministic_algorithms(True)

if not os.path.exists(log_folder):
    os.makedirs(log_folder)


# input=torch.ones(64,32,32)
input = torch.randn(1, 64, 32, 32)
# image_name = f'./cifar_images/image_0.png'
# img = Image.open(image_name)
# input_tensor = cifar_test_transform(img)
# input = input_tensor.unsqueeze(0)
# Use a separate QuantIdentity so that the quantized output can be fed to both layers

num_classes = 10

ds = DataShaper()


def init_pad_input(x, input_channels, desired_channels=4):
    padding = torch.zeros(1, input_channels * (desired_channels - 1), 32, 32)
    return torch.cat((x, padding), 1)


# try:
for i in range(0, 1):

    class QuantBottleneck_projected(nn.Module):
        expansion = 4

        def __init__(self, in_planes=64, planes=64):
            super(QuantBottleneck_projected, self).__init__()
            self.quant_id_1 = QuantIdentity(
                act_quant=Int8ActPerTensorFixedPoint,
                bit_width=8,
                return_quant_tensor=True,
            )
            self.quant_conv1 = QuantConv2d(
                in_planes,
                planes,
                kernel_size=1,
                bit_width=8,
                weight_bit_width=8,
                bias=False,
                weight_quant=Int8WeightPerTensorFixedPoint,
                return_quant_tensor=True,
            )
            self.quant_relu1 = QuantReLU(
                act_quant=Uint8ActPerTensorFixedPoint,
                bit_width=8,
                return_quant_tensor=True,
            )

        def forward(self, x):
            out_q = self.quant_id_1(x)
            out = self.quant_conv1(out_q)
            out = self.quant_relu1(out)
            return out

    quant_bottleneck_model = QuantBottleneck_projected()

    quant_id_1 = QuantIdentity(
        act_quant=Int8ActPerTensorFixedPoint, bit_width=8, return_quant_tensor=True
    )
    quant_bottleneck_model.eval()
    quant_id_1.eval()

    init_scale = quant_bottleneck_model.quant_id_1.quant_act_scale()
    block_0_relu_1 = quant_bottleneck_model.quant_relu1.quant_act_scale()

    block_0_weight_scale1 = quant_bottleneck_model.quant_conv1.quant_weight_scale()

    block_0_combined_scale1 = -torch.log2(
        init_scale * block_0_weight_scale1 / block_0_relu_1
    )

    print("combined_scale after first conv1x1:", block_0_combined_scale1.item())

    block_0_int_weight_1 = quant_bottleneck_model.quant_conv1.quant_weight().int(
        float_datatype=True
    )

    q_bottleneck_out = quant_bottleneck_model(input)
    gold_out = q_bottleneck_out.int(float_datatype=True).data.numpy().astype(dtype_out)
    print("Golden::Brevitas::", gold_out)
    gold_out.tofile(log_folder + "/gold_out.txt", sep=",", format="%d")

    from brevitas.export import export_onnx_qcdq

    # ref_input = torch.ones(1, 3, 32, 32, device="cpu", dtype=dtype)
    export_onnx_qcdq(quant_bottleneck_model, input, log_folder + "/" + design + ".onnx")
    # # Brevitas convolution
    q_inp = quant_id_1(input)
    int_inp = q_inp.int(float_datatype=True)

    before_input = int_inp.squeeze().data.numpy().astype(dtype_in)
    # print(before_input)
    before_input.tofile(
        log_folder + "/before_ifm_mem_fmt_1x1.txt", sep=",", format="%d"
    )
    # ifm_mem_fmt = ds.reorder_mat(int_inp.squeeze().data.numpy().astype(dtype_in),'YCX' , 'CYX' )
    # ifm_mem_fmt = ds.reorder_mat(before_input,'YCX' , 'CYX' )
    ifm_mem_fmt = ds.reorder_mat(before_input, "YCXC8", "CYX")
    # print("Input after:::",ifm_mem_fmt.reshape((32,64, 32)))
    ifm_mem_fmt.tofile(log_folder + "/after_ifm_mem_fmt_1x1.txt", sep=",", format="%d")
    # print("ifm_mem_fmt shape",ifm_mem_fmt.shape)
    # wts1 = ds.reorder_mat(int_weight1.data.numpy().astype(dtype_in),'OIYX' , 'OIYX' )
    # wts2 = ds.reorder_mat(int_weight2.data.numpy().astype(dtype_in),'OIYX' , 'OIYX' )
    # wts3 = ds.reorder_mat(int_weight3.data.numpy().astype(dtype_in),'OIYX' , 'OIYX' )
    wts1 = ds.reorder_mat(
        block_0_int_weight_1.data.numpy().astype(dtype_wts), "OIYXI8O8", "OIYX"
    )

    total_wts = np.concatenate((wts1), axis=None)
    total_wts.tofile(log_folder + "/weights_mem_fmt_final.txt", sep=",", format="%d")
    print("total_wts", total_wts.shape)
    for i in range(0, 2):
        app.buffers[2].write(ifm_mem_fmt)  # input's standard format CYX | scalar YCX
        app.buffers[3].write(total_wts)  # wts's standard format OIYX | scalar OIYX
        # app.buffers[3].write(int_weight2.data.numpy().astype(dtype_in),offset=2048) # wts's standard format OIYX | scalar OIYX
        app.run()
        output3 = app.buffers[4].read()
        if enable_trace:
            output3, trace = extract_trace(output3, shape_out, dtype_out)
            write_out_trace(trace, trace_file)
    # temp_out=output3.reshape(32,256, 32)
    # ofm_mem_fmt = temp_out.swapaxes(0,1)
    temp_out = output3.reshape(32, 8, 32, 8)
    # print("AIE temp_out:::",temp_out)

    temp2_out = ds.reorder_mat(temp_out, "CDYX", "YCXD")
    # print("AIE reorder temp_out:::",temp_out)
    ofm_mem_fmt = temp2_out.reshape(64, 32, 32)
    ofm_mem_fmt.tofile(
        log_folder + "/after_ofm_mem_fmt_final.txt", sep=",", format="%d"
    )

    ofm_mem_fmt = torch.from_numpy(ofm_mem_fmt).unsqueeze(0)
    print("AIE output:::", ofm_mem_fmt)
    print(type(ofm_mem_fmt))
    print(type(q_bottleneck_out))
    print(
        "difference::",
        torch.max(torch.abs(ofm_mem_fmt * block_0_relu_1 - q_bottleneck_out)),
    )
    assert np.allclose(ofm_mem_fmt, gold_out, rtol=0, atol=2.0)