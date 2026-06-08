#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024-2026 Advanced Micro Devices, Inc. or its affiliates
"""Shared brevitas-reference generator for bottleneck_A per-block fixtures.

Five sibling ``gen_golden_bn{1,2,3,6,7}.py`` scripts shared an identical
3-conv-1×1+3×3+1×1 quantized bottleneck recipe, differing only in:

* the per-block shape constants (``tensor{In,Out}{W,H,C}``, depthwise
  stride / channels);
* whether a residual skip-add is added after conv3;
* whether the model is calibrated against an ImageNet subset (bn1, bn2)
  or with the seeded random input alone (bn3, bn6, bn7).

This module collapses that recipe into ``generate_bottleneck_fixtures``.
Each per-block wrapper supplies the constants + the two flags and
delegates here.  ``bn0`` and ``bn8_9`` have meaningfully different
forward graphs and stay separate.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from aie.utils.ml import DataShaper

from brevitas.nn import QuantConv2d, QuantIdentity, QuantReLU
from brevitas.quant.fixed_point import (
    Int8ActPerTensorFixedPoint,
    Int8WeightPerTensorFixedPoint,
    Uint8ActPerTensorFixedPoint,
)

torch.use_deterministic_algorithms(True)
torch.manual_seed(0)

VECTOR_SIZE = 8


@dataclass
class BottleneckSpec:
    """Shape + structural knobs for one bottleneck_A per-block fixture."""

    name: str  # "bn1", "bn2", ...
    tensor_in_w: int
    tensor_in_h: int
    tensor_in_c: int
    tensor_out_c: int
    depthwise_stride: int
    depthwise_channels: int
    has_skip: bool
    calibrate_imagenet: bool


def _build_model(spec: BottleneckSpec):
    """Build the quantized 1×1 + 3×3-depthwise + 1×1 bottleneck module.

    Module attribute names use ``bn_quant_conv{1,2,3}`` / ``bn_quant_relu{1,2}``
    / ``bn_quant_id_2`` regardless of the per-block name.  These names are
    runtime-introspection only and do not appear in any of the emitted
    ``.txt`` fixtures.
    """

    in_planes = spec.tensor_in_c
    expand = spec.depthwise_channels
    project = spec.tensor_out_c
    depthwise_stride = spec.depthwise_stride
    has_skip = spec.has_skip

    class QuantBottleneckA(nn.Module):
        def __init__(self):
            super().__init__()
            self.quant_id_1 = QuantIdentity(
                act_quant=Int8ActPerTensorFixedPoint,
                bit_width=8,
                return_quant_tensor=True,
            )

            self.bn_quant_conv1 = QuantConv2d(
                in_planes,
                expand,
                kernel_size=1,
                bit_width=8,
                weight_bit_width=8,
                bias=False,
                weight_quant=Int8WeightPerTensorFixedPoint,
                return_quant_tensor=True,
            )
            self.bn_quant_conv2 = QuantConv2d(
                expand,
                expand,
                kernel_size=3,
                stride=depthwise_stride,
                padding=1,
                padding_mode="zeros",
                bit_width=8,
                groups=expand,
                weight_bit_width=8,
                bias=False,
                weight_quant=Int8WeightPerTensorFixedPoint,
                return_quant_tensor=True,
            )
            self.bn_quant_conv3 = QuantConv2d(
                expand,
                project,
                kernel_size=1,
                bit_width=8,
                weight_bit_width=8,
                bias=False,
                weight_quant=Int8WeightPerTensorFixedPoint,
                return_quant_tensor=True,
            )
            self.bn_quant_relu1 = QuantReLU(
                act_quant=Uint8ActPerTensorFixedPoint,
                bit_width=8,
                return_quant_tensor=True,
            )
            self.bn_quant_relu2 = QuantReLU(
                act_quant=Uint8ActPerTensorFixedPoint,
                bit_width=8,
                return_quant_tensor=True,
            )
            # bn_quant_id_2 is constructed in every variant for state-parity
            # with the original per-bn scripts, but is only *used* in the
            # forward graph for the no-skip variants.  Skip variants route
            # the conv3 output through ``quant_id_1`` (re-applying the input
            # quant) and into ``bn_add`` instead.
            self.bn_quant_id_2 = QuantIdentity(
                act_quant=Int8ActPerTensorFixedPoint,
                bit_width=8,
                return_quant_tensor=True,
            )
            if has_skip:
                self.bn_add = QuantIdentity(
                    act_quant=Int8ActPerTensorFixedPoint,
                    bit_width=8,
                    return_quant_tensor=True,
                )

        def forward(self, x):
            out_q = self.quant_id_1(x)
            out = self.bn_quant_conv1(out_q)
            out = self.bn_quant_relu1(out)
            out = self.bn_quant_conv2(out)
            out = self.bn_quant_relu2(out)
            out = self.bn_quant_conv3(out)
            if has_skip:
                out = self.quant_id_1(out)
                out = out + out_q
                out = self.bn_add(out)
            else:
                out = self.bn_quant_id_2(out)
            return out

    return QuantBottleneckA()


def _calibrate_with_imagenet(model, expand_to_channels: int) -> None:
    """Calibrate ``model`` over 32 ImageNet samples (bn1, bn2 path).

    Matches the original scripts bit-for-bit: same transforms, same
    ImageFolder path, same DataLoader settings.
    """
    from mb_utils import ExpandChannels
    from brevitas_examples.imagenet_classification.ptq.ptq_common import calibrate
    import torchvision
    import torch.utils.data as data_utils
    from torchvision import transforms

    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ExpandChannels(target_channels=expand_to_channels),
        ]
    )

    src_data = "/group/xrlabs2/imagenet/calibration"
    dataset = torchvision.datasets.ImageFolder(src_data, transform)
    indices = torch.arange(4)
    val_sub = data_utils.Subset(dataset, indices)
    calib_loader = torch.utils.data.DataLoader(
        dataset=val_sub, batch_size=32, shuffle=False
    )
    calibrate(calib_loader, model)


def generate_bottleneck_fixtures(spec: BottleneckSpec, *, log_dir: str | Path) -> None:
    """Run the brevitas model and emit ``log/`` fixtures for ``spec``.

    Emits the canonical filenames each downstream test expects:

    * ``golden_output.txt`` — quantized brevitas output (CHW int8).
    * ``before_ifm_mem_fmt_1x1.txt`` — pre-reorder input (int8).
    * ``after_ifm_mem_fmt.txt`` — input reordered ``YCXC8 → CYX``.
    * ``after_weights_mem_fmt_final.txt`` — concatenated reordered
      weights for conv1 / conv2 (depthwise) / conv3.

    No return value; everything lands in ``log_dir``.
    """
    print(
        f"Running torch reference model for bottleneck_A_subblocks for {spec.name} ..."
    )
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    dtype_in = np.dtype("int8")
    dtype_wts = np.dtype("int8")
    dtype_out = np.dtype("int8")

    in_c1_vec = math.floor(spec.tensor_in_c / VECTOR_SIZE)
    out_c3_vec = math.floor(spec.tensor_out_c / VECTOR_SIZE)
    in_w3 = spec.tensor_in_w // spec.depthwise_stride
    in_h3 = spec.tensor_in_h // spec.depthwise_stride

    # Shape banners that the original scripts printed — preserved so any
    # downstream check that reads the log still sees the same shape lines.
    shape_total_wts = (
        spec.tensor_in_c * spec.depthwise_channels
        + 3 * 3 * spec.depthwise_channels
        + spec.depthwise_channels * spec.tensor_out_c,
        1,
    )
    shape_in_act = (
        spec.tensor_in_h,
        in_c1_vec,
        spec.tensor_in_w,
        VECTOR_SIZE,
    )  # YCXC8 → CYX
    shape_out = (in_h3, out_c3_vec, in_w3, VECTOR_SIZE)  # HCWC8
    shape_out_final = (out_c3_vec * VECTOR_SIZE, in_h3, in_w3)  # CHW
    del shape_in_act, shape_out, shape_out_final  # computed for parity with originals

    # Seeded random input (matches torch.manual_seed(0) at module load).
    input = torch.randn(1, in_c1_vec * VECTOR_SIZE, spec.tensor_in_h, spec.tensor_in_w)

    model = _build_model(spec)
    if spec.calibrate_imagenet:
        _calibrate_with_imagenet(model, expand_to_channels=spec.tensor_in_c)
    model.eval()

    q_bottleneck_out = model(input)
    golden_output = (
        q_bottleneck_out.int(float_datatype=True).data.numpy().astype(dtype_out)
    )
    print("Golden::Brevitas::", golden_output)
    q_inp = model.quant_id_1(input)
    int_inp = q_inp.int(float_datatype=True)

    # Combined-scale prints — informational; the originals always emitted these.
    # No-skip variants divide cs3 by bn_quant_id_2's scale; skip variants
    # divide by the input quant_id_1's scale (because the skip path routes
    # conv3 → quant_id_1 → + out_q → bn_add).
    init_scale = model.quant_id_1.act_quant.scale()
    relu_1 = model.bn_quant_relu1.act_quant.scale()
    relu_2 = model.bn_quant_relu2.act_quant.scale()
    post_conv3_scale = (
        init_scale if spec.has_skip else model.bn_quant_id_2.act_quant.scale()
    )
    w1 = model.bn_quant_conv1.weight_quant.scale()
    w2 = model.bn_quant_conv2.weight_quant.scale()
    w3 = model.bn_quant_conv3.weight_quant.scale()
    cs1 = -torch.log2(init_scale * w1 / relu_1)
    cs2 = -torch.log2(relu_1 * w2 / relu_2)
    cs3 = -torch.log2(relu_2 * w3 / post_conv3_scale)

    banner = "*" * 20 + spec.name.upper() + "*" * 23
    print(banner)
    print("combined_scale after conv1x1:", cs1.item())
    print("combined_scale after conv3x3:", cs2.item())
    print("combined_scale after conv1x1:", cs3.item())
    if spec.has_skip:
        skip_scale = model.bn_add.act_quant.scale()
        skip_cs = -torch.log2(init_scale / skip_scale)
        print("combined_scale after skip add:", skip_cs.item())
    print(banner)

    int_w1 = model.bn_quant_conv1.quant_weight().int(float_datatype=True)
    int_w2 = model.bn_quant_conv2.quant_weight().int(float_datatype=True)
    int_w3 = model.bn_quant_conv3.quant_weight().int(float_datatype=True)

    # Emit the .txt fixtures (canonical filenames — overwritten per-run).
    print("Writing golden output txt file.")
    golden_output.tofile(str(log_dir / "golden_output.txt"), sep=",", format="%d")

    ds = DataShaper()
    before_input = int_inp.squeeze().data.numpy().astype(dtype_in)

    print("Writing input txt file.")
    before_input.tofile(
        str(log_dir / "before_ifm_mem_fmt_1x1.txt"), sep=",", format="%d"
    )
    ifm_mem_fmt = ds.reorder_mat(before_input, "YCXC8", "CYX")
    ifm_mem_fmt.tofile(str(log_dir / "after_ifm_mem_fmt.txt"), sep=",", format="%d")

    wts1 = ds.reorder_mat(int_w1.data.numpy().astype(dtype_wts), "OIYXI8O8", "OIYX")
    wts2 = ds.reorder_mat(int_w2.data.numpy().astype(dtype_wts), "OIYXI1O8", "OIYX")
    wts3 = ds.reorder_mat(int_w3.data.numpy().astype(dtype_wts), "OIYXI8O8", "OIYX")
    total_wts = np.concatenate((wts1, wts2, wts3), axis=None)

    print("Writing weights txt files.")
    total_wts.tofile(
        str(log_dir / "after_weights_mem_fmt_final.txt"), sep=",", format="%d"
    )

    print("{}+{}+{}".format(wts1.shape, wts2.shape, wts3.shape))
    print(shape_total_wts)
    print(total_wts.shape)

    print("Done.")
