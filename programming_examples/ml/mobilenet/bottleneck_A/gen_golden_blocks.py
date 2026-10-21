#
# Copyright (C) 2024-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Brevitas-reference generator for bottleneck_A's per-block fixtures.

Blocks bn1, bn2, bn3, bn6, bn7 and bn8 share one 3-conv 1×1+3×3+1×1
quantized bottleneck recipe, differing only in:

* the per-block shape constants (``tensor{In,Out}{W,H,C}``, depthwise
  stride / channels);
* whether a residual skip-add is added after conv3;
* whether the model is calibrated on the calibration images (bn1, bn2)
  or on the seeded random input alone (bn3, bn6, bn7, bn8).

``--block`` picks a ``SPECS`` entry; ``generate_bottleneck_fixtures`` writes
the block's fixtures and its ``scale_factors_per_bn.json`` entry into
``data/``. Run from programming_examples/ml as
``python3 -m mobilenet.bottleneck_A.gen_golden_blocks --block bn1``.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch  # pyright: ignore[reportMissingImports]
import torch.nn as nn  # pyright: ignore[reportMissingImports]
from aie.utils.ml import DataShaper
from brevitas.nn import (  # pyright: ignore[reportMissingImports]
    QuantConv2d,
    QuantIdentity,
    QuantReLU,
)
from brevitas.quant.fixed_point import (  # pyright: ignore[reportMissingImports]
    Int8ActPerTensorFixedPoint,
    Int8WeightPerTensorFixedPoint,
    Uint8ActPerTensorFixedPoint,
)
from brevitas_examples.imagenet_classification.ptq.ptq_common import (  # pyright: ignore[reportMissingImports]
    calibrate,
)
from torchvision import transforms  # pyright: ignore[reportMissingImports]

from .. import mb_utils
from ..calibration import calibration_loader

torch.use_deterministic_algorithms(True)
torch.manual_seed(0)

VECTOR_SIZE = 8
DATA_DIR = Path(__file__).parent / "data"
PER_BN_SCALES = DATA_DIR / "scale_factors_per_bn.json"


@dataclass
class BottleneckSpec:
    """Shape + structural parameters for one bottleneck_A per-block fixture."""

    name: str  # "bn1", "bn2", ...
    tensor_in_w: int
    tensor_in_h: int
    tensor_in_c: int
    tensor_out_c: int
    depthwise_stride: int
    depthwise_channels: int
    has_skip: bool
    calibrate_images: bool


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
            # Only the no-skip forward reads bn_quant_id_2: a skip variant
            # requantizes conv3 at the input's scale (quant_id_1) for bn_add.
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


def _calibrate(model, expand_to_channels: int) -> None:
    """Calibrate ``model`` on the calibration images (bn1, bn2)."""
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            mb_utils.ExpandChannels(target_channels=expand_to_channels),
        ]
    )

    calibrate(calibration_loader(transform), model)


def shift_of(combined: torch.Tensor, what: str) -> int:
    """The integer shift a combined scale, ``-log2`` of a scale ratio, stands for.

    Args:
        combined: The ``-log2`` of a ratio of power-of-two scales.
        what: The layer it shifts, for the error.

    Returns:
        ``combined`` rounded to the nearest integer.

    Raises:
        ValueError: If ``combined`` is not an integer to within float error,
            so no shift implements the ratio.
    """
    value = combined.item()
    shift = round(value)
    if abs(value - shift) > 1e-6:
        raise ValueError(f"{what}: combined scale {value} is not an integer shift")
    return shift


def generate_bottleneck_fixtures(spec: BottleneckSpec) -> None:
    """Run the brevitas model and write ``spec``'s fixtures into ``data/``.

    Writes, for ``spec.name`` ``bnN``:

    * ``golden_output_bnN_single.txt``: quantized brevitas output (CHW int8).
    * ``input_bnN_single.txt``: the quantized input (int8).
    * ``bnN_single.txt``: conv1 / conv2 (depthwise) / conv3 weights,
      reordered and concatenated.
    * the ``BNN`` entry of ``scale_factors_per_bn.json``.
    """
    print(
        f"Running torch reference model for bottleneck_A_subblocks for {spec.name} ..."
    )

    dtype_in = np.dtype("int8")
    dtype_wts = np.dtype("int8")
    dtype_out = np.dtype("int8")

    in_c1_vec = math.floor(spec.tensor_in_c / VECTOR_SIZE)

    # Seeded random input (matches torch.manual_seed(0) at module load).
    input = torch.randn(1, in_c1_vec * VECTOR_SIZE, spec.tensor_in_h, spec.tensor_in_w)

    model = _build_model(spec)
    if spec.calibrate_images:
        _calibrate(model, expand_to_channels=spec.tensor_in_c)
    model.eval()

    q_bottleneck_out = model(input)
    golden_output = (
        q_bottleneck_out.int(float_datatype=True).data.numpy().astype(dtype_out)
    )
    print("Golden::Brevitas::", golden_output)
    q_inp = model.quant_id_1(input)
    int_inp = q_inp.int(float_datatype=True)

    init_scale = model.quant_id_1.act_quant.scale()
    relu_1 = model.bn_quant_relu1.act_quant.scale()
    relu_2 = model.bn_quant_relu2.act_quant.scale()
    post_conv3_scale = (
        init_scale if spec.has_skip else model.bn_quant_id_2.act_quant.scale()
    )
    w1 = model.bn_quant_conv1.weight_quant.scale()
    w2 = model.bn_quant_conv2.weight_quant.scale()
    w3 = model.bn_quant_conv3.weight_quant.scale()
    shifts = {
        "conv1x1_1": shift_of(-torch.log2(init_scale * w1 / relu_1), "conv1x1_1"),
        "conv3x3": shift_of(-torch.log2(relu_1 * w2 / relu_2), "conv3x3"),
        "conv1x1_2": shift_of(-torch.log2(relu_2 * w3 / post_conv3_scale), "conv1x1_2"),
        "skip_add": 0,
    }
    if spec.has_skip:
        skip_scale = model.bn_add.act_quant.scale()
        shifts["skip_add"] = shift_of(-torch.log2(init_scale / skip_scale), "skip_add")
    print(f"{spec.name} shifts: {shifts}")

    scales = (
        mb_utils.read_scale_factors(PER_BN_SCALES) if PER_BN_SCALES.exists() else {}
    )
    scales[spec.name.upper()] = shifts
    mb_utils.write_scale_factors(PER_BN_SCALES, dict(sorted(scales.items())))

    int_w1 = model.bn_quant_conv1.quant_weight().int(float_datatype=True)
    int_w2 = model.bn_quant_conv2.quant_weight().int(float_datatype=True)
    int_w3 = model.bn_quant_conv3.quant_weight().int(float_datatype=True)

    print("Writing golden output txt file.")
    golden_output.tofile(
        DATA_DIR / f"golden_output_{spec.name}_single.txt", sep=",", format="%d"
    )

    ds = DataShaper()
    before_input = int_inp.squeeze().data.numpy().astype(dtype_in)

    print("Writing input txt file.")
    before_input.tofile(
        DATA_DIR / f"input_{spec.name}_single.txt", sep=",", format="%d"
    )

    wts1 = ds.reorder_mat(int_w1.data.numpy().astype(dtype_wts), "OIYXI8O8", "OIYX")
    wts2 = ds.reorder_mat(int_w2.data.numpy().astype(dtype_wts), "OIYXI1O8", "OIYX")
    wts3 = ds.reorder_mat(int_w3.data.numpy().astype(dtype_wts), "OIYXI8O8", "OIYX")
    total_wts = np.concatenate((wts1, wts2, wts3), axis=None)

    print("Writing weights txt files.")
    total_wts.tofile(DATA_DIR / f"{spec.name}_single.txt", sep=",", format="%d")

    print("{}+{}+{}".format(wts1.shape, wts2.shape, wts3.shape))
    print(total_wts.shape)

    print("Done.")


SPECS = {
    "bn1": BottleneckSpec(
        "bn1",
        tensor_in_w=112,
        tensor_in_h=112,
        tensor_in_c=16,
        tensor_out_c=24,
        depthwise_stride=2,
        depthwise_channels=64,
        has_skip=False,
        calibrate_images=True,
    ),
    "bn2": BottleneckSpec(
        "bn2",
        tensor_in_w=56,
        tensor_in_h=56,
        tensor_in_c=24,
        tensor_out_c=24,
        depthwise_stride=1,
        depthwise_channels=72,
        has_skip=True,
        calibrate_images=True,
    ),
    "bn3": BottleneckSpec(
        "bn3",
        tensor_in_w=56,
        tensor_in_h=56,
        tensor_in_c=24,
        tensor_out_c=40,
        depthwise_stride=2,
        depthwise_channels=72,
        has_skip=False,
        calibrate_images=False,
    ),
    "bn6": BottleneckSpec(
        "bn6",
        tensor_in_w=28,
        tensor_in_h=28,
        tensor_in_c=40,
        tensor_out_c=80,
        depthwise_stride=2,
        depthwise_channels=240,
        has_skip=False,
        calibrate_images=False,
    ),
    "bn7": BottleneckSpec(
        "bn7",
        tensor_in_w=14,
        tensor_in_h=14,
        tensor_in_c=80,
        tensor_out_c=80,
        depthwise_stride=1,
        depthwise_channels=200,
        has_skip=True,
        calibrate_images=False,
    ),
    "bn8": BottleneckSpec(
        "bn8",
        tensor_in_w=14,
        tensor_in_h=14,
        tensor_in_c=80,
        tensor_out_c=80,
        depthwise_stride=1,
        depthwise_channels=184,
        has_skip=True,
        calibrate_images=False,
    ),
}


def main():
    parser = argparse.ArgumentParser(description="bottleneck_A per-block fixtures")
    parser.add_argument("--block", required=True, choices=sorted(SPECS))
    generate_bottleneck_fixtures(SPECS[parser.parse_args().block])


if __name__ == "__main__":
    main()
