# conv2d_14x14/conv2dk14.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024-2026 Advanced Micro Devices, Inc. or its affiliates
"""14x14 Conv2D — Iron API design with ``@iron.jit`` compilation.

This design processes a 896x896 input with 4 channels through a
conv2dk14 kernel producing 1152 output channels. The kernel runs on
sub-tiles (16 tiles × 16 output channels per call) and is invoked
``out_channels/16 × height/14 × 4`` times to cover the full image.

The kernel comes from ``aie_kernels/aie2p/conv2dk14.cc`` (aie2p only).
The library's ``kernels.conv2dk14`` uses a different per-call output
layout, so this design wires an ``ExternalFunction`` directly with the
sub-tile sizing the original kernel writes.

Compile-only entrypoint:
  ``python3 conv2dk14.py -d npu2 --xclbin-path build/final.xclbin
                                 --insts-path build/insts.bin``
End-to-end verification lives in ``test.py``.
"""

import argparse
from pathlib import Path

import numpy as np

import aie.iron as iron
from aie.iron import Compile, In, Out, ObjectFifo, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.device import from_name
from aie.iron.kernel import ExternalFunction
from aie.utils import config
from aie.utils.hostruntime.argparse import add_compile_args
from aie.utils.hostruntime.cli import run_design_cli

# Sub-tile sizing baked into the conv2dk14 kernel.
_SUB_OUT_CHANNELS = 16
_SUB_TILES = 16
_KERNEL_SIZE = 14
_IN_CHANNELS = 4
_OUT_CHANNELS = 1152
_X_BLOCKS = 4
_KERNEL_SRC = Path(__file__).resolve().parents[3] / "aie_kernels/aie2p/conv2dk14.cc"


def _conv2dk14_extern(act_in_ty, weights_ty, out_ty):
    return ExternalFunction(
        "conv2dk14_i8",
        source_file=str(_KERNEL_SRC),
        arg_types=[
            act_in_ty, weights_ty, out_ty,
            np.int32, np.int32, np.int32, np.int32, np.int32,
        ],
        include_dirs=[config.cxx_header_path()],
        compile_flags=["-DUINT8_ACT"],
    )


@iron.jit
def conv2dk14(
    a_in: In,
    w_in: In,
    b_out: Out,
    *,
    width: Compile[int] = 896,
    height: Compile[int] = 896,
    scale: Compile[int] = 14,
):
    if width % 8 != 0 or width < 8:
        raise ValueError("width must be a multiple of 8 and >= 8")
    if height % 8 != 0 or height < 8:
        raise ValueError("height must be a multiple of 8 and >= 8")

    device = iron.get_current_device()

    act_in = _KERNEL_SIZE * _KERNEL_SIZE * _IN_CHANNELS * _SUB_TILES
    weights = _KERNEL_SIZE * _KERNEL_SIZE * _IN_CHANNELS * _SUB_OUT_CHANNELS
    act_out = _SUB_TILES * _SUB_OUT_CHANNELS

    out_channels_group = _OUT_CHANNELS // _SUB_OUT_CHANNELS
    width_out = width // _KERNEL_SIZE
    height_out = height // _KERNEL_SIZE

    tensor_in_size = width * height * _IN_CHANNELS * out_channels_group
    tensor_wts_size = weights * out_channels_group
    tensor_out_size = width_out * height_out * _SUB_OUT_CHANNELS * out_channels_group

    buf_in = _KERNEL_SIZE * width * _IN_CHANNELS
    buf_out = _SUB_OUT_CHANNELS * width_out * height_out

    act_in_ty = np.ndarray[(act_in,), np.dtype[np.uint8]]
    buf_in_ty = np.ndarray[(buf_in,), np.dtype[np.uint8]]
    weights_ty = np.ndarray[(weights,), np.dtype[np.int8]]
    out_ty = np.ndarray[(act_out,), np.dtype[np.int8]]
    buf_out_ty = np.ndarray[(buf_out,), np.dtype[np.int8]]
    tensor_in_ty = np.ndarray[(tensor_in_size,), np.dtype[np.uint8]]
    tensor_wts_ty = np.ndarray[(tensor_wts_size,), np.dtype[np.int8]]
    tensor_out_ty = np.ndarray[(tensor_out_size,), np.dtype[np.int8]]

    conv_fn = _conv2dk14_extern(act_in_ty, weights_ty, out_ty)

    of_act_l3l2 = ObjectFifo(
        buf_in_ty,
        name="inOF_act_L3L2",
        dims_from_stream_per_cons=[
            (_KERNEL_SIZE, _KERNEL_SIZE * _IN_CHANNELS),
            (64, _KERNEL_SIZE * _KERNEL_SIZE * _IN_CHANNELS),
            (_KERNEL_SIZE * _IN_CHANNELS, 1),
        ],
    )
    of_act_l2 = of_act_l3l2.cons().forward(
        obj_type=act_in_ty,
        name="act_L2_02",
        dims_to_stream=[
            (2, _KERNEL_SIZE * _KERNEL_SIZE * _IN_CHANNELS * 8),
            (_KERNEL_SIZE * _KERNEL_SIZE // 2, 2 * _IN_CHANNELS),
            (8, _KERNEL_SIZE * _KERNEL_SIZE * _IN_CHANNELS),
            (2 * _IN_CHANNELS, 1),
        ],
    )

    of_wts_l3l2 = ObjectFifo(weights_ty, depth=1, name="inOF_wts_0_L3L2")

    of_out_l2 = ObjectFifo(out_ty, name="out_02_L2")
    of_out_l3 = of_out_l2.cons().forward(
        obj_type=buf_out_ty,
        name="outOFL2L3",
        dims_to_stream=[(256, 256), (16, 8), (2, 128), (8, 1)],
    )

    def core_fn(of_wts, of_act, of_out, kernel):
        x_dim = width // _X_BLOCKS
        elem_wts = of_wts.acquire(1)
        for _ in range_(height // _KERNEL_SIZE):
            for _ in range_(_X_BLOCKS):
                elem_in = of_act.acquire(1)
                elem_out = of_out.acquire(1)
                kernel(elem_in, elem_wts, elem_out,
                       x_dim, _IN_CHANNELS, _SUB_OUT_CHANNELS,
                       _KERNEL_SIZE, scale)
                of_act.release(1)
                of_out.release(1)
        of_wts.release(1)

    worker = Worker(
        core_fn,
        [of_wts_l3l2.cons(), of_act_l2.cons(), of_out_l2.prod(), conv_fn],
        stack_size=0x600,
    )

    rt = Runtime()
    with rt.sequence(tensor_in_ty, tensor_wts_ty, tensor_out_ty) as (I, W, O):
        rt.start(worker)
        rt.fill(of_act_l3l2.prod(), I)
        rt.fill(of_wts_l3l2.prod(), W)
        rt.drain(of_out_l3.cons(), O, wait=True)

    return Program(device, rt).resolve_program()


def _make_argparser():
    p = argparse.ArgumentParser(prog="AIE Conv2D 14x14 (aie2p)")
    add_compile_args(p)
    p.add_argument("-wd", "--width", type=int, default=896)
    p.add_argument("-ht", "--height", type=int, default=896)
    p.add_argument("--scale", type=int, default=14)
    return p


def _compile_kwargs(opts):
    return dict(width=opts.width, height=opts.height, scale=opts.scale)


def main():
    opts = _make_argparser().parse_args()
    run_design_cli(
        conv2dk14,
        opts,
        compile_kwargs=_compile_kwargs,
        device=lambda o: from_name(o.dev, n_cols=1),
    )


if __name__ == "__main__":
    main()
