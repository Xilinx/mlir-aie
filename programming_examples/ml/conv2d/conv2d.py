# conv2d/conv2d.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024-2026 Advanced Micro Devices, Inc. or its affiliates
"""1x1 int8 Conv2D — Iron API design with ``@iron.jit`` compilation.

The per-tile kernel comes from ``aie.iron.kernels.conv2dk1_i8`` (auto-built
by ``compile_mlir_module(device=...)``). The runtime "scale" parameter
is lifted to a ``Compile[int]`` so it recompiles on change — that lets
the design skip the RTP buffer + barrier dance.

Compile-only entrypoint:
  ``python3 conv2d.py -d npu --xclbin-path build/final.xclbin
                              --insts-path build/insts.bin``
End-to-end verification lives in ``test.py``.
"""

import argparse

import numpy as np

import aie.iron as iron
from aie.iron import Compile, In, Out, ObjectFifo, Program, Runtime, Worker, kernels
from aie.iron.controlflow import range_
from aie.iron.device import device_from_args
from aie.utils.hostruntime.argparse import add_compile_args
from aie.utils.hostruntime.cli import run_design_cli


@iron.jit
def conv2d(
    a_in: In,
    w_in: In,
    b_out: Out,
    *,
    width: Compile[int] = 32,
    height: Compile[int] = 32,
    in_channels: Compile[int] = 64,
    out_channels: Compile[int] = 64,
    scale: Compile[int] = 10,
):
    if width % 8 != 0 or width < 8:
        raise ValueError("width must be a multiple of 8 and >= 8")
    if height < 2:
        raise ValueError("height must be > 1")
    if in_channels % 8 != 0 or in_channels < 8:
        raise ValueError("in_channels must be a multiple of 8 and >= 8")
    if out_channels % 8 != 0 or out_channels < 8:
        raise ValueError("out_channels must be a multiple of 8 and >= 8")

    device = iron.get_current_device()

    act_in = width * in_channels
    buf_in = act_in * 2
    weights = in_channels * out_channels
    act_out = width * out_channels
    buf_out = act_out * 2
    tensor_in_size = width * height * in_channels
    tensor_out_size = width * height * out_channels

    act_in_ty = np.ndarray[(act_in,), np.dtype[np.int8]]
    buf_in_ty = np.ndarray[(buf_in,), np.dtype[np.int8]]
    weights_ty = np.ndarray[(weights,), np.dtype[np.int8]]
    out_ty = np.ndarray[(act_out,), np.dtype[np.int8]]
    buf_out_ty = np.ndarray[(buf_out,), np.dtype[np.int8]]
    tensor_in_ty = np.ndarray[(tensor_in_size,), np.dtype[np.int8]]
    tensor_out_ty = np.ndarray[(tensor_out_size,), np.dtype[np.int8]]

    conv_fn = kernels.conv2dk1_i8(
        input_width=width,
        input_channels=in_channels,
        output_channels=out_channels,
    )

    of_act_l3l2 = ObjectFifo(buf_in_ty, name="inOF_act_L3L2")
    of_act_l2 = of_act_l3l2.cons().forward(obj_type=act_in_ty, name="act_L2_02")
    of_wts_l3l2 = ObjectFifo(weights_ty, depth=1, name="inOF_wts_0_L3L2")
    of_out_l2 = ObjectFifo(out_ty, name="out_02_L2")
    of_out_l3 = of_out_l2.cons().forward(obj_type=buf_out_ty, name="outOFL2L3")

    def core_fn(of_wts, of_act, of_out, kernel):
        elem_wts = of_wts.acquire(1)
        for _ in range_(height):
            elem_in = of_act.acquire(1)
            elem_out = of_out.acquire(1)
            kernel(elem_in, elem_wts, elem_out, width, in_channels, out_channels, scale)
            of_act.release(1)
            of_out.release(1)
        of_wts.release(1)

    worker = Worker(
        core_fn,
        [of_wts_l3l2.cons(), of_act_l2.cons(), of_out_l2.prod(), conv_fn],
        stack_size=0x600,
    )

    rt = Runtime()
    with rt.sequence(tensor_in_ty, weights_ty, tensor_out_ty) as (I, W, O):
        rt.start(worker)
        rt.fill(of_act_l3l2.prod(), I)
        rt.fill(of_wts_l3l2.prod(), W)
        rt.drain(of_out_l3.cons(), O, wait=True)

    return Program(device, rt).resolve_program()


def _make_argparser():
    p = argparse.ArgumentParser(prog="AIE Conv2D 1x1 int8")
    add_compile_args(p)
    p.add_argument("-wd", "--width", type=int, default=32)
    p.add_argument("-ht", "--height", type=int, default=32)
    p.add_argument("-ic", "--in_channels", type=int, default=64)
    p.add_argument("-oc", "--out_channels", type=int, default=64)
    p.add_argument("--scale", type=int, default=10)
    return p


def _compile_kwargs(opts):
    return dict(
        width=opts.width,
        height=opts.height,
        in_channels=opts.in_channels,
        out_channels=opts.out_channels,
        scale=opts.scale,
    )


def main():
    opts = _make_argparser().parse_args()
    run_design_cli(
        conv2d,
        opts,
        compile_kwargs=_compile_kwargs,
        device=lambda o: device_from_args(o, n_cols=1),
    )


if __name__ == "__main__":
    main()
