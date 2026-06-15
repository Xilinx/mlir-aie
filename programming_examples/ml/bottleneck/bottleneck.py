# bottleneck/bottleneck.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024-2026 Advanced Micro Devices, Inc. or its affiliates
"""ResNet-style bottleneck block (1x1 → 3x3 → 1x1+skip) — @iron.jit port.

Per-layer kernels come from the library:
  * 1x1 conv:           ``kernels.conv2dk1(act_dtype=int8)``  (uint8 out)
  * 3x3 conv:           ``kernels.conv2dk3(act_dtype=uint8, weight_output_channels=full)``
                        — the per-worker ``output_channels`` is half, with a
                        ``channel_offset`` runtime arg picking each worker's slice.
  * 1x1 + skip:         ``kernels.conv2dk1_skip(act_dtype=int8)``

The 3x3 stage is pinned to Tile(0,3) and Tile(0,5) to sidestep an
auto-placer limitation on bottleneck (see project memory
``project_iron_placer_constraint_bugs.md``). The three RTP scales are
lifted to ``CompileTime[int]`` so the design drops the RTP buffer +
WorkerRuntimeBarrier dance.
"""

import argparse

import numpy as np

import aie.iron as iron
from aie.iron import CompileTime, In, Out, ObjectFifo, Program, Runtime, Worker, kernels
from aie.iron.controlflow import range_
from aie.iron.device import AnyMemTile, Tile
from aie.utils.hostruntime.argparse import (
    device_from_args,
    add_compile_args,
)
from aie.utils.hostruntime.cli import run_design_cli


@iron.jit
def bottleneck(
    a_in: In,
    w_in: In,
    b_out: Out,
    *,
    tensor_w: CompileTime[int] = 32,
    tensor_h: CompileTime[int] = 32,
    tensor_in_c: CompileTime[int] = 256,
    scale_1x1: CompileTime[int] = 1,
    scale_3x3: CompileTime[int] = 11,
    scale_skip: CompileTime[int] = 1,
    skip_scale: CompileTime[int] = 0,
):
    device = iron.get_current_device()

    l1_in_c = tensor_in_c
    l1_out_c = l1_in_c // 4
    l2_in_c = l1_out_c
    l2_out_c = l2_in_c
    l3_in_c = l2_out_c
    l3_out_c = l3_in_c * 4

    activations_in = tensor_w * tensor_h * tensor_in_c
    wts1_sz = l1_in_c * l1_out_c
    wts2_sz = 3 * 3 * l2_in_c * l2_out_c
    wts3_sz = l3_in_c * l3_out_c
    total_wts = wts1_sz + wts2_sz + wts3_sz

    act_in_l3_ty = np.ndarray[(activations_in,), np.dtype[np.int8]]
    wts_in_l3_ty = np.ndarray[(total_wts,), np.dtype[np.int8]]

    l1_in_ty = np.ndarray[(tensor_w, 1, l1_in_c), np.dtype[np.int8]]
    wts1_ty = np.ndarray[(wts1_sz,), np.dtype[np.int8]]
    l1_out_ty = np.ndarray[(tensor_w, 1, l1_out_c), np.dtype[np.uint8]]

    l2_in_ty = np.ndarray[(tensor_w, 1, l2_in_c), np.dtype[np.uint8]]
    wts2_ty = np.ndarray[(wts2_sz,), np.dtype[np.int8]]
    l2_out_ty = np.ndarray[(tensor_w, 1, l2_out_c // 2), np.dtype[np.uint8]]

    l3_in_ty = np.ndarray[(tensor_w, 1, l3_in_c // 2), np.dtype[np.uint8]]
    wts3_ty = np.ndarray[(wts3_sz,), np.dtype[np.int8]]
    l3_out_ty = np.ndarray[(tensor_w, 1, l3_out_c), np.dtype[np.uint8]]

    conv1 = kernels.conv2dk1(
        input_width=tensor_w,
        input_channels=l1_in_c,
        output_channels=l1_out_c,
        act_dtype=np.int8,
    )
    conv3 = kernels.conv2dk3(
        input_width=tensor_w,
        input_channels=l2_in_c,
        output_channels=l2_out_c // 2,
        weight_output_channels=l2_out_c,
        act_dtype=np.uint8,
    )
    conv1_skip = kernels.conv2dk1_skip(
        input_width=tensor_w,
        input_channels=l3_in_c,
        output_channels=l3_out_c,
        act_dtype=np.int8,
    )

    of_act_l3l2 = ObjectFifo(l1_in_ty, name="inOF_act_L3L2")
    of_skip_buf = of_act_l3l2.cons(4).forward(depth=2, tile=AnyMemTile, name="skip_buf")

    of_wts_l3l2 = ObjectFifo(wts_in_l3_ty, depth=1, name="inOF_wts_0_L3L2")
    wts_offsets = [0, wts1_sz, wts1_sz + wts2_sz]
    wts_buf_00, wts_buf_01, wts_buf_02 = of_wts_l3l2.cons().split(
        wts_offsets,
        obj_types=[wts1_ty, wts2_ty, wts3_ty],
        names=[f"wts_buf_0{i}" for i in range(3)],
    )

    of_act_2_3_5 = ObjectFifo(l1_out_ty, name="act_2_3_5")
    of_act_3_4 = ObjectFifo(l2_out_ty, name="act_3_4")
    of_act_5_4 = ObjectFifo(l2_out_ty, name="act_5_4")
    of_out_l2l3 = ObjectFifo(l3_out_ty, name="outOFL2L3")

    workers = []

    def conv1x1_fn(of_wts, of_act_in, of_act_out, conv1x1):
        elem_wts = of_wts.acquire(1)
        for _ in range_(tensor_h):
            elem_in = of_act_in.acquire(1)
            elem_out = of_act_out.acquire(1)
            conv1x1(elem_in, elem_wts, elem_out, tensor_w, l1_in_c, l1_out_c, scale_1x1)
            of_act_in.release(1)
            of_act_out.release(1)
        of_wts.release(1)

    workers.append(
        Worker(
            conv1x1_fn,
            fn_args=[wts_buf_00.cons(), of_act_l3l2.cons(), of_act_2_3_5.prod(), conv1],
        )
    )

    def conv3x3_fn(of_wts, of_act_in, of_act_out, conv3x3, channel_offset):
        elem_wts = of_wts.acquire(1)

        # top row
        elems_in = of_act_in.acquire(2)
        elem_out = of_act_out.acquire(1)
        conv3x3(
            elems_in[0],
            elems_in[0],
            elems_in[1],
            elem_wts,
            elem_out,
            tensor_w,
            l2_in_c,
            l2_out_c,
            3,
            3,
            0,
            scale_3x3,
            channel_offset,
        )
        of_act_out.release(1)

        # middle rows
        for _ in range_(tensor_h - 2):
            elems_in = of_act_in.acquire(3)
            elem_out = of_act_out.acquire(1)
            conv3x3(
                elems_in[0],
                elems_in[1],
                elems_in[2],
                elem_wts,
                elem_out,
                tensor_w,
                l2_in_c,
                l2_out_c,
                3,
                3,
                1,
                scale_3x3,
                channel_offset,
            )
            of_act_in.release(1)
            of_act_out.release(1)

        # bottom row
        elems_in = of_act_in.acquire(2)
        elem_out = of_act_out.acquire(1)
        conv3x3(
            elems_in[0],
            elems_in[1],
            elems_in[1],
            elem_wts,
            elem_out,
            tensor_w,
            l2_in_c,
            l2_out_c,
            3,
            3,
            2,
            scale_3x3,
            channel_offset,
        )

        of_act_in.release(2)
        of_act_out.release(1)
        of_wts.release(1)

    workers.append(
        Worker(
            conv3x3_fn,
            fn_args=[
                wts_buf_01.cons(),
                of_act_2_3_5.cons(4),
                of_act_3_4.prod(),
                conv3,
                0,
            ],
            tile=Tile(0, 3),
        )
    )
    workers.append(
        Worker(
            conv3x3_fn,
            fn_args=[
                wts_buf_01.cons(),
                of_act_2_3_5.cons(4),
                of_act_5_4.prod(),
                conv3,
                l2_out_c // 2,
            ],
            tile=Tile(0, 5),
        )
    )

    def conv1x1_skip_fn(of_wts, of_in0, of_in1, of_skip, of_out, conv_skip):
        elem_wts = of_wts.acquire(1)
        for _ in range_(tensor_h):
            elem_in0 = of_in0.acquire(1)
            elem_in1 = of_in1.acquire(1)
            elem_skip = of_skip.acquire(1)
            elem_out = of_out.acquire(1)
            conv_skip(
                elem_in0,
                elem_in1,
                elem_wts,
                elem_out,
                elem_skip,
                tensor_w,
                l3_in_c,
                l3_out_c,
                scale_skip,
                skip_scale,
            )
            of_out.release(1)
            of_in0.release(1)
            of_in1.release(1)
            of_skip.release(1)
        of_wts.release(1)

    workers.append(
        Worker(
            conv1x1_skip_fn,
            fn_args=[
                wts_buf_02.cons(),
                of_act_3_4.cons(),
                of_act_5_4.cons(),
                of_skip_buf.cons(),
                of_out_l2l3.prod(),
                conv1_skip,
            ],
            tile=Tile(0, 4),
            stack_size=0xA00,
        )
    )

    rt = Runtime()
    with rt.sequence(act_in_l3_ty, wts_in_l3_ty, act_in_l3_ty) as (I, W, O):
        rt.start(*workers)
        rt.fill(of_act_l3l2.prod(), I)
        rt.fill(of_wts_l3l2.prod(), W)
        rt.drain(of_out_l2l3.cons(), O, wait=True)

    return Program(device, rt).resolve_program()


def _make_argparser():
    p = argparse.ArgumentParser(prog="AIE ResNet bottleneck (int8)")
    add_compile_args(p)
    return p


def _compile_kwargs(opts):
    return {}


def main():
    opts = _make_argparser().parse_args()
    run_design_cli(
        bottleneck,
        opts,
        compile_kwargs=_compile_kwargs,
        device=lambda o: device_from_args(o, n_cols=1),
    )


if __name__ == "__main__":
    main()
