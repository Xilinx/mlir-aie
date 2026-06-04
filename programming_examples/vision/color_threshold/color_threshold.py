# vision/color_threshold/color_threshold.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024-2026 AMD Inc.
"""Color threshold -- ``@iron.jit`` parallel-channel threshold pipeline.

A line-based RGBA image is split into its four channels (R, G, B, A); each
channel is independently thresholded on its own AIE compute tile; the four
thresholded channels are joined back into RGBA.  Threshold value/max/type
are read at runtime from per-channel ``Buffer(use_write_rtp=True)`` RTPs,
gated by ``WorkerRuntimeBarrier``s so each worker waits for the runtime
sequence's ``set_rtps()`` before reading.
"""

import argparse

import numpy as np

import aie.iron as iron
from aie.iron import (
    Buffer,
    Compile,
    In,
    ObjectFifo,
    Out,
    Program,
    Runtime,
    Worker,
    WorkerRuntimeBarrier,
    kernels,
)
from aie.iron.device import device_from_args

from aie.extras.dialects import arith
from aie.helpers.util import np_ndarray_type_get_shape
from aie.dialects.aie import T

from aie.utils.hostruntime.argparse import add_compile_args
from aie.utils.hostruntime.cli import run_design_cli
from aie.utils.verify import assert_pass


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def color_threshold(
    in_tensor: In,
    _b_unused: In,
    out_tensor: Out,
    *,
    width: Compile[int] = 1920,
    height: Compile[int] = 1080,
):
    line_width = width
    line_width_channels = width * 4  # 4 channels (RGBA)
    tensor_size = width * height

    tensor_ty = np.ndarray[(tensor_size,), np.dtype[np.int8]]
    line_channels_ty = np.ndarray[(line_width_channels,), np.dtype[np.uint8]]
    line_ty = np.ndarray[(line_width,), np.dtype[np.uint8]]
    unused_ty = np.ndarray[(32,), np.dtype[np.int32]]  # 2nd buffer is unused

    threshold_line = kernels.threshold(line_width=line_width, dtype=np.uint8)

    in_oob_l3l2 = ObjectFifo(line_channels_ty, name="inOOB_L3L2")
    of_offsets = [np.prod(np_ndarray_type_get_shape(line_ty)) * i for i in range(4)]
    in_oob_l2l1s = in_oob_l3l2.cons().split(
        of_offsets,
        obj_types=[line_ty] * 4,
        names=[f"inOOB_L2L1_{i}" for i in range(4)],
    )

    out_oob_l2l3 = ObjectFifo(line_channels_ty, name="outOOB_L2L3")
    out_oob_l1l2s = out_oob_l2l3.prod().join(
        of_offsets,
        obj_types=[line_ty] * 4,
        names=[f"outOOB_L1L2_{i}" for i in range(4)],
    )

    rtps = [
        Buffer(
            np.ndarray[(16,), np.dtype[np.int32]],
            name=f"rtp{i}",
            use_write_rtp=True,
        )
        for i in range(4)
    ]
    worker_barriers = [WorkerRuntimeBarrier() for _ in range(4)]

    def core_fn(of_in, of_out, my_rtp, threshold_fn, barrier):
        # RTPs written from the instruction stream must be synchronized with the
        # runtime sequence via a barrier.
        barrier.wait_for_value(1)
        threshold_value = arith.trunci(T.i16(), my_rtp[0])
        max_value = arith.trunci(T.i16(), my_rtp[1])
        threshold_type = arith.trunci(T.i8(), my_rtp[2])

        elem_in = of_in.acquire(1)
        elem_out = of_out.acquire(1)
        threshold_fn(
            elem_in,
            elem_out,
            line_width,
            threshold_value,
            max_value,
            threshold_type,
        )
        of_in.release(1)
        of_out.release(1)

    workers = [
        Worker(
            core_fn,
            [
                in_oob_l2l1s[i].cons(),
                out_oob_l1l2s[i].prod(),
                rtps[i],
                threshold_line,
                worker_barriers[i],
            ],
        )
        for i in range(4)
    ]

    rt = Runtime()
    with rt.sequence(tensor_ty, unused_ty, tensor_ty) as (i_in, _b, o_out):

        def set_rtps(*args):
            for rtp in args:
                rtp[0] = 50
                rtp[1] = 255
                rtp[2] = 0

        rt.inline_ops(set_rtps, rtps)

        for i in range(4):
            rt.set_barrier(worker_barriers[i], 1)

        rt.start(*workers)
        rt.fill(in_oob_l3l2.prod(), i_in)
        rt.drain(out_oob_l2l3.cons(), o_out, wait=True)

    return Program(iron.get_current_device(), rt).resolve_program()


def _make_argparser():
    p = argparse.ArgumentParser(prog="AIE Color Threshold")
    add_compile_args(p)
    p.add_argument("-W", "--width", type=int, default=1920)
    p.add_argument("-H", "--height", type=int, default=1080)
    return p


def _compile_kwargs(opts):
    return dict(width=opts.width, height=opts.height)


def _run_and_verify(opts):
    """JIT-compile + run + verify against a numpy reference.

    Every byte of the input is independently thresholded by one of the four
    channel workers; all workers apply the same binary-threshold
    (mode 0: ``out = (in > 50) ? 255 : 0`` -- strict ``>``, matching the
    kernel's ``aie::lt(thresh, data)``).  The reference therefore collapses
    to a single ``np.where`` over the whole buffer regardless of how the
    cons().split() distributes bytes to workers.
    """
    tensor_size = opts.width * opts.height
    rng = np.random.default_rng(0)
    in_np = rng.integers(-128, 127, size=(tensor_size,), dtype=np.int8)
    unused_np = np.zeros((32,), dtype=np.int32)
    out_np = np.zeros((tensor_size,), dtype=np.int8)

    in_t = iron.tensor(in_np, dtype=np.int8, device="npu")
    unused_t = iron.tensor(unused_np, dtype=np.int32, device="npu")
    out_t = iron.tensor(out_np, dtype=np.int8, device="npu")

    color_threshold(in_t, unused_t, out_t, **_compile_kwargs(opts))

    # The kernel sees uint8 bytes; reinterpret the int8 host buffer.
    in_uint8 = in_np.view(np.uint8)
    expected_uint8 = np.where(in_uint8 > 50, np.uint8(255), np.uint8(0))
    expected = expected_uint8.view(np.int8)

    actual = out_t.numpy()
    n_mismatch = int(np.sum(actual != expected))
    assert_pass(
        actual,
        expected,
        fail_msg=f"{n_mismatch} byte(s) mismatch the binary-threshold reference",
    )


def main():
    opts = _make_argparser().parse_args()
    run_design_cli(
        color_threshold,
        opts,
        compile_kwargs=_compile_kwargs,
        run_and_verify=_run_and_verify,
        device=device_from_args,
    )


if __name__ == "__main__":
    main()
