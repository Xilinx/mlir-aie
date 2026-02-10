#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc.
import numpy as np
import argparse
import sys

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.extras.context import mlir_mod_ctx

import aie.utils.trace as trace_utils
from aie.utils.trace.events import PortEvent, CoreEvent, MemEvent

from aie.iron.controlflow import range_
from aie.extras.dialects import arith
from aie.helpers.util import np_dtype_to_mlir_type
from aie.extras import types as T
from aie.helpers.util import np_ndarray_type_get_shape


class group2:
    def __init__(
        self,
        _name,
        _computeTile,
        _objectArchive,
        _din,
        # _dout,
    ):
        self.name = _name
        self.computeTile = _computeTile
        self.objectArchive = _objectArchive
        self.din = _din

        din_size = 16 * 43
        din_ty = np.ndarray[(din_size,), np.dtype[np.int16]]

        lut0_size = 32768
        lut1_size = 32768
        lut2_size = 32768
        lut3_size = 20064
        lut0_ty = np.ndarray[(lut0_size,), np.dtype[np.int16]]
        lut1_ty = np.ndarray[(lut1_size,), np.dtype[np.int16]]
        lut2_ty = np.ndarray[(lut2_size,), np.dtype[np.int16]]
        lut3_ty = np.ndarray[(lut3_size,), np.dtype[np.int16]]

        curr_col = int(_computeTile.col)
        curr_col_m1 = int(_computeTile.col) - 1
        curr_row = int(_computeTile.row)
        curr_row_m1 = int(_computeTile.row) - 1
        curr_row_p1 = int(_computeTile.row) + 1
        north_tile = tile(curr_col, curr_row_p1)
        west_tile = tile(curr_col_m1, curr_row)
        south_tile = tile(curr_col, curr_row_m1)

        lut0_arr = np.loadtxt("./data/lut0_group2.txt", delimiter=",")
        lut1_arr = np.loadtxt("./data/lut1_group2.txt", delimiter=",")
        lut2_arr = np.loadtxt("./data/lut2_group2.txt", delimiter=",")
        lut3_arr = np.loadtxt("./data/lut3_group2.txt", delimiter=",")

        lut2_buf = buffer(
            south_tile,
            lut2_ty,
            name="lut2_buf",
            initial_value=np.array(lut2_arr, dtype=np.int16),
        )
        lut0_buf = buffer(
            west_tile,
            lut0_ty,
            name="lut0_buf",
            initial_value=np.array(lut0_arr, dtype=np.int16),
        )
        lut1_buf = buffer(
            north_tile,
            lut1_ty,
            name="lut1_buf",
            initial_value=np.array(lut1_arr, dtype=np.int16),
        )
        lut3_buf = buffer(
            _computeTile,
            lut3_ty,
            name="lut3_buf",
            initial_value=np.array(lut3_arr, dtype=np.int16),
        )

        # kernel definitions
        group2_func = external_func(
            "group2_kernel",
            inputs=[din_ty, lut0_ty, lut1_ty, lut2_ty, lut3_ty],
        )

        output_lock = lock(
            self.computeTile, lock_id=8, init=0
        )  # chooose id=8, objfifo doesn't use it

        @core(self.computeTile, self.objectArchive)
        def core_body():
            for _ in range_(sys.maxsize):
                di = self.din.acquire(ObjectFifoPort.Consume, 1)
                group2_func(di, lut0_buf, lut1_buf, lut2_buf, lut3_buf)
                self.din.release(ObjectFifoPort.Consume, 1)
                use_lock(output_lock, LockAction.Release, value=1)


def group2_impl(dev, trace_size):
    @device(dev)
    def device_body():

        objectArchive = "group2.o"

        ShimTile = tile(1, 0)
        MemTile = tile(1, 1)
        ComputeTile2 = tile(1, 3)

        # define types
        din_size = 16 * 43
        # dout_size = 214*2
        dout_size = 214
        tensorIn_ty = np.ndarray[(din_size,), np.dtype[np.int16]]
        tensorOut_ty = np.ndarray[(dout_size,), np.dtype[np.int32]]
        scalar_ty = np.ndarray[(1,), np.dtype[np.int32]]

        # set up data movement with OFs
        # input tensor
        of_din_L3L2 = object_fifo(
            "of_din_L3L2",
            ShimTile,
            MemTile,
            2,
            tensorIn_ty,
        )
        of_din_L2L1 = object_fifo("of_din_L2L1", MemTile, ComputeTile2, 2, tensorIn_ty)
        object_fifo_link(of_din_L3L2, of_din_L2L1)

        of_dout_L1L3 = object_fifo(
            "of_dout_L1L3", ComputeTile2, ShimTile, 2, tensorOut_ty
        )
        of_dout_L1L3.set_aie_stream(stream_end=0, stream_port=0)

        group2("group2", ComputeTile2, objectArchive, of_din_L2L1)

        # Set up a packet-switched flow from core to shim for tracing information
        tiles_to_trace = [ComputeTile2, ShimTile]
        if trace_size > 0:
            trace_utils.configure_packet_tracing_flow(tiles_to_trace, ShimTile)

        # instruction stream generation
        @runtime_sequence(tensorIn_ty, scalar_ty, tensorOut_ty)
        def sequence(A, B, C):

            if trace_size > 0:
                trace_utils.configure_packet_tracing_aie2(
                    tiles_to_trace=tiles_to_trace,
                    shim=ShimTile,
                    trace_size=trace_size,
                    coretile_events=[
                        CoreEvent.INSTR_EVENT_0,
                        CoreEvent.INSTR_EVENT_1,
                        CoreEvent.INSTR_VECTOR,
                        PortEvent(CoreEvent.PORT_RUNNING_0, 1, True),  # master(1)
                        PortEvent(CoreEvent.PORT_RUNNING_1, 2, True),  # master(2)
                        PortEvent(CoreEvent.PORT_RUNNING_2, 1, False),  # slave(1)
                        # CoreEvent.INSTR_LOCK_ACQUIRE_REQ,
                        # CoreEvent.LOCK_STALL,
                        CoreEvent.INSTR_STREAM_PUT,
                        CoreEvent.STREAM_STALL,
                    ],
                )

            din_task = shim_dma_single_bd_task(
                of_din_L3L2, A, sizes=[1, 1, 1, din_size]
            )
            dout_task = shim_dma_single_bd_task(
                of_dout_L1L3,
                C,
                sizes=[1, 1, 1, dout_size],
                issue_token=True,
            )

            dma_start_task(din_task, dout_task)

            dma_await_task(dout_task)
            # dma_free_task(din_task)

            trace_utils.gen_trace_done_aie2(ShimTile)


if len(sys.argv) < 3:
    raise ValueError(
        "[ERROR] Need at least 4 arguments (dev, in1_size, in2_size, out_size)"
    )

p = argparse.ArgumentParser()
p.add_argument("-d", "--dev", required=True, dest="device", help="AIE Device")
p.add_argument(
    "-t",
    "--trace_size",
    required=False,
    dest="trace_size",
    default=0,
    help="Trace buffer size",
)
opts = p.parse_args(sys.argv[1:])

if opts.device == "npu":
    dev = AIEDevice.npu1
elif opts.device == "npu2":
    dev = AIEDevice.npu2
else:
    raise ValueError("[ERROR] Device name {} is unknown".format(sys.argv[1]))
trace_size = int(opts.trace_size)

with mlir_mod_ctx() as ctx:
    group2_impl(dev, trace_size)
    res = ctx.module.operation.verify()
    if res == True:
        print(ctx.module)
    else:
        print(res)
