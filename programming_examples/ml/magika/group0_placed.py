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


class group0a:
    def __init__(
        self,
        _name,
        _computeTile,
        _objectArchive,
        _din,
        _dout,
    ):
        self.name = _name
        self.computeTile = _computeTile
        self.objectArchive = _objectArchive
        self.din = _din
        self.dout = _dout

        din_size = 2048
        # dout_size = 4096*32
        dout_size = 4096
        din_ty = np.ndarray[(din_size,), np.dtype[np.int16]]
        dout_ty = np.ndarray[(dout_size,), np.dtype[np.int16]]
        scalar_ty = np.int32

        lut0a_size = 16448
        lut0a_ty = np.ndarray[(lut0a_size,), np.dtype[np.int16]]
        lut0a_arr = np.loadtxt("./data/lut0a_group0.txt", delimiter=",")

        lut0a_buf = buffer(
            _computeTile,
            lut0a_ty,
            name="lut0a_buf",
            address=0x1000,
            initial_value=np.array(lut0a_arr, dtype=np.int16),
        )

        # kernel definitions
        group0a_func = external_func(
            "group0a_kernel",
            inputs=[din_ty, dout_ty, lut0a_ty, scalar_ty, scalar_ty],
        )

        @core(self.computeTile, self.objectArchive, stack_size=4096)
        def core_body():
            for _ in range_(sys.maxsize):
                di = self.din.acquire(ObjectFifoPort.Consume, 1)
                for xid in range_(4):
                    xid_i32 = arith.index_cast(xid, to=np_dtype_to_mlir_type(np.int32))
                    for cid in range_(8):  # 64/8
                        cid_i32 = arith.index_cast(
                            cid, to=np_dtype_to_mlir_type(np.int32)
                        )
                        do = self.dout.acquire(ObjectFifoPort.Produce, 1)
                        group0a_func(di, do, lut0a_buf, xid_i32, cid_i32)
                        self.dout.release(ObjectFifoPort.Produce, 1)
                self.din.release(ObjectFifoPort.Consume, 1)


class group0b:
    def __init__(
        self,
        _name,
        _computeTile,
        _objectArchive,
        _din,
        _dout,
    ):
        self.name = _name
        self.computeTile = _computeTile
        self.objectArchive = _objectArchive
        self.din = _din
        self.dout = _dout

        din_size = 4096
        dout_size = 4096
        din_ty = np.ndarray[(din_size,), np.dtype[np.int16]]
        dout_ty = np.ndarray[(dout_size,), np.dtype[np.int16]]

        lut0b_a_size = 8704
        lut0b_b_size = 4608
        lut0b_a_ty = np.ndarray[(lut0b_a_size,), np.dtype[np.int16]]
        lut0b_b_ty = np.ndarray[(lut0b_b_size,), np.dtype[np.int16]]
        lut0b_a_arr = np.loadtxt("./data/lut0b_a_group0.txt", delimiter=",")
        lut0b_b_arr = np.loadtxt("./data/lut0b_b_group0.txt", delimiter=",")

        lut0b_a_buf = buffer(
            _computeTile,
            lut0b_a_ty,
            name="lut0b_a_buf",
            initial_value=np.array(lut0b_a_arr, dtype=np.int16),
        )

        lut0b_b_buf = buffer(
            _computeTile,
            lut0b_b_ty,
            name="lut0b_b_buf",
            initial_value=np.array(lut0b_b_arr, dtype=np.int16),
        )

        # kernel definitions
        group0b_func = external_func(
            "group0b_kernel",
            inputs=[din_ty, dout_ty, lut0b_a_ty, lut0b_b_ty],
        )

        @core(self.computeTile, self.objectArchive)
        def core_body():
            for _ in range_(sys.maxsize):
                for ite in range_(32):  # 256/8
                    do = self.dout.acquire(ObjectFifoPort.Produce, 1)
                    di = self.din.acquire(ObjectFifoPort.Consume, 1)
                    group0b_func(di, do, lut0b_a_buf, lut0b_b_buf)
                    self.din.release(ObjectFifoPort.Consume, 1)
                    self.dout.release(ObjectFifoPort.Produce, 1)


class group0:
    def __init__(
        self,
        _name,
        _computeTile1,
        _computeTile2,
        _din,
        _dout,
    ):
        self.name = _name
        self.din = _din
        self.dout = _dout
        self.computeTile1 = _computeTile1
        self.computeTile2 = _computeTile2

        data_size = 4096
        data_ty = np.ndarray[(data_size,), np.dtype[np.int16]]

        of_int = object_fifo("of_int", self.computeTile1, self.computeTile2, 2, data_ty)

        group0a(
            "group0a",
            self.computeTile1,
            "group0a.o",
            self.din,
            of_int,
        )

        group0b(
            "group0b",
            self.computeTile2,
            "group0b.o",
            of_int,
            self.dout,
        )


def group0_impl(dev, trace_size):
    @device(dev)
    def device_body():

        ShimTile = tile(0, 0)
        MemTile = tile(0, 1)
        ComputeTile2 = tile(0, 2)
        ComputeTile3 = tile(0, 3)

        # define types
        din_size = 2048
        dout_size = 4096
        tensorIn_ty = np.ndarray[(din_size,), np.dtype[np.int16]]
        tensorOut_ty = np.ndarray[(dout_size,), np.dtype[np.int16]]
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

        # output tensor
        of_dout_L1L2 = object_fifo(
            "of_dout_L1L2", ComputeTile3, MemTile, 2, tensorOut_ty
        )
        of_dout_L2L3 = object_fifo("of_dout_L2L3", MemTile, ShimTile, 2, tensorOut_ty)
        object_fifo_link(of_dout_L1L2, of_dout_L2L3)

        group0(
            "group0",
            ComputeTile2,
            ComputeTile3,
            of_din_L2L1,
            of_dout_L1L2,
        )

        tiles_to_trace = [ComputeTile2, ComputeTile3, ShimTile]
        if trace_size > 0:
            trace_utils.configure_packet_tracing_flow(tiles_to_trace, ShimTile)

        # instruction stream generation
        @runtime_sequence(tensorIn_ty, tensorOut_ty, scalar_ty)
        def sequence(A, C, notUsed):

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
                of_dout_L2L3,
                C,
                sizes=[1, 1, 1, dout_size * 32],
                issue_token=True,
            )

            dma_start_task(din_task, dout_task)

            dma_await_task(dout_task)
            # dma_free_task(din_task)

            trace_utils.gen_trace_done_aie2(ShimTile)


if len(sys.argv) < 1:
    raise ValueError("[ERROR] Need at least 1 arguments (dev)")

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
    dev = AIEDevice.npu1_1col
elif opts.device == "npu2":
    dev = AIEDevice.npu2
else:
    raise ValueError("[ERROR] Device name {} is unknown".format(sys.argv[1]))
trace_size = int(opts.trace_size)

with mlir_mod_ctx() as ctx:
    group0_impl(dev, trace_size)
    res = ctx.module.operation.verify()
    if res == True:
        print(ctx.module)
    else:
        print(res)
