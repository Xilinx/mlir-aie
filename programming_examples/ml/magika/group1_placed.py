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
from aie.helpers.dialects.ext.scf import _for as range_
from aie.helpers.util import np_ndarray_type_get_shape

# tracing definitions
trace_sz_in_bytes = 8192
trace_sz_in_i32s = trace_sz_in_bytes // 4
enableTrace = False


class group1_x1:
    def __init__(
        self,
        _name,
        _id,
        _computeTile,
        _objectArchive,
        _din,
        _dout,
    ):
        self.name = _name
        self.id = _id
        self.computeTile = _computeTile
        self.objectArchive = _objectArchive
        self.din = _din
        self.dout = _dout

        din_size = 8 * 512
        dout_size = 16
        din_ty = np.ndarray[(din_size,), np.dtype[np.int16]]
        dout_ty = np.ndarray[(dout_size,), np.dtype[np.int16]]

        # kernel definitions
        group1_func = external_func(
            f"group1_{id}_kernel",
            inputs=[din_ty, dout_ty],
        )

        @core(self.computeTile, self.objectArchive)
        def core_body():
            for _ in range_(sys.maxsize):
                do = self.dout.acquire(ObjectFifoPort.Produce, 1)
                di = self.din.acquire(ObjectFifoPort.Consume, 1)
                group1_func(di, do)
                self.din.release(ObjectFifoPort.Consume, 1)
                self.dout.release(ObjectFifoPort.Produce, 1)


# -----
# |id2|  Computetile2
# -----
#   |
# -----
# |id1|  ComputeTile1
# -----
#
class group1_x2:
    def __init__(
        self,
        _name,
        _id1,
        _id2,
        _computeTile1,
        _computeTile2,
        _objectArchive,
        _din,
        _dout,
    ):
        self.name = _name
        self.id1 = _id1
        self.id2 = _id2
        self.computeTile1 = _computeTile1
        self.computeTile2 = _computeTile2
        self.objectArchive = _objectArchive
        self.din = _din
        self.dout = _dout

        din_size = 8 * 512
        dout_size = 32
        din_ty = np.ndarray[(din_size,), np.dtype[np.int16]]
        dout_ty = np.ndarray[(dout_size,), np.dtype[np.int16]]
        # dout is a stream connection rather than objfifo

        dout2_size = 16
        dout2_ty = np.ndarray[(dout2_size,), np.dtype[np.int16]]

        of_int = object_fifo(
            "of_int", self.computeTile1, self.computeTile2, 2, dout2_ty
        )

        # kernel definitions
        group1a_func = external_func(
            f"group1_{id1}_kernel",
            inputs=[din_ty, dout2_ty],
        )

        group1b_func = external_func(
            f"group1_{id2}_kernel",
            inputs=[din_ty, dout2_ty, dout_ty],
        )

        @core(self.computeTile1, self.objectArchive)
        def core_body():
            for _ in range_(sys.maxsize):
                do = self.of_int.acquire(ObjectFifoPort.Produce, 1)
                di = self.din.acquire(ObjectFifoPort.Consume, 1)
                group1a_func(di, do)
                self.din.release(ObjectFifoPort.Consume, 1)
                self.of_int.release(ObjectFifoPort.Produce, 1)

        @core(self.computeTile2, self.objectArchive)
        def core_body():
            for _ in range_(sys.maxsize):
                do = self.dout.acquire(ObjectFifoPort.Produce, 1)
                di = self.din.acquire(ObjectFifoPort.Consume, 1)
                di2 = self.of_int.acquire(ObjectFifoPort.Consume, 1)
                group1b_func(di, di2, do)
                self.din.release(ObjectFifoPort.Consume, 1)
                self.of_int.release(ObjectFifoPort.Consume, 1)
                self.dout.release(ObjectFifoPort.Produce, 1)


# -----
# |id3|  Computetile3
# -----
#   |
# -----
# |id2|  Computetile2
# -----
#   |
# -----
# |id1|  ComputeTile1
# -----
#
class group1_x3:
    def __init__(
        self,
        _name,
        _id1,
        _id2,
        _id3,
        _computeTile1,
        _computeTile2,
        _computeTile3,
        _objectArchive,
        _din,
        _dout,
    ):
        self.name = _name
        self.id1 = _id1
        self.id2 = _id2
        self.computeTile1 = _computeTile1
        self.computeTile2 = _computeTile2
        self.objectArchive = _objectArchive
        self.din = _din
        self.dout = _dout

        din_size = 8 * 512
        dout_size = 48
        din_ty = np.ndarray[(din_size,), np.dtype[np.int16]]
        dout_ty = np.ndarray[(dout_size,), np.dtype[np.int16]]
        # dout is a stream connection rather than objfifo

        dout2_size = 16
        dout2_ty = np.ndarray[(dout2_size,), np.dtype[np.int16]]

        of_int2 = object_fifo(
            "of_int2", self.computeTile1, self.computeTile2, 2, dout2_ty
        )

        of_int3 = object_fifo(
            "of_int3", self.computeTile1, self.computeTile2, 2, dout2_ty
        )

        # kernel definitions
        group1a_func = external_func(
            f"group1_{id1}_kernel",
            inputs=[din_ty, dout2_ty],
        )

        group1b_func = external_func(
            f"group1_{id2}_kernel",
            inputs=[din_ty, dout2_ty, dout2_ty, dout_ty],
        )

        group1c_func = external_func(
            f"group1_{id3}_kernel",
            inputs=[din_ty, dout2_ty],
        )

        @core(self.computeTile1, self.objectArchive)
        def core_body():
            for _ in range_(sys.maxsize):
                do = self.of_int.acquire(ObjectFifoPort.Produce, 1)
                di = self.din.acquire(ObjectFifoPort.Consume, 1)
                group1a_func(di, do)
                self.din.release(ObjectFifoPort.Consume, 1)
                self.of_int.release(ObjectFifoPort.Produce, 1)

        @core(self.computeTile2, self.objectArchive)
        def core_body():
            for _ in range_(sys.maxsize):
                do = self.dout.acquire(ObjectFifoPort.Produce, 1)
                di = self.din.acquire(ObjectFifoPort.Consume, 1)
                di2 = self.of_int2.acquire(ObjectFifoPort.Consume, 1)
                di3 = self.of_int3.acquire(ObjectFifoPort.Consume, 1)
                group1b_func(di, di2, di3, do)
                self.din.release(ObjectFifoPort.Consume, 1)
                self.of_int2.release(ObjectFifoPort.Consume, 1)
                self.of_int3.release(ObjectFifoPort.Consume, 1)
                self.dout.release(ObjectFifoPort.Produce, 1)

        @core(self.computeTile3, self.objectArchive)
        def core_body():
            for _ in range_(sys.maxsize):
                do = self.of_int2.acquire(ObjectFifoPort.Produce, 1)
                di = self.din.acquire(ObjectFifoPort.Consume, 1)
                group1b_func(di, do)
                self.din.release(ObjectFifoPort.Consume, 1)
                self.of_int2.release(ObjectFifoPort.Consume, 1)


# class group1:
#     def __init__(
#         self,
#         _name,
#         _computeTile1,
#         _computeTile2,
#         _objectArchive,
#         _din,
#         _dout,
#     ):
#         self.name = _name
#         self.din = _din
#         self.dout = _dout
#         self.computeTile1 = _computeTile1
#         self.computeTile2 = _computeTile2
#         self.objectArchive = _objectArchive

#         data_size = 4096
#         data_ty = np.ndarray[(data_size,), np.dtype[np.int16]]

#         of_int = object_fifo(
#             "of_int", self.computeTile1, self.computeTile2, 2, data_ty
#         )

#         group0a(
#             "group0a",
#             self.computeTile1,
#             self.objectArchive,
#             self.din,
#             of_int,
#         )

#         group0b(
#             "group0b",
#             self.computeTile2,
#             self.objectArchive,
#             of_int,
#             self.dout,
#         )


def group1_impl(dev, trace_size):

    # print("group1_impl")

    @device(dev)
    def device_body():

        objectArchive = "group1.o"

        ShimTile = tile(0, 0)
        MemTile = tile(0, 1)
        ComputeTile2 = tile(0, 2)

        # define types
        din_size = 8 * 512
        dout_size = 16
        tensorIn_ty = np.ndarray[(din_size,), np.dtype[np.int16]]
        tensorOut_ty = np.ndarray[(dout_size,), np.dtype[np.int16]]
        scalar_ty = np.ndarray[(1,), np.dtype[np.int32]]

        if enableTrace:
            packetflow(
                30, ComputeTile2, WireBundle.Trace, 0, ShimTile, WireBundle.DMA, 1
            )

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

        group1("group1", "01", ComputeTile2, objectArchive, of_din_L2L1, of_dout_L1L2)

        # instruction stream generation
        @runtime_sequence(tensorIn_ty, scalar_ty, tensorOut_ty)
        def sequence(A, B, C):

            # if enableTrace:

            din_task = shim_dma_single_bd_task(
                of_din_L3L2, A, sizes=[1, 1, 1, din_size]
            )
            dout_task = shim_dma_single_bd_task(
                of_dout_L2L3,
                C,
                sizes=[1, 1, 1, dout_size],
                issue_token=True,
            )

            dma_start_task(din_task, dout_task)

            dma_await_task(dout_task)
            # dma_free_task(din_task)


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
    dev = AIEDevice.npu1_1col
elif opts.device == "npu2":
    dev = AIEDevice.npu2
else:
    raise ValueError("[ERROR] Device name {} is unknown".format(sys.argv[1]))
trace_size = int(opts.trace_size)

with mlir_mod_ctx() as ctx:
    group1_impl(dev, trace_size)
    res = ctx.module.operation.verify()
    if res == True:
        print(ctx.module)
    else:
        print(res)
