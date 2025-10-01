# Copyright (C) 2023, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %PYTHON %s | FileCheck %s

from aie.dialects.aiex import *
from aie.dialects.aie import device, AIEDevice, object_fifo, tile
from aie.extras.dialects.ext import arith
from aie.extras import types as T
from util import construct_and_print_module


# CHECK-LABEL: getTileOp
# CHECK: aiex.getTile
@construct_and_print_module
def getTileOp():
    four = arith.constant(4, index=True)
    two = arith.constant(2, index=True)
    GetTileOp(T.index(), four, two)


# CHECK-LABEL: runtimeSeq
# CHECK: aiex.runtime_sequence @sequence0()
# CHECK: aiex.runtime_sequence @seq1()
@construct_and_print_module
def runtimeSeq():
    @device(AIEDevice.npu1)
    def device_body():
        @runtime_sequence()
        def sequence0():
            npu_write32(0xFFFF, 0xEEEE)

        @runtime_sequence(sym_name="seq1")
        def sequence1():
            npu_write32(0x1111, 0x2222)


# CHECK-LABEL: NpuDmaMemcpyNdOp
# CHECK: aiex.runtime_sequence(%arg0: memref<100xi8>)
@construct_and_print_module
def NpuDmaMemcpyNdOp():
    @device(AIEDevice.npu1)
    def device_body():
        @runtime_sequence(np.ndarray[(100,), np.dtype[np.int8]])
        def sequence(B):
            # CHECK: aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0][1, 1, 1, 100][0, 0, 0, 1], packet = <pkt_type = 1, pkt_id = 4>) {id = 1 : i64, issue_token = true, metadata = @objFifo_out0} : memref<100xi8>
            npu_dma_memcpy_nd(
                metadata="objFifo_out0",
                bd_id=1,
                mem=B,
                offsets=[0, 0, 0, 0],
                sizes=[1, 1, 1, 100],
                strides=[0, 0, 0, 1],
                packet=(1, 4),
                issue_token=True,
            )


# CHECK-LABEL: v8bfp16ebs8Binding
# CHECK: !aie.objectfifo<memref<256x!aiex.bfp<"v8bfp16ebs8">>>
@construct_and_print_module
def v8bfp16ebs8Binding():
    @device(AIEDevice.npu2)
    def device_body():
        P = tile(0, 0)
        C = tile(1, 2)

        object_fifo(
            "dummy", P, C, 1, datatype=np.ndarray[(256,), np.dtype[v8bfp16ebs8]]
        )


# CHECK-LABEL: v16bfp16ebs16Binding
# CHECK: !aie.objectfifo<memref<256x!aiex.bfp<"v16bfp16ebs16">>>
@construct_and_print_module
def v16bfp16ebs16Binding():
    @device(AIEDevice.npu2)
    def device_body():
        P = tile(0, 0)
        C = tile(1, 2)

        object_fifo(
            "dummy", P, C, 1, datatype=np.ndarray[(256,), np.dtype[v16bfp16ebs16]]
        )
