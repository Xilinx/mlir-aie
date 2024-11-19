# vector_vector_add/aie2.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 Advanced Micro Devices, Inc. or its affiliates

# REQUIRES: ryzen_ai
#
# RUN: %python %S/aie2.py > ./aie2.mlir
# RUN: %python aiecc.py --no-aiesim --aie-generate-cdo --aie-generate-npu --no-compile-host --alloc-scheme=basic-sequential --xclbin-name=aie.xclbin --npu-insts-name=insts.txt ./aie2.mlir
# RUN: clang %S/test.cpp -o test.exe -std=c++11 -Wall %xrt_flags -lrt -lstdc++ %test_utils_flags
# RUN: %run_on_npu ./test.exe -x aie.xclbin -k MLIR_AIE -i insts.txt | FileCheck %s
# CHECK: PASS!
import numpy as np
import sys

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.extras.context import mlir_mod_ctx
from aie.helpers.dialects.ext.scf import _for as range_
from aie.dialects import memref


def my_vector_add():
    N = 256
    n = 16
    N_div_n = N // n

    @device(AIEDevice.npu1_4col)
    def device_body():
        # AIE Core Function declarations
        tensor_ty_c = np.ndarray[(N,), np.dtype[np.int32]]
        tensor_ty = np.ndarray[(N // 4,), np.dtype[np.int32]]
        tensor_ty_s = np.ndarray[(n,), np.dtype[np.int32]]

        memref.global_("out", T.memref(16, T.i32()), sym_visibility="public")
        memref.global_("in1", T.memref(16, T.i32()), sym_visibility="public")

        # Tile declarations
        ShimTile = tile(0, 0)
        MemTile = tile(0, 1)
        MemTile2 = tile(1, 1)
        MemTile3 = tile(2, 1)
        ComputeTile2 = tile(0, 2)

        # MemTile elements
        in2_mem_prod_lock = lock(MemTile, lock_id=0, init=0)
        in2_mem_cons_lock = lock(MemTile, lock_id=1, init=1)
        in2_mem_buff_0 = buffer(
            tile=MemTile,
            datatype=tensor_ty,
            name="in2_mem_buff_0",
            initial_value=np.arange(N // 4, dtype=np.int32),
        )

        # MemTile2 elements
        in3_mem_prod_lock = lock(MemTile2, lock_id=0, init=0)
        in3_mem_cons_lock = lock(MemTile2, lock_id=1, init=1)
        in3_mem_buff_0 = buffer(
            tile=MemTile2,
            datatype=tensor_ty,
            name="in3_mem_buff_0",
            initial_value=np.arange(N // 4, (N // 4) * 2, dtype=np.int32),
        )

        # MemTile3 elements
        in4_mem_prod_lock = lock(MemTile3, lock_id=0, init=0)
        in4_mem_cons_lock = lock(MemTile3, lock_id=1, init=2)
        in4_mem_buff_0 = buffer(
            tile=MemTile3,
            datatype=tensor_ty,
            name="in4_mem_buff_0",
            initial_value=np.arange((N // 4) * 2, (N // 4) * 3, dtype=np.int32),
        )
        in4_mem_buff_1 = buffer(
            tile=MemTile3,
            datatype=tensor_ty,
            name="in4_mem_buff_1",
            initial_value=np.arange((N // 4) * 3, N, dtype=np.int32),
        )

        # ComputeTile2 elements
        # Input from ShimTile
        in1_cons_prod_lock = lock(ComputeTile2, lock_id=0, init=1)
        in1_cons_cons_lock = lock(ComputeTile2, lock_id=1, init=0)
        in1_cons_buff_0 = buffer(
            tile=ComputeTile2,
            datatype=tensor_ty_s,
            name="in1_cons_buff_0",
            initial_value=np.arange(n, dtype=np.int32),
        )
        # Input from MemTile
        in2_mem_cons_prod_lock = lock(ComputeTile2, lock_id=2, init=1)
        in2_mem_cons_cons_lock = lock(ComputeTile2, lock_id=3, init=0)
        in2_mem_cons_buff_0 = buffer(
            tile=ComputeTile2,
            datatype=tensor_ty_c,
            name="in2_mem_cons_buff_0",
            initial_value=np.arange(N, dtype=np.int32),
        )
        # Output to ShimTile
        out_prod_lock = lock(ComputeTile2, lock_id=4, init=1)
        out_cons_lock = lock(ComputeTile2, lock_id=5, init=0)
        out_buff_0 = buffer(
            tile=ComputeTile2,
            datatype=tensor_ty_s,
            name="out_buff_0",
            initial_value=np.arange(n, dtype=np.int32),
        )

        flow(ShimTile, WireBundle.DMA, 0, ComputeTile2, WireBundle.DMA, 0)
        flow(MemTile2, WireBundle.DMA, 0, ComputeTile2, WireBundle.DMA, 1)
        flow(ComputeTile2, WireBundle.DMA, 0, ShimTile, WireBundle.DMA, 0)

        # AIE-array data movement
        shim_dma_allocation("in1", DMAChannelDir.MM2S, 0, 0)
        shim_dma_allocation("out", DMAChannelDir.S2MM, 0, 0)

        @memtile_dma(MemTile2)
        def m(block):
            s0 = dma_start(DMAChannelDir.MM2S, 0, dest=block[1], chain=block[5])
            with block[1]:
                use_lock(in2_mem_cons_lock, LockAction.AcquireGreaterEqual)
                dma_bd(in2_mem_buff_0)
                use_lock(in2_mem_prod_lock, LockAction.Release)
                next_bd(block[2])
            with block[2]:
                use_lock(in3_mem_cons_lock, LockAction.AcquireGreaterEqual)
                dma_bd(in3_mem_buff_0)
                use_lock(in3_mem_prod_lock, LockAction.Release)
                next_bd(block[3])
            with block[3]:
                use_lock(in4_mem_cons_lock, LockAction.AcquireGreaterEqual)
                dma_bd(in4_mem_buff_0)
                use_lock(in4_mem_prod_lock, LockAction.Release)
                next_bd(block[4])
            with block[4]:
                use_lock(in4_mem_cons_lock, LockAction.AcquireGreaterEqual)
                dma_bd(in4_mem_buff_1)
                use_lock(in4_mem_prod_lock, LockAction.Release)
                next_bd(block[1])
            with block[5]:
                EndOp()

        @mem(ComputeTile2)
        def m(block):
            s0 = dma_start(DMAChannelDir.S2MM, 0, dest=block[1], chain=block[2])
            with block[1]:
                use_lock(in1_cons_prod_lock, LockAction.AcquireGreaterEqual)
                dma_bd(in1_cons_buff_0)
                use_lock(in1_cons_cons_lock, LockAction.Release)
                next_bd(block[1])
            with block[2]:
                s1 = dma_start(DMAChannelDir.S2MM, 1, dest=block[3], chain=block[4])
            with block[3]:
                use_lock(in2_mem_cons_prod_lock, LockAction.AcquireGreaterEqual)
                dma_bd(in2_mem_cons_buff_0)
                use_lock(in2_mem_cons_cons_lock, LockAction.Release)
                next_bd(block[3])
            with block[4]:
                s2 = dma_start(DMAChannelDir.MM2S, 0, dest=block[5], chain=block[6])
            with block[5]:
                use_lock(out_cons_lock, LockAction.AcquireGreaterEqual)
                dma_bd(out_buff_0)
                use_lock(out_prod_lock, LockAction.Release)
                next_bd(block[5])
            with block[6]:
                EndOp()

        # Set up compute tiles

        # Compute tile 2
        @core(ComputeTile2)
        def core_body():
            # Effective while(1)
            for _ in range_(sys.maxsize):
                # Number of sub-vector "tile" iterations
                use_lock(in2_mem_cons_cons_lock, LockAction.AcquireGreaterEqual)
                for j in range_(N_div_n):
                    use_lock(in1_cons_cons_lock, LockAction.AcquireGreaterEqual)
                    use_lock(out_prod_lock, LockAction.AcquireGreaterEqual)
                    for i in range_(n):
                        out_buff_0[i] = (
                            in2_mem_cons_buff_0[j * N_div_n + i] + in1_cons_buff_0[i]
                        )
                    use_lock(in1_cons_prod_lock, LockAction.Release)
                    use_lock(out_cons_lock, LockAction.Release)
                use_lock(in2_mem_cons_prod_lock, LockAction.Release)

        # To/from AIE-array data movement
        @runtime_sequence(tensor_ty_c, tensor_ty_c, tensor_ty_c)
        def sequence(A, B, C):
            npu_dma_memcpy_nd(metadata="in1", bd_id=1, mem=A, sizes=[1, 1, 1, N])
            npu_dma_memcpy_nd(metadata="out", bd_id=0, mem=C, sizes=[1, 1, 1, N])
            npu_dma_wait("out")


with mlir_mod_ctx() as ctx:
    my_vector_add()
    res = ctx.module.operation.verify()
    if res == True:
        print(ctx.module)
    else:
        print(res)
