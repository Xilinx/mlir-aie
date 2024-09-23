#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2023 AMD Inc.

from aie.extras.context import mlir_mod_ctx

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.dialects.scf import *
from aie.extras.dialects.ext.scf import _for as range_


def my_matmul():
    M = 288
    K = 288
    m = 32
    k = 32

    n_cores = 1

    A_sz = M * K
    B_sz = K
    C_sz = M
    C_sz_div_n_cores = C_sz // n_cores

    M_div_m = M // m
    M_div_m_div_n_cores = M // (m * n_cores)
    K_div_k = K // k

    m_x_k = m * k
    m_x_K = m * K

    # FIXME vectorized kernel is currently erroneous
    vectorized = False

    dtype_in = T.i16
    dtype_in_str = "i16"
    dtype_out = T.i32
    dtype_out_str = "i32"

    with mlir_mod_ctx() as ctx:

        @device(AIEDevice.npu1_4col)
        def device_body():
            memRef_inA_ty = T.memref(m * k, dtype_in())
            memRef_inB_ty = T.memref(k, dtype_in())
            memRef_outC_ty = T.memref(m, dtype_out())
            memRef_A_ty = T.memref(m, k, dtype_in())

            # AIE Core Function declarations
            zero_scalar = external_func(
                f"zero_scalar_{dtype_out_str}", inputs=[memRef_outC_ty]
            )
            zero = external_func(
                f"zero_vectorized_{dtype_out_str}", inputs=[memRef_outC_ty]
            )
            matvec_scalar = external_func(
                f"matvec_scalar_{dtype_in_str}_{dtype_out_str}",
                inputs=[memRef_A_ty, memRef_inB_ty, memRef_outC_ty],
            )
            matvec = external_func(
                f"matvec_vectorized_{dtype_in_str}_{dtype_out_str}",
                inputs=[memRef_A_ty, memRef_inB_ty, memRef_outC_ty],
            )

            # Tile declarations
            ShimTile0 = tile(0, 0)
            ShimTile1 = tile(1, 0)
            ShimTile2 = tile(2, 0)
            ShimTile3 = tile(3, 0)
            ShimTiles = [ShimTile0, ShimTile1, ShimTile2, ShimTile3]
            MemTile0 = tile(0, 1)
            MemTile1 = tile(1, 1)
            MemTile2 = tile(2, 1)
            MemTile3 = tile(3, 1)
            MemTiles = [MemTile0, MemTile1, MemTile2, MemTile3]
            ComputeTile0 = tile(0, 2)
            ComputeTile1 = tile(1, 2)
            ComputeTile2 = tile(2, 2)
            ComputeTile3 = tile(3, 2)
            cores = [ComputeTile0, ComputeTile1, ComputeTile2, ComputeTile3]
            memA_fifo_names = ["memA0", "memA1", "memA2", "memA3"]
            memA_fifos = {}
            inA_fifo_names = ["inA0", "inA1", "inA2", "inA3"]
            inA_fifos = {}
            inB_fifo_names = ["inB"]
            inB_fifos = {}
            outC_fifo_names = ["outC0", "outC1", "outC2", "outC3"]
            outC_fifos = {}

            # AIE-array data movement with object fifos
            # Input A
            for i in range(n_cores):
                memA_fifos[memA_fifo_names[i]] = object_fifo(
                    memA_fifo_names[i],
                    ShimTiles[i],
                    MemTiles[i],
                    2,
                    memRef_inA_ty,
                )
                inA_fifos[inA_fifo_names[i]] = object_fifo(
                    inA_fifo_names[i],
                    MemTiles[i],
                    cores[i],
                    2,
                    memRef_A_ty,
                    (
                        [
                            (k // 2 // 2, 2),
                            (m, k),
                            (2, 1),
                        ]
                        if vectorized
                        else []
                    ),  # transpose at 4-byte (2xbf16) granularity
                )
                object_fifo_link(
                    memA_fifos[memA_fifo_names[i]], inA_fifos[inA_fifo_names[i]]
                )

            # Input B
            inB_fifos[inB_fifo_names[0]] = object_fifo(
                inB_fifo_names[0],
                ShimTiles[1 % n_cores],
                cores[0:n_cores],
                2,
                memRef_inB_ty,
            )

            # Output C
            for i in range(n_cores):
                outC_fifos[outC_fifo_names[i]] = object_fifo(
                    outC_fifo_names[i],
                    cores[i],
                    ShimTiles[i],
                    2,
                    memRef_outC_ty,
                )

            # Set up compute tiles
            for i in range(n_cores):
                # Compute tile i
                @core(cores[i], f"mv_{m}x{k}.o")
                def core_body():
                    for _ in range_(0xFFFFFFFF):
                        elem_out = outC_fifos[outC_fifo_names[i]].acquire(
                            ObjectFifoPort.Produce,
                            1,
                        )
                        if vectorized or True:
                            call(zero, [elem_out])
                        else:
                            call(zero_scalar, [elem_out])

                        for _ in range_(K_div_k):
                            elem_in_a = inA_fifos[inA_fifo_names[i]].acquire(
                                ObjectFifoPort.Consume,
                                1,
                            )
                            elem_in_b = inB_fifos[inB_fifo_names[0]].acquire(
                                ObjectFifoPort.Consume,
                                1,
                            )
                            if vectorized:
                                call(matvec, [elem_in_a, elem_in_b, elem_out])
                            else:
                                call(matvec_scalar, [elem_in_a, elem_in_b, elem_out])
                            inA_fifos[inA_fifo_names[i]].release(
                                ObjectFifoPort.Consume,
                                1,
                            )
                            inB_fifos[inB_fifo_names[0]].release(
                                ObjectFifoPort.Consume,
                                1,
                            )

                        outC_fifos[outC_fifo_names[i]].release(
                            ObjectFifoPort.Produce,
                            1,
                        )

            # To/from AIE-array data movement

            @runtime_sequence(
                T.memref(A_sz, dtype_in()),
                T.memref(B_sz, dtype_in()),
                T.memref(C_sz, dtype_out()),
            )
            def sequence(A, B, C):
                npu_dma_memcpy_nd(
                    metadata=inB_fifo_names[0],
                    bd_id=2,
                    mem=B,
                    sizes=[M_div_m_div_n_cores, 1, 1, K],
                    strides=[0, 0, 0, 1],
                )
                for i in range(n_cores):
                    A_offset = i * M_div_m_div_n_cores * m * K
                    C_offset = i * M_div_m_div_n_cores * m
                    npu_dma_memcpy_nd(
                        metadata=memA_fifo_names[i],
                        bd_id=1,
                        mem=A,
                        offsets=[0, 0, 0, A_offset],
                        sizes=[M_div_m_div_n_cores, K_div_k, m, k],
                        strides=[m_x_K, k, K, 1],
                    )
                    npu_dma_memcpy_nd(
                        metadata=outC_fifo_names[i],
                        bd_id=0,
                        mem=C,
                        offsets=[0, 0, 0, C_offset],
                        sizes=[1, 1, 1, C_sz_div_n_cores],
                        strides=[0, 0, 0, 1],
                    )

                for i in range(n_cores):
                    npu_sync(column=i, row=0, direction=0, channel=0)

    print(ctx.module)


my_matmul()
