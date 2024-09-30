#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2023 AMD Inc.
import numpy as np

from aie.extras.context import mlir_mod_ctx
from aie.dialects.aie import *
from aie.dialects.aiex import *
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

    dtype_in = np.int16
    dtype_in_str = "i16"
    dtype_out = np.int32
    dtype_out_str = "i32"

    with mlir_mod_ctx() as ctx:

        @device(AIEDevice.npu1_4col)
        def device_body():
            inA_ty = np.ndarray[dtype_in, (m * k,)]
            inB_ty = np.ndarray[dtype_in, (k,)]
            outC_ty = np.ndarray[dtype_in, (m,)]
            A_ty = np.ndarray[dtype_in, (m, k)]

            # AIE Core Function declarations
            func_type = "vectorized" if vectorized else "scalar"
            zero = external_func(f"zero_{func_type}_{dtype_out_str}", inputs=[outC_ty])
            matvec = external_func(
                f"matvec_{func_type}_{dtype_in_str}_{dtype_out_str}",
                inputs=[A_ty, inB_ty, outC_ty],
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
            memA_fifos = []
            inA_fifos = []
            outC_fifos = []

            # AIE-array data movement with object fifos
            # Input A
            for i in range(n_cores):
                memA_fifos.append(
                    object_fifo(f"memA{i}", ShimTiles[i], MemTiles[i], 2, inA_ty)
                )
                inA_fifos.append(
                    object_fifo(
                        f"inA{i}",
                        MemTiles[i],
                        cores[i],
                        2,
                        A_ty,
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
                )
                object_fifo_link(memA_fifos[i], inA_fifos[i])

                # Output C
                outC_fifos.append(
                    object_fifo(f"outC{i}", cores[i], ShimTiles[i], 2, outC_ty)
                )

            # Input B
            inB_fifo = object_fifo(
                "inB", ShimTiles[1 % n_cores], cores[0:n_cores], 2, inB_ty
            )

            # Set up compute tiles
            for i in range(n_cores):
                # Compute tile i
                @core(cores[i], f"mv_{m}x{k}.o")
                def core_body():
                    for _ in range_(0xFFFFFFFF):
                        elem_out = outC_fifos[i].acquire(
                            ObjectFifoPort.Produce,
                            1,
                        )
                        zero(elem_out)

                        for _ in range_(K_div_k):
                            elem_in_a = inA_fifos[i].acquire(ObjectFifoPort.Consume, 1)
                            elem_in_b = inB_fifo.acquire(ObjectFifoPort.Consume, 1)
                            matvec(elem_in_a, elem_in_b, elem_out)
                            inA_fifos[i].release(ObjectFifoPort.Consume, 1)
                            inB_fifo.release(ObjectFifoPort.Consume, 1)

                        outC_fifos[i].release(ObjectFifoPort.Produce, 1)

            # To/from AIE-array data movement

            @runtime_sequence(
                np.ndarray[dtype_in, (A_sz,)],
                np.ndarray[dtype_in, (B_sz,)],
                np.ndarray[dtype_out, (C_sz,)],
            )
            def sequence(A, B, C):
                npu_dma_memcpy_nd(
                    metadata=inB_fifo,
                    bd_id=2,
                    mem=B,
                    sizes=[M_div_m_div_n_cores, 1, 1, K],
                    strides=[0, 0, 0, 1],
                )
                for i in range(n_cores):
                    A_offset = i * M_div_m_div_n_cores * m * K
                    C_offset = i * M_div_m_div_n_cores * m
                    npu_dma_memcpy_nd(
                        metadata=memA_fifos[i],
                        bd_id=1,
                        mem=A,
                        offsets=[0, 0, 0, A_offset],
                        sizes=[M_div_m_div_n_cores, K_div_k, m, k],
                        strides=[m_x_K, k, K, 1],
                    )
                    npu_dma_memcpy_nd(
                        metadata=outC_fifos[i],
                        bd_id=0,
                        mem=C,
                        offsets=[0, 0, 0, C_offset],
                        sizes=[1, 1, 1, C_sz_div_n_cores],
                        strides=[0, 0, 0, 1],
                    )
                dma_wait(*outC_fifos)

    print(ctx.module)


my_matmul()
