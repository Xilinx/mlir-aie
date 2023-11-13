#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2023 AMD Inc.

from aie.ir import *
from aie.dialects.func import *
from aie.dialects.scf import *
from aie.dialects.aie import *
from aie.dialects.aiex import *

M = 128
K = 128
N = 128
m = 64
k = 32
n = 64
word_size_in = 2
word_size_out = 2

A_sz_in_i32s = M*K*word_size_in//4
B_sz_in_i32s = K*N*word_size_in//4
C_sz_in_bytes = M*N*word_size_out
C_sz_in_i32s = C_sz_in_bytes//4

M_div_m = M // m
K_div_k = K // k
N_div_n = N // n
tiles = M_div_m * N_div_n

# Matrix A: MxK, submatrices a: mxk
k_in_i32s     = k*word_size_in//4
K_in_i32s     = K*word_size_in//4

# Matrix B: KxN, submatrices b: kxn
n_in_i32s     = n*word_size_in//4
N_in_i32s     = N*word_size_in//4
k_x_N_in_i32s = k*N*word_size_in//4

# Output Matrix C: MxN
n_in_i32s_out     = n*word_size_out//4
N_in_i32s_out     = N*word_size_out//4
m_x_N_in_i32s_out = m*N*word_size_out//4

vectorized = True

@constructAndPrintInModule
def my_matmul():
    @device("ipu")
    def deviceBody():
        in_ty = IntegerType.get_signless(16)
        out_ty = IntegerType.get_signless(16)
        memRef_A_ty = MemRefType.get((m,k,), in_ty)
        memRef_B_ty = MemRefType.get((k,n,), in_ty)
        memRef_C_ty = MemRefType.get((m,n,), out_ty)

        zero_scalar = privateFunc("zero_scalar_i16", inputs = [memRef_C_ty])
        zero = privateFunc("zero_i16", inputs = [memRef_C_ty])
        matmul_scalar = privateFunc("matmul_scalar_i16_i16", inputs = [memRef_A_ty, memRef_B_ty, memRef_C_ty])
        matmul = privateFunc("matmul_i16_i16", inputs = [memRef_A_ty, memRef_B_ty, memRef_C_ty])

        S = Tile(0, 0)
        M = Tile(0, 1)
        T = Tile(0, 2)

        OrderedObjectBuffer("inA", S, T, 2, memRef_A_ty)
        OrderedObjectBuffer("inB", S, T, 2, memRef_B_ty)
        OrderedObjectBuffer("outC", T, S, 2, memRef_C_ty)

        @core(T, "mm.o")
        def coreBody():
            @forLoop(lowerBound = 0, upperBound = 0XFFFFFFFF, step = 1)
            def loopReps():
                @forLoop(lowerBound = 0, upperBound = tiles, step = 1)
                def loopTile():
                    elemOut = Acquire("outC", "Produce", 1, memRef_C_ty).acquiredElem() 
                    if vectorized:
                        call(zero, [elemOut])
                    else:
                        call(zero_scalar, [elemOut])

                    @forLoop(lowerBound = 0, upperBound = K_div_k, step = 1)
                    def loopK():
                        elemInA = Acquire("inA", "Consume", 1, memRef_A_ty).acquiredElem()
                        elemInB = Acquire("inB", "Consume", 1, memRef_B_ty).acquiredElem()
                        if vectorized:
                            call(matmul, [elemInA, elemInB, elemOut])
                        else:
                            call(matmul_scalar, [elemInA, elemInB, elemOut])
                        Release("inA", "Consume", 1)
                        Release("inB", "Consume", 1)

                    Release("outC", "Produce", 1)


        int32_ty = IntegerType.get_signless(32)
        memRef_Ain_ty  =  MemRefType.get((A_sz_in_i32s,), int32_ty)
        memRef_Bin_ty  =  MemRefType.get((B_sz_in_i32s,), int32_ty)
        memRef_Cout_ty =  MemRefType.get((C_sz_in_i32s,), int32_ty)
        @FuncOp.from_py_func(memRef_Ain_ty, memRef_Bin_ty, memRef_Cout_ty)
        def sequence(A, B, C):
            # only do 5 tile rows at a time before synchronizing, so we can reuse BDs
            rows_per_block = 5
            for tile_row_block in range((M_div_m+rows_per_block-1)//rows_per_block):
                C_row_offset_in_i32s = tile_row_block*rows_per_block*m*N*word_size_out//4
                num_tile_rows = min([rows_per_block, M_div_m-tile_row_block*rows_per_block])
                IpuDmaMemcpyNd(metadata = "outC", bd_id = 0, mem = C, offsets = [0, 0, 0, C_row_offset_in_i32s], 
                                                                      lengths = [num_tile_rows, N_div_n, m, n_in_i32s_out], 
                                                                      strides = [m_x_N_in_i32s_out, n_in_i32s_out, N_in_i32s_out])
                for tile_row in range(num_tile_rows):
                    A_row_offset_in_i32s = ((tile_row_block*rows_per_block)+tile_row)*m*K*word_size_in//4
                    IpuDmaMemcpyNd(metadata = "inA", bd_id = 2*tile_row+1, mem = A, offsets = [0, 0, 0, A_row_offset_in_i32s],
                                                                                    lengths = [N_div_n, K_div_k, m, k_in_i32s], 
                                                                                    strides = [0, k_in_i32s, K_in_i32s])
                    IpuDmaMemcpyNd(metadata = "inB", bd_id = 2*tile_row+2, mem = B, lengths = [N_div_n, K_div_k, k, n_in_i32s], 
                                                                                    strides = [n_in_i32s, k_x_N_in_i32s, N_in_i32s])

                IpuSync(column = 0, row = 0, direction = 0, channel = 0)
