//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2022-2024 Advanced Micro Devices, Inc. or its affiliates
// Copyright (C) 2020-2022, Xilinx Inc.
//
//===----------------------------------------------------------------------===//

module {
  aie.device(npu1_1col) {
    memref.global "public" @objFifo_in0 : memref<64xi32>
    memref.global "public" @objFifo_out0 : memref<64xi32>

    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    %objFifo_in1_cons_buff_0 = aie.buffer(%tile_0_2) {sym_name = "objFifo_in1_cons_buff_0"} : memref<8xi32>
    %objFifo_in1_cons_buff_1 = aie.buffer(%tile_0_2) {sym_name = "objFifo_in1_cons_buff_1"} : memref<8xi32>
    %objFifo_out1_buff_0 = aie.buffer(%tile_0_2) {sym_name = "objFifo_out1_buff_0"} : memref<8xi32>
    %objFifo_out1_buff_1 = aie.buffer(%tile_0_2) {sym_name = "objFifo_out1_buff_1"} : memref<8xi32>
    %constant_buffer = aie.buffer(%tile_0_2) {sym_name = "constant_buffer"} : memref<8xi32>

    %objFifo_in1_cons_prod_lock = aie.lock(%tile_0_2, 0) {init = 2 : i32, sym_name = "objFifo_in1_cons_prod_lock"}
    %objFifo_in1_cons_cons_lock = aie.lock(%tile_0_2, 1) {init = 0 : i32, sym_name = "objFifo_in1_cons_cons_lock"}
    %objFifo_out1_prod_lock = aie.lock(%tile_0_2, 2) {init = 2 : i32, sym_name = "objFifo_out1_prod_lock"}
    %objFifo_out1_cons_lock = aie.lock(%tile_0_2, 3) {init = 0 : i32, sym_name = "objFifo_out1_cons_lock"}

    aie.flow(%tile_0_0, DMA : 0, %tile_0_2, DMA : 0)
    aie.flow(%tile_0_2, DMA : 0, %tile_0_0, DMA : 0)

    %core_0_2 = aie.core(%tile_0_2) {
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c1_i32 = arith.constant 1 : i32
      %c2 = arith.constant 2 : index
      scf.for %arg0 = %c0 to %c8 step %c2 {
        aie.use_lock(%objFifo_in1_cons_cons_lock, AcquireGreaterEqual, 1)
        aie.use_lock(%objFifo_out1_prod_lock, AcquireGreaterEqual, 1)

        scf.for %arg1 = %c0 to %c8 step %c1 {
          %0 = memref.load %objFifo_in1_cons_buff_0[%arg1] : memref<8xi32>
          %1 = memref.load %constant_buffer[%arg1] : memref<8xi32>
          %2 = arith.addi %0, %1 : i32
          memref.store %2, %objFifo_out1_buff_0[%arg1] : memref<8xi32>
        }

        aie.use_lock(%objFifo_in1_cons_prod_lock, Release, 1)
        aie.use_lock(%objFifo_out1_cons_lock, Release, 1)

        aie.use_lock(%objFifo_in1_cons_cons_lock, AcquireGreaterEqual, 1)
        aie.use_lock(%objFifo_out1_prod_lock, AcquireGreaterEqual, 1)

        scf.for %arg1 = %c0 to %c8 step %c1 {
          %0 = memref.load %objFifo_in1_cons_buff_1[%arg1] : memref<8xi32>
          %1 = memref.load %constant_buffer[%arg1] : memref<8xi32>
          %2 = arith.addi %0, %1 : i32
          memref.store %2, %objFifo_out1_buff_1[%arg1] : memref<8xi32>
        }

        aie.use_lock(%objFifo_in1_cons_prod_lock, Release, 1)
        aie.use_lock(%objFifo_out1_cons_lock, Release, 1)
      }
      aie.end
    }

    aie.shim_dma_allocation @objFifo_in0(MM2S, 0, 0)
    aie.shim_dma_allocation @objFifo_out0(S2MM, 0, 0)

    memref.global "private" @myData : memref<8xi32> = dense<[1, 2, 3, 4, 5, 6, 7, 8]>
    aiex.runtime_sequence(%arg0: memref<64xi32>, %arg1: memref<32xi32>, %arg2: memref<64xi32>) {
      %c0_i64 = arith.constant 0 : i64
      %c1_i64 = arith.constant 1 : i64
      %c64_i64 = arith.constant 64 : i64
      %0 = memref.get_global @myData : memref<8xi32>
      aiex.npu.blockwrite(%0) {buffer = @constant_buffer, address = 0 : ui32} : memref<8xi32>
      aiex.npu.write32 {buffer = @constant_buffer, address = 4 : ui32, value = 42 : ui32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64] [%c1_i64, %c1_i64, %c1_i64, %c64_i64] [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {id = 0 : i64, issue_token = true, metadata = @objFifo_in0} : memref<64xi32>
      aiex.npu.dma_memcpy_nd(0, 0, %arg2[%c0_i64, %c0_i64, %c0_i64, %c0_i64] [%c1_i64, %c1_i64, %c1_i64, %c64_i64] [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {id = 1 : i64, issue_token = true, metadata = @objFifo_out0} : memref<64xi32>
      aiex.npu.dma_wait {symbol = @objFifo_in0}
      aiex.npu.dma_wait {symbol = @objFifo_out0}
    }

    %mem_0_2 = aie.mem(%tile_0_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%objFifo_in1_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%objFifo_in1_cons_buff_0 : memref<8xi32>, 0, 8)
      aie.use_lock(%objFifo_in1_cons_cons_lock, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%objFifo_in1_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%objFifo_in1_cons_buff_1 : memref<8xi32>, 0, 8)
      aie.use_lock(%objFifo_in1_cons_cons_lock, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%objFifo_out1_cons_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%objFifo_out1_buff_0 : memref<8xi32>, 0, 8)
      aie.use_lock(%objFifo_out1_prod_lock, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%objFifo_out1_cons_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%objFifo_out1_buff_1 : memref<8xi32>, 0, 8)
      aie.use_lock(%objFifo_out1_prod_lock, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      aie.end
    }
  }
}
