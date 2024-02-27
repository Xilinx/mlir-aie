//===- dma_op.mlir --------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2023 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-translate --aie-generate-cdo %s --cdo-axi-debug 2>&1 | FileCheck %s

// CHECK: Generating: {{.*}}aie_cdo_error_handling.bin
// CHECK: Generating: {{.*}}aie_cdo_init.bin

module {
  aie.device(ipu) {
    %tile_0_1 = aie.tile(0, 1)
    %tile_0_2 = aie.tile(0, 2)

    %objFifo_in0_cons_buff_0 = aie.buffer(%tile_0_1) {address = 0 : i32} : memref<16xi32>
    %objFifo_in0_cons_buff_1 = aie.buffer(%tile_0_1) {address = 64 : i32} : memref<16xi32>
    %objFifo_out0_buff_0 = aie.buffer(%tile_0_1) {address = 128 : i32} : memref<16xi32>
    %objFifo_out0_buff_1 = aie.buffer(%tile_0_1) {address = 192 : i32} : memref<16xi32>

    %objFifo_in1_cons_buff_0 = aie.buffer(%tile_0_2) {address = 0 : i32} : memref<8xi32>
    %objFifo_in1_cons_buff_1 = aie.buffer(%tile_0_2) {address = 32 : i32} : memref<8xi32>
    %objFifo_out1_buff_0 = aie.buffer(%tile_0_2) {address = 64 : i32} : memref<8xi32>
    %objFifo_out1_buff_1 = aie.buffer(%tile_0_2) {address = 96 : i32} : memref<8xi32>

    %objFifo_in0_cons_prod_lock = aie.lock(%tile_0_1, 0) {init = 2 : i32}
    %objFifo_in0_cons_cons_lock = aie.lock(%tile_0_1, 1) {init = 0 : i32}
    %objFifo_out0_prod_lock = aie.lock(%tile_0_1, 2) {init = 2 : i32}
    %objFifo_out0_cons_lock = aie.lock(%tile_0_1, 3) {init = 0 : i32}

    %objFifo_in1_cons_prod_lock = aie.lock(%tile_0_2, 0) {init = 2 : i32}
    %objFifo_in1_cons_cons_lock = aie.lock(%tile_0_2, 1) {init = 0 : i32}
    %objFifo_out1_prod_lock = aie.lock(%tile_0_2, 2) {init = 2 : i32}
    %objFifo_out1_cons_lock = aie.lock(%tile_0_2, 3) {init = 0 : i32}

    %memtile_dma_0_1 = aie.memtile_dma(%tile_0_1) {
      aie.dma(S2MM, 0) [{
        aie.use_lock(%objFifo_in0_cons_prod_lock, AcquireGreaterEqual, 1)
        aie.dma_bd(%objFifo_in0_cons_buff_0 : memref<16xi32>, 0, 16)
        aie.use_lock(%objFifo_in0_cons_cons_lock, Release, 1)
      }, {
        aie.use_lock(%objFifo_in0_cons_prod_lock, AcquireGreaterEqual, 1)
        aie.dma_bd(%objFifo_in0_cons_buff_1 : memref<16xi32>, 0, 16)
        aie.use_lock(%objFifo_in0_cons_cons_lock, Release, 1)
      }]
      aie.dma(MM2S, 0) [{
        aie.use_lock(%objFifo_in0_cons_cons_lock, AcquireGreaterEqual, 1)
        aie.dma_bd(%objFifo_in0_cons_buff_0 : memref<16xi32>, 0, 16)
        aie.use_lock(%objFifo_in0_cons_prod_lock, Release, 1)
      }, {
        aie.use_lock(%objFifo_in0_cons_cons_lock, AcquireGreaterEqual, 1)
        aie.dma_bd(%objFifo_in0_cons_buff_1 : memref<16xi32>, 0, 16)
        aie.use_lock(%objFifo_in0_cons_prod_lock, Release, 1)
      }]
      aie.dma(MM2S, 0) [{
        aie.use_lock(%objFifo_out0_cons_lock, AcquireGreaterEqual, 1)
        aie.dma_bd(%objFifo_out0_buff_0 : memref<16xi32>, 0, 16)
        aie.use_lock(%objFifo_out0_prod_lock, Release, 1)
      }, {
        aie.use_lock(%objFifo_out0_cons_lock, AcquireGreaterEqual, 1)
        aie.dma_bd(%objFifo_out0_buff_1 : memref<16xi32>, 0, 16)
        aie.use_lock(%objFifo_out0_prod_lock, Release, 1)
      }]
      aie.dma(S2MM, 1) [{
        aie.use_lock(%objFifo_out0_prod_lock, AcquireGreaterEqual, 1)
        aie.dma_bd(%objFifo_out0_buff_0 : memref<16xi32>, 0, 16)
        aie.use_lock(%objFifo_out0_cons_lock, Release, 1)
      }, {
        aie.use_lock(%objFifo_out0_prod_lock, AcquireGreaterEqual, 1)
        aie.dma_bd(%objFifo_out0_buff_1 : memref<16xi32>, 0, 16)
        aie.use_lock(%objFifo_out0_cons_lock, Release, 1)
      }]
      aie.end
    }

    %mem_0_2 = aie.mem(%tile_0_2) {
      aie.dma(S2MM, 0) [{
        aie.use_lock(%objFifo_in1_cons_prod_lock, AcquireGreaterEqual, 1)
        aie.dma_bd(%objFifo_in1_cons_buff_0 : memref<8xi32>, 0, 8)
        aie.use_lock(%objFifo_in1_cons_cons_lock, Release, 1)
      }, {
        aie.use_lock(%objFifo_in1_cons_prod_lock, AcquireGreaterEqual, 1)
        aie.dma_bd(%objFifo_in1_cons_buff_1 : memref<8xi32>, 0, 8)
        aie.use_lock(%objFifo_in1_cons_cons_lock, Release, 1)
      }]
      aie.dma(MM2S, 0) {loop = false} [{
        aie.use_lock(%objFifo_out1_cons_lock, AcquireGreaterEqual, 1)
        aie.dma_bd(%objFifo_out1_buff_0 : memref<8xi32>, 0, 8)
        aie.use_lock(%objFifo_out1_prod_lock, Release, 1)
      }, {
        aie.use_lock(%objFifo_out1_cons_lock, AcquireGreaterEqual, 1)
        aie.dma_bd(%objFifo_out1_buff_1 : memref<8xi32>, 0, 8)
        aie.use_lock(%objFifo_out1_prod_lock, Release, 1)
      }]
      aie.end
    }
  }
}
