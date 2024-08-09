//===- dma_op.mlir --------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2023 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: not aie-opt -split-input-file %s 2>&1 | FileCheck %s

// CHECK: error: 'aie.dma' op DMAOp can only appear in single block region
module {
  aie.device(npu1_4col) {
    %tile_0_1 = aie.tile(0, 1)
    %objFifo_in0_cons_buff_0 = aie.buffer(%tile_0_1) {address = 0 : i32} : memref<16xi32>
    %objFifo_in0_cons_prod_lock = aie.lock(%tile_0_1, 0) {init = 2 : i32}
    %objFifo_in0_cons_cons_lock = aie.lock(%tile_0_1, 1) {init = 0 : i32}

    %memtile_dma_0_1 = aie.memtile_dma(%tile_0_1) {
    ^bb0:
      aie.dma(S2MM, 0) [{
        aie.use_lock(%objFifo_in0_cons_prod_lock, AcquireGreaterEqual, 1)
        aie.dma_bd(%objFifo_in0_cons_buff_0 : memref<16xi32>, 0, 16)
        aie.use_lock(%objFifo_in0_cons_cons_lock, Release, 1)
      }]
      aie.next_bd ^bb1
    ^bb1:
      aie.end
    }

  }
}

// -----

// CHECK: error: 'aie.dma_bd' op Packet ID field can only hold 5 bits.
module {
  aie.device(npu1_4col) {
    %tile14 = aie.tile(1, 4)
    %buf14 = aie.buffer(%tile14) { sym_name = "buf14" } : memref<128xi32>
    %mem14 = aie.mem(%tile14) {
      %srcDma = aie.dma_start("MM2S", 0, ^bd0, ^end)
      ^bd0:
        aie.dma_bd(%buf14 : memref<128xi32>, 0, 128, [<size = 1, stride = 128>]) {packet = #aie.packet_info<pkt_type = 7, pkt_id = 33>}
        aie.next_bd ^end
      ^end: 
        aie.end
    }
  }
}
