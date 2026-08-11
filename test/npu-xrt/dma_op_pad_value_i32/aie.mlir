//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2024 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Minimal MemTile passthrough exercising the aie.dma region form with a
// per-channel pad_value: 13 i32 arrive on S2MM, and the padded MM2S sends 16
// (2 before, 1 after) with the pad region filled by pad_value = 42.

module {
  aie.device(NPUDEVICE) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_1 = aie.tile(0, 1)

    aie.flow(%tile_0_0, DMA : 0, %tile_0_1, DMA : 0)
    aie.flow(%tile_0_1, DMA : 0, %tile_0_0, DMA : 0)

    aie.shim_dma_allocation @in0(%tile_0_0, MM2S, 0)
    aie.shim_dma_allocation @out0(%tile_0_0, S2MM, 0)

    aie.runtime_sequence(%arg0: memref<13xi32>, %arg1: memref<16xi32>) {
      %c0 = arith.constant 0 : i64
      %c1 = arith.constant 1 : i64
      %c13 = arith.constant 13 : i64
      %c16 = arith.constant 16 : i64
      aiex.npu.dma_memcpy_nd(%arg0[%c0, %c0, %c0, %c0][%c1, %c1, %c1, %c13][%c0, %c0, %c0, %c1]) {id = 0 : i64, metadata = @in0} : memref<13xi32>
      aiex.npu.dma_memcpy_nd(%arg1[%c0, %c0, %c0, %c0][%c1, %c1, %c1, %c16][%c0, %c0, %c0, %c1]) {id = 1 : i64, metadata = @out0, issue_token = true} : memref<16xi32>
      aiex.npu.dma_wait {symbol = @out0}
    }

    %memtile_dma_0_1 = aie.memtile_dma(%tile_0_1) {
      %buff = aie.buffer(%tile_0_1) {sym_name = "buff"} : memref<16xi32>
      %prod_lock = aie.lock(%tile_0_1, 0) {init = 1 : i32, sym_name = "prod_lock"}
      %cons_lock = aie.lock(%tile_0_1, 1) {init = 0 : i32, sym_name = "cons_lock"}
      %0 = aie.dma(S2MM, 0) [{
        %c1_i32 = arith.constant 1 : i32
        aie.use_lock(%prod_lock, AcquireGreaterEqual, %c1_i32)
        aie.dma_bd(%buff : memref<16xi32> offset = 0 len = 13)
        %c1_i32_0 = arith.constant 1 : i32
        aie.use_lock(%cons_lock, Release, %c1_i32_0)
      }]
      %1 = aie.dma(MM2S, 0) {pad_value = 42 : i32} [{
        %c1_i32 = arith.constant 1 : i32
        aie.use_lock(%cons_lock, AcquireGreaterEqual, %c1_i32)
        aie.dma_bd(%buff : memref<16xi32> offset = 0 len = 16 sizes = [13] strides = [1] pad [<const_pad_before = 2, const_pad_after = 1>])
        %c1_i32_0 = arith.constant 1 : i32
        aie.use_lock(%prod_lock, Release, %c1_i32_0)
      }]
      aie.end
    }
  }
}
