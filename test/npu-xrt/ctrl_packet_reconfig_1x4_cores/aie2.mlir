//===- aie2.mlir -----------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2024 Advanced Micro Devices, Inc. or its affiliates
//
//===----------------------------------------------------------------------===//

module {
  aie.device(NPUDEVICE) {
    memref.global "public" @objFifo_in0 : memref<64x64xi8>
    memref.global "public" @objFifo_out0 : memref<64x64xi8>

    %tile_0_0 = aie.tile(0, 0)
    %tile_0_1 = aie.tile(0, 1)
    %tile_0_2 = aie.tile(0, 2)
    %tile_0_3 = aie.tile(0, 3)
    %tile_0_4 = aie.tile(0, 4)
    %tile_0_5 = aie.tile(0, 5)

    // Tile 0, 2 buffers and locks

    %tile_0_2_buff_0 = aie.buffer(%tile_0_2) {sym_name = "tile_0_2_buff_0"} : memref<64x64xi8>
    %tile_0_2_buff_1 = aie.buffer(%tile_0_2) {sym_name = "tile_0_2_buff_1"} : memref<64x64xi8>

    %tile_0_2_lock_0 = aie.lock(%tile_0_2, 0) {init = 1 : i32, sym_name = "tile_0_2_lock_0"}
    %tile_0_2_lock_1 = aie.lock(%tile_0_2, 1) {init = 0 : i32, sym_name = "tile_0_2_lock_1"}
    %tile_0_2_lock_2 = aie.lock(%tile_0_2, 2) {init = 0 : i32, sym_name = "tile_0_2_lock_2"}
    %tile_0_2_lock_3 = aie.lock(%tile_0_2, 3) {init = 1 : i32, sym_name = "tile_0_2_lock_3"}

    // Tile 0, 3 buffers and locks

    %tile_0_3_buff_0 = aie.buffer(%tile_0_3) {sym_name = "tile_0_3_buff_0"} : memref<64x64xi8>
    %tile_0_3_buff_1 = aie.buffer(%tile_0_3) {sym_name = "tile_0_3_buff_1"} : memref<64x64xi8>

    %tile_0_3_lock_0 = aie.lock(%tile_0_3, 0) {init = 1 : i32, sym_name = "tile_0_3_lock_0"}
    %tile_0_3_lock_1 = aie.lock(%tile_0_3, 1) {init = 0 : i32, sym_name = "tile_0_3_lock_1"}
    %tile_0_3_lock_2 = aie.lock(%tile_0_3, 2) {init = 0 : i32, sym_name = "tile_0_3_lock_2"}
    %tile_0_3_lock_3 = aie.lock(%tile_0_3, 3) {init = 1 : i32, sym_name = "tile_0_3_lock_3"}

    // Tile 0, 4 buffers and locks

    %tile_0_4_buff_0 = aie.buffer(%tile_0_4) {sym_name = "tile_0_4_buff_0"} : memref<64x64xi8>
    %tile_0_4_buff_1 = aie.buffer(%tile_0_4) {sym_name = "tile_0_4_buff_1"} : memref<64x64xi8>

    %tile_0_4_lock_0 = aie.lock(%tile_0_4, 0) {init = 1 : i32, sym_name = "tile_0_4_lock_0"}
    %tile_0_4_lock_1 = aie.lock(%tile_0_4, 1) {init = 0 : i32, sym_name = "tile_0_4_lock_1"}
    %tile_0_4_lock_2 = aie.lock(%tile_0_4, 2) {init = 0 : i32, sym_name = "tile_0_4_lock_2"}
    %tile_0_4_lock_3 = aie.lock(%tile_0_4, 3) {init = 1 : i32, sym_name = "tile_0_4_lock_3"}

    // Tile 0, 5 buffers and locks

    %tile_0_5_buff_0 = aie.buffer(%tile_0_5) {sym_name = "tile_0_5_buff_0"} : memref<64x64xi8>
    %tile_0_5_buff_1 = aie.buffer(%tile_0_5) {sym_name = "tile_0_5_buff_1"} : memref<64x64xi8>

    %tile_0_5_lock_0 = aie.lock(%tile_0_5, 0) {init = 1 : i32, sym_name = "tile_0_5_lock_0"}
    %tile_0_5_lock_1 = aie.lock(%tile_0_5, 1) {init = 0 : i32, sym_name = "tile_0_5_lock_1"}
    %tile_0_5_lock_2 = aie.lock(%tile_0_5, 2) {init = 0 : i32, sym_name = "tile_0_5_lock_2"}
    %tile_0_5_lock_3 = aie.lock(%tile_0_5, 3) {init = 1 : i32, sym_name = "tile_0_5_lock_3"}

    aie.flow(%tile_0_1, DMA : 0, %tile_0_2, DMA : 0)
    aie.flow(%tile_0_1, DMA : 1, %tile_0_3, DMA : 0)
    aie.flow(%tile_0_1, DMA : 2, %tile_0_4, DMA : 0)
    aie.flow(%tile_0_1, DMA : 3, %tile_0_5, DMA : 0)

    aie.flow(%tile_0_2, DMA : 0, %tile_0_1, DMA : 1)
    aie.flow(%tile_0_3, DMA : 0, %tile_0_1, DMA : 2)
    aie.flow(%tile_0_4, DMA : 0, %tile_0_1, DMA : 3)
    aie.flow(%tile_0_5, DMA : 0, %tile_0_1, DMA : 4)

    aie.packet_flow(0) {
      aie.packet_source<%tile_0_0, DMA : 0>
      aie.packet_dest<%tile_0_1, DMA : 0>
    }
    aie.flow(%tile_0_1, DMA : 4, %tile_0_0, DMA : 0)

    // Tile 0, 2 core and dma
    %core_0_2 = aie.core(%tile_0_2) {
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c12_i8 = arith.constant 12 : i8
      %c2 = arith.constant 2 : index
      %c64 = arith.constant 64 : index
      aie.use_lock(%tile_0_2_lock_3, AcquireGreaterEqual, 1)
      aie.use_lock(%tile_0_2_lock_1, AcquireGreaterEqual, 1)
      scf.for %arg1 = %c0 to %c64 step %c1 {
        scf.for %arg2 = %c0 to %c64 step %c1 {
          %0 = memref.load %tile_0_2_buff_0[%arg1, %arg2] : memref<64x64xi8>
          %1 = arith.addi %0, %c12_i8 : i8
          memref.store %1, %tile_0_2_buff_1[%arg1, %arg2] : memref<64x64xi8>
        }
      }
      aie.use_lock(%tile_0_2_lock_0, Release, 1)
      aie.use_lock(%tile_0_2_lock_2, Release, 1)
      aie.end
    }

    %mem_0_2 = aie.mem(%tile_0_2) {
      %0 = aie.dma(S2MM, 0) [{
        aie.use_lock(%tile_0_2_lock_0, AcquireGreaterEqual, 1)
        aie.dma_bd(%tile_0_2_buff_0 : memref<64x64xi8>)
        aie.use_lock(%tile_0_2_lock_1, Release, 1)
      }]
      %1 = aie.dma(MM2S, 0) [{
        aie.use_lock(%tile_0_2_lock_2, AcquireGreaterEqual, 1)
        aie.dma_bd(%tile_0_2_buff_1 : memref<64x64xi8>)
        aie.use_lock(%tile_0_2_lock_3, Release, 1)
      }]
      aie.end
    }

    // Tile 0, 3 core and dma
    %core_0_3 = aie.core(%tile_0_3) {
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c13_i8 = arith.constant 13 : i8
      %c2 = arith.constant 2 : index
      %c64 = arith.constant 64 : index
      aie.use_lock(%tile_0_3_lock_3, AcquireGreaterEqual, 1)
      aie.use_lock(%tile_0_3_lock_1, AcquireGreaterEqual, 1)
      scf.for %arg1 = %c0 to %c64 step %c1 {
        scf.for %arg2 = %c0 to %c64 step %c1 {
          %0 = memref.load %tile_0_3_buff_0[%arg1, %arg2] : memref<64x64xi8>
          %1 = arith.addi %0, %c13_i8 : i8
          memref.store %1, %tile_0_3_buff_1[%arg1, %arg2] : memref<64x64xi8>
        }
      }
      aie.use_lock(%tile_0_3_lock_0, Release, 1)
      aie.use_lock(%tile_0_3_lock_2, Release, 1)
      aie.end
    }

    %mem_0_3 = aie.mem(%tile_0_3) {
      %0 = aie.dma(S2MM, 0) [{
        aie.use_lock(%tile_0_3_lock_0, AcquireGreaterEqual, 1)
        aie.dma_bd(%tile_0_3_buff_0 : memref<64x64xi8>)
        aie.use_lock(%tile_0_3_lock_1, Release, 1)
      }]
      %1 = aie.dma(MM2S, 0) [{
        aie.use_lock(%tile_0_3_lock_2, AcquireGreaterEqual, 1)
        aie.dma_bd(%tile_0_3_buff_1 : memref<64x64xi8>)
        aie.use_lock(%tile_0_3_lock_3, Release, 1)
      }]
      aie.end
    }

    // Tile 0, 4 core and dma
    %core_0_4 = aie.core(%tile_0_4) {
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c14_i8 = arith.constant 14 : i8
      %c2 = arith.constant 2 : index
      %c64 = arith.constant 64 : index
      aie.use_lock(%tile_0_4_lock_3, AcquireGreaterEqual, 1)
      aie.use_lock(%tile_0_4_lock_1, AcquireGreaterEqual, 1)
      scf.for %arg1 = %c0 to %c64 step %c1 {
        scf.for %arg2 = %c0 to %c64 step %c1 {
          %0 = memref.load %tile_0_4_buff_0[%arg1, %arg2] : memref<64x64xi8>
          %1 = arith.addi %0, %c14_i8 : i8
          memref.store %1, %tile_0_4_buff_1[%arg1, %arg2] : memref<64x64xi8>
        }
      }
      aie.use_lock(%tile_0_4_lock_0, Release, 1)
      aie.use_lock(%tile_0_4_lock_2, Release, 1)
      aie.end
    }

    %mem_0_4 = aie.mem(%tile_0_4) {
      %0 = aie.dma(S2MM, 0) [{
        aie.use_lock(%tile_0_4_lock_0, AcquireGreaterEqual, 1)
        aie.dma_bd(%tile_0_4_buff_0 : memref<64x64xi8>)
        aie.use_lock(%tile_0_4_lock_1, Release, 1)
      }]
      %1 = aie.dma(MM2S, 0) [{
        aie.use_lock(%tile_0_4_lock_2, AcquireGreaterEqual, 1)
        aie.dma_bd(%tile_0_4_buff_1 : memref<64x64xi8>)
        aie.use_lock(%tile_0_4_lock_3, Release, 1)
      }]
      aie.end
    }

    // Tile 0, 5 core and dma
    %core_0_5 = aie.core(%tile_0_5) {
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c15_i8 = arith.constant 15 : i8
      %c2 = arith.constant 2 : index
      %c64 = arith.constant 64 : index
      aie.use_lock(%tile_0_5_lock_3, AcquireGreaterEqual, 1)
      aie.use_lock(%tile_0_5_lock_1, AcquireGreaterEqual, 1)
      scf.for %arg1 = %c0 to %c64 step %c1 {
        scf.for %arg2 = %c0 to %c64 step %c1 {
          %0 = memref.load %tile_0_5_buff_0[%arg1, %arg2] : memref<64x64xi8>
          %1 = arith.addi %0, %c15_i8 : i8
          memref.store %1, %tile_0_5_buff_1[%arg1, %arg2] : memref<64x64xi8>
        }
      }
      aie.use_lock(%tile_0_5_lock_0, Release, 1)
      aie.use_lock(%tile_0_5_lock_2, Release, 1)
      aie.end
    }

    %mem_0_5 = aie.mem(%tile_0_5) {
      %0 = aie.dma(S2MM, 0) [{
        aie.use_lock(%tile_0_5_lock_0, AcquireGreaterEqual, 1)
        aie.dma_bd(%tile_0_5_buff_0 : memref<64x64xi8>)
        aie.use_lock(%tile_0_5_lock_1, Release, 1)
      }]
      %1 = aie.dma(MM2S, 0) [{
        aie.use_lock(%tile_0_5_lock_2, AcquireGreaterEqual, 1)
        aie.dma_bd(%tile_0_5_buff_1 : memref<64x64xi8>)
        aie.use_lock(%tile_0_5_lock_3, Release, 1)
      }]
      aie.end
    }

    %memtile_dma_0_1 = aie.memtile_dma(%tile_0_1) {
      %buff_0 = aie.buffer(%tile_0_1) {sym_name = "memtile_buff_0"} : memref<4x64x64xi8>
      %buff_1 = aie.buffer(%tile_0_1) {sym_name = "memtile_buff_1"} : memref<4x64x64xi8>
      %memtile_lock_0 = aie.lock(%tile_0_1, 0) {init = 4 : i32, sym_name = "memtile_lock_0"}
      %memtile_lock_1 = aie.lock(%tile_0_1, 1) {init = 0 : i32, sym_name = "memtile_lock_1"}
      %memtile_lock_2 = aie.lock(%tile_0_1, 2) {init = 0 : i32, sym_name = "memtile_lock_2"}
      %memtile_lock_3 = aie.lock(%tile_0_1, 3) {init = 4 : i32, sym_name = "memtile_lock_3"}
      %0 = aie.dma(S2MM, 0) [{
        aie.use_lock(%memtile_lock_0, AcquireGreaterEqual, 4)
        aie.dma_bd(%buff_0 : memref<4x64x64xi8>, 0, 16384)
        aie.use_lock(%memtile_lock_1, Release, 4)
      }]
      %1 = aie.dma(MM2S, 0) [{
        aie.use_lock(%memtile_lock_1, AcquireGreaterEqual, 1)
        aie.dma_bd(%buff_0 : memref<4x64x64xi8>, 0, 4096)
        aie.use_lock(%memtile_lock_0, Release, 1)
      }]
      %2 = aie.dma(MM2S, 1) [{
        aie.use_lock(%memtile_lock_1, AcquireGreaterEqual, 1)
        aie.dma_bd(%buff_0 : memref<4x64x64xi8>, 4096, 4096)
        aie.use_lock(%memtile_lock_0, Release, 1)
      }]
      %3 = aie.dma(MM2S, 2) [{
        aie.use_lock(%memtile_lock_1, AcquireGreaterEqual, 1)
        aie.dma_bd(%buff_0 : memref<4x64x64xi8>, 8192, 4096)
        aie.use_lock(%memtile_lock_0, Release, 1)
      }]
      %4 = aie.dma(MM2S, 3) [{
        aie.use_lock(%memtile_lock_1, AcquireGreaterEqual, 1)
        aie.dma_bd(%buff_0 : memref<4x64x64xi8>, 12288, 4096)
        aie.use_lock(%memtile_lock_0, Release, 1)
      }]

      %5 = aie.dma(S2MM, 1) [{
        aie.use_lock(%memtile_lock_3, AcquireGreaterEqual, 1)
        aie.dma_bd(%buff_1 : memref<4x64x64xi8>, 0, 4096)
        aie.use_lock(%memtile_lock_2, Release, 1)
      }]
      %6 = aie.dma(S2MM, 2) [{
        aie.use_lock(%memtile_lock_3, AcquireGreaterEqual, 1)
        aie.dma_bd(%buff_1 : memref<4x64x64xi8>, 4096, 4096)
        aie.use_lock(%memtile_lock_2, Release, 1)
      }]
      %7 = aie.dma(S2MM, 3) [{
        aie.use_lock(%memtile_lock_3, AcquireGreaterEqual, 1)
        aie.dma_bd(%buff_1 : memref<4x64x64xi8>, 8192, 4096)
        aie.use_lock(%memtile_lock_2, Release, 1)
      }]
      %8 = aie.dma(S2MM, 4) [{
        aie.use_lock(%memtile_lock_3, AcquireGreaterEqual, 1)
        aie.dma_bd(%buff_1 : memref<4x64x64xi8>, 12288, 4096)
        aie.use_lock(%memtile_lock_2, Release, 1)
      }]
      %9 = aie.dma(MM2S, 4) [{
        aie.use_lock(%memtile_lock_2, AcquireGreaterEqual, 4)
        aie.dma_bd(%buff_1 : memref<4x64x64xi8>, 0, 16384)
        aie.use_lock(%memtile_lock_3, Release, 4)
      }]
      aie.end
    }

    aie.shim_dma_allocation @objFifo_in0(MM2S, 0, 0)
    aie.shim_dma_allocation @objFifo_out0(S2MM, 0, 0)

    aiex.runtime_sequence @run(%arg0: memref<4x64x64xi8>, %arg1: memref<4x64x64xi8>) {
      %c0_i64 = arith.constant 0 : i64
      %c1_i64 = arith.constant 1 : i64
      %c4_i64 = arith.constant 4 : i64
      %c4096_i64 = arith.constant 4096 : i64
      %c64_i64 = arith.constant 64 : i64
      aiex.npu.configure
      aiex.npu.dma_memcpy_nd (%arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64][%c1_i64, %c4_i64, %c64_i64, %c64_i64][%c0_i64, %c4096_i64, %c64_i64, %c1_i64], packet = <pkt_id = 0, pkt_type = 0>) {id = 0 : i64, metadata = @objFifo_in0} : memref<4x64x64xi8>
      aiex.npu.dma_memcpy_nd (%arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64][%c1_i64, %c4_i64, %c64_i64, %c64_i64][%c0_i64, %c4096_i64, %c64_i64, %c1_i64]) {id = 1 : i64, metadata = @objFifo_out0, issue_token = true} : memref<4x64x64xi8>
      aiex.npu.dma_wait { symbol = @objFifo_out0 }
    }
  }
}
