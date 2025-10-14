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
    aie.device(NPUDEVICE) @base {
    %tile_0_0 = aie.tile(0, 0)
    %tile_1_0 = aie.tile(1, 0)
    %tile_2_0 = aie.tile(2, 0)
    %tile_3_0 = aie.tile(3, 0)
    %tile_0_1 = aie.tile(0, 1)
    %tile_1_1 = aie.tile(1, 1)
    %tile_2_1 = aie.tile(2, 1)
    %tile_3_1 = aie.tile(3, 1)
    %tile_0_2 = aie.tile(0, 2)
    %tile_1_2 = aie.tile(1, 2)
    %tile_2_2 = aie.tile(2, 2)
    %tile_3_2 = aie.tile(3, 2)
  }
  aie.device(NPUDEVICE) @main {
    memref.global "public" @shim_in_0 : memref<64x64xi8>
    memref.global "public" @shim_in_1 : memref<64x64xi8>
    memref.global "public" @shim_in_2 : memref<64x64xi8>
    memref.global "public" @shim_in_3 : memref<64x64xi8>
    memref.global "public" @shim_out_0 : memref<64x64xi8>
    memref.global "public" @shim_out_1 : memref<64x64xi8>
    memref.global "public" @shim_out_2 : memref<64x64xi8>
    memref.global "public" @shim_out_3 : memref<64x64xi8>

    %tile_0_0 = aie.tile(0, 0)
    %tile_1_0 = aie.tile(1, 0)
    %tile_2_0 = aie.tile(2, 0)
    %tile_3_0 = aie.tile(3, 0)
    %tile_0_1 = aie.tile(0, 1)
    %tile_1_1 = aie.tile(1, 1)
    %tile_2_1 = aie.tile(2, 1)
    %tile_3_1 = aie.tile(3, 1)
    %tile_0_2 = aie.tile(0, 2)
    %tile_1_2 = aie.tile(1, 2)
    %tile_2_2 = aie.tile(2, 2)
    %tile_3_2 = aie.tile(3, 2)

    // Tile 0, 2 buffers and locks

    %tile_0_2_buff_0 = aie.buffer(%tile_0_2) {sym_name = "tile_0_2_buff_0"} : memref<64x64xi8>
    %tile_0_2_buff_1 = aie.buffer(%tile_0_2) {sym_name = "tile_0_2_buff_1"} : memref<64x64xi8>

    %tile_0_2_lock_0 = aie.lock(%tile_0_2, 0) {init = 1 : i32, sym_name = "tile_0_2_lock_0"}
    %tile_0_2_lock_1 = aie.lock(%tile_0_2, 1) {init = 0 : i32, sym_name = "tile_0_2_lock_1"}
    %tile_0_2_lock_2 = aie.lock(%tile_0_2, 2) {init = 0 : i32, sym_name = "tile_0_2_lock_2"}
    %tile_0_2_lock_3 = aie.lock(%tile_0_2, 3) {init = 1 : i32, sym_name = "tile_0_2_lock_3"}

    // Tile 1, 2 buffers and locks

    %tile_1_2_buff_0 = aie.buffer(%tile_1_2) {sym_name = "tile_1_2_buff_0"} : memref<64x64xi8>
    %tile_1_2_buff_1 = aie.buffer(%tile_1_2) {sym_name = "tile_1_2_buff_1"} : memref<64x64xi8>

    %tile_1_2_lock_0 = aie.lock(%tile_1_2, 0) {init = 1 : i32, sym_name = "tile_1_2_lock_0"}
    %tile_1_2_lock_1 = aie.lock(%tile_1_2, 1) {init = 0 : i32, sym_name = "tile_1_2_lock_1"}
    %tile_1_2_lock_2 = aie.lock(%tile_1_2, 2) {init = 0 : i32, sym_name = "tile_1_2_lock_2"}
    %tile_1_2_lock_3 = aie.lock(%tile_1_2, 3) {init = 1 : i32, sym_name = "tile_1_2_lock_3"}

    // Tile 2, 2 buffers and locks

    %tile_2_2_buff_0 = aie.buffer(%tile_2_2) {sym_name = "tile_2_2_buff_0"} : memref<64x64xi8>
    %tile_2_2_buff_1 = aie.buffer(%tile_2_2) {sym_name = "tile_2_2_buff_1"} : memref<64x64xi8>

    %tile_2_2_lock_0 = aie.lock(%tile_2_2, 0) {init = 1 : i32, sym_name = "tile_2_2_lock_0"}
    %tile_2_2_lock_1 = aie.lock(%tile_2_2, 1) {init = 0 : i32, sym_name = "tile_2_2_lock_1"}
    %tile_2_2_lock_2 = aie.lock(%tile_2_2, 2) {init = 0 : i32, sym_name = "tile_2_2_lock_2"}
    %tile_2_2_lock_3 = aie.lock(%tile_2_2, 3) {init = 1 : i32, sym_name = "tile_2_2_lock_3"}

    // Tile 3, 2 buffers and locks

    %tile_3_2_buff_0 = aie.buffer(%tile_3_2) {sym_name = "tile_3_2_buff_0"} : memref<64x64xi8>
    %tile_3_2_buff_1 = aie.buffer(%tile_3_2) {sym_name = "tile_3_2_buff_1"} : memref<64x64xi8>

    %tile_3_2_lock_0 = aie.lock(%tile_3_2, 0) {init = 1 : i32, sym_name = "tile_3_2_lock_0"}
    %tile_3_2_lock_1 = aie.lock(%tile_3_2, 1) {init = 0 : i32, sym_name = "tile_3_2_lock_1"}
    %tile_3_2_lock_2 = aie.lock(%tile_3_2, 2) {init = 0 : i32, sym_name = "tile_3_2_lock_2"}
    %tile_3_2_lock_3 = aie.lock(%tile_3_2, 3) {init = 1 : i32, sym_name = "tile_3_2_lock_3"}

    aie.flow(%tile_0_2, DMA : 0, %tile_0_0, DMA : 0)
    aie.flow(%tile_1_2, DMA : 0, %tile_1_0, DMA : 0)
    aie.flow(%tile_2_2, DMA : 0, %tile_2_0, DMA : 0)
    aie.flow(%tile_3_2, DMA : 0, %tile_3_0, DMA : 0)

    aie.packet_flow(1) {
      aie.packet_source<%tile_0_0, DMA : 0>
      aie.packet_dest<%tile_0_2, DMA : 0>
    }
    aie.packet_flow(1) {
      aie.packet_source<%tile_1_0, DMA : 0>
      aie.packet_dest<%tile_1_2, DMA : 0>
    }
    aie.packet_flow(1) {
      aie.packet_source<%tile_2_0, DMA : 0>
      aie.packet_dest<%tile_2_2, DMA : 0>
    }
    aie.packet_flow(1) {
      aie.packet_source<%tile_3_0, DMA : 0>
      aie.packet_dest<%tile_3_2, DMA : 0>
    }

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

    // Tile 1, 2 core and dma
    %core_1_2 = aie.core(%tile_1_2) {
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c13_i8 = arith.constant 13 : i8
      %c2 = arith.constant 2 : index
      %c64 = arith.constant 64 : index
      aie.use_lock(%tile_1_2_lock_3, AcquireGreaterEqual, 1)
      aie.use_lock(%tile_1_2_lock_1, AcquireGreaterEqual, 1)
      scf.for %arg1 = %c0 to %c64 step %c1 {
        scf.for %arg2 = %c0 to %c64 step %c1 {
          %0 = memref.load %tile_1_2_buff_0[%arg1, %arg2] : memref<64x64xi8>
          %1 = arith.addi %0, %c13_i8 : i8
          memref.store %1, %tile_1_2_buff_1[%arg1, %arg2] : memref<64x64xi8>
        }
      }
      aie.use_lock(%tile_1_2_lock_0, Release, 1)
      aie.use_lock(%tile_1_2_lock_2, Release, 1)
      aie.end
    }

    %mem_1_2 = aie.mem(%tile_1_2) {
      %0 = aie.dma(S2MM, 0) [{
        aie.use_lock(%tile_1_2_lock_0, AcquireGreaterEqual, 1)
        aie.dma_bd(%tile_1_2_buff_0 : memref<64x64xi8>)
        aie.use_lock(%tile_1_2_lock_1, Release, 1)
      }]
      %1 = aie.dma(MM2S, 0) [{
        aie.use_lock(%tile_1_2_lock_2, AcquireGreaterEqual, 1)
        aie.dma_bd(%tile_1_2_buff_1 : memref<64x64xi8>)
        aie.use_lock(%tile_1_2_lock_3, Release, 1)
      }]
      aie.end
    }

    // Tile 2, 2 core and dma
    %core_2_2 = aie.core(%tile_2_2) {
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c14_i8 = arith.constant 14 : i8
      %c2 = arith.constant 2 : index
      %c64 = arith.constant 64 : index
      aie.use_lock(%tile_2_2_lock_3, AcquireGreaterEqual, 1)
      aie.use_lock(%tile_2_2_lock_1, AcquireGreaterEqual, 1)
      scf.for %arg1 = %c0 to %c64 step %c1 {
        scf.for %arg2 = %c0 to %c64 step %c1 {
          %0 = memref.load %tile_2_2_buff_0[%arg1, %arg2] : memref<64x64xi8>
          %1 = arith.addi %0, %c14_i8 : i8
          memref.store %1, %tile_2_2_buff_1[%arg1, %arg2] : memref<64x64xi8>
        }
      }
      aie.use_lock(%tile_2_2_lock_0, Release, 1)
      aie.use_lock(%tile_2_2_lock_2, Release, 1)
      aie.end
    }

    %mem_2_2 = aie.mem(%tile_2_2) {
      %0 = aie.dma(S2MM, 0) [{
        aie.use_lock(%tile_2_2_lock_0, AcquireGreaterEqual, 1)
        aie.dma_bd(%tile_2_2_buff_0 : memref<64x64xi8>)
        aie.use_lock(%tile_2_2_lock_1, Release, 1)
      }]
      %1 = aie.dma(MM2S, 0) [{
        aie.use_lock(%tile_2_2_lock_2, AcquireGreaterEqual, 1)
        aie.dma_bd(%tile_2_2_buff_1 : memref<64x64xi8>)
        aie.use_lock(%tile_2_2_lock_3, Release, 1)
      }]
      aie.end
    }

    // Tile 3, 2 core and dma
    %core_3_2 = aie.core(%tile_3_2) {
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c15_i8 = arith.constant 15 : i8
      %c2 = arith.constant 2 : index
      %c64 = arith.constant 64 : index
      aie.use_lock(%tile_3_2_lock_3, AcquireGreaterEqual, 1)
      aie.use_lock(%tile_3_2_lock_1, AcquireGreaterEqual, 1)
      scf.for %arg1 = %c0 to %c64 step %c1 {
        scf.for %arg2 = %c0 to %c64 step %c1 {
          %0 = memref.load %tile_3_2_buff_0[%arg1, %arg2] : memref<64x64xi8>
          %1 = arith.addi %0, %c15_i8 : i8
          memref.store %1, %tile_3_2_buff_1[%arg1, %arg2] : memref<64x64xi8>
        }
      }
      aie.use_lock(%tile_3_2_lock_0, Release, 1)
      aie.use_lock(%tile_3_2_lock_2, Release, 1)
      aie.end
    }

    %mem_3_2 = aie.mem(%tile_3_2) {
      %0 = aie.dma(S2MM, 0) [{
        aie.use_lock(%tile_3_2_lock_0, AcquireGreaterEqual, 1)
        aie.dma_bd(%tile_3_2_buff_0 : memref<64x64xi8>)
        aie.use_lock(%tile_3_2_lock_1, Release, 1)
      }]
      %1 = aie.dma(MM2S, 0) [{
        aie.use_lock(%tile_3_2_lock_2, AcquireGreaterEqual, 1)
        aie.dma_bd(%tile_3_2_buff_1 : memref<64x64xi8>)
        aie.use_lock(%tile_3_2_lock_3, Release, 1)
      }]
      aie.end
    }

    aie.shim_dma_allocation @shim_in_0(MM2S, 0, 0)
    aie.shim_dma_allocation @shim_in_1(MM2S, 0, 1)
    aie.shim_dma_allocation @shim_in_2(MM2S, 0, 2)
    aie.shim_dma_allocation @shim_in_3(MM2S, 0, 3)
    aie.shim_dma_allocation @shim_out_0(S2MM, 0, 0)
    aie.shim_dma_allocation @shim_out_1(S2MM, 0, 1)
    aie.shim_dma_allocation @shim_out_2(S2MM, 0, 2)
    aie.shim_dma_allocation @shim_out_3(S2MM, 0, 3)

    aiex.runtime_sequence @run(%arg0: memref<4x64x64xi8>, %arg1: memref<4x64x64xi8>) {
      %c0_i64 = arith.constant 0 : i64
      %c1_i64 = arith.constant 1 : i64
      %c2_i64 = arith.constant 2 : i64
      %c3_i64 = arith.constant 3 : i64
      %c4_i64 = arith.constant 4 : i64
      %c4096_i64 = arith.constant 4096 : i64
      %c64_i64 = arith.constant 64 : i64
      aiex.configure @main {
        aiex.npu.dma_memcpy_nd (%arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64][%c1_i64, %c1_i64, %c64_i64, %c64_i64][%c0_i64, %c4096_i64, %c64_i64, %c1_i64], packet = <pkt_id = 1, pkt_type = 0>) {id = 0 : i64, metadata = @shim_in_0} : memref<4x64x64xi8>
        aiex.npu.dma_memcpy_nd (%arg0[%c0_i64, %c1_i64, %c0_i64, %c0_i64][%c1_i64, %c1_i64, %c64_i64, %c64_i64][%c0_i64, %c4096_i64, %c64_i64, %c1_i64], packet = <pkt_id = 1, pkt_type = 0>) {id = 1 : i64, metadata = @shim_in_1} : memref<4x64x64xi8>
        aiex.npu.dma_memcpy_nd (%arg0[%c0_i64, %c2_i64, %c0_i64, %c0_i64][%c1_i64, %c1_i64, %c64_i64, %c64_i64][%c0_i64, %c4096_i64, %c64_i64, %c1_i64], packet = <pkt_id = 1, pkt_type = 0>) {id = 2 : i64, metadata = @shim_in_2} : memref<4x64x64xi8>
        aiex.npu.dma_memcpy_nd (%arg0[%c0_i64, %c3_i64, %c0_i64, %c0_i64][%c1_i64, %c1_i64, %c64_i64, %c64_i64][%c0_i64, %c4096_i64, %c64_i64, %c1_i64], packet = <pkt_id = 1, pkt_type = 0>) {id = 3 : i64, metadata = @shim_in_3} : memref<4x64x64xi8>

        aiex.npu.dma_memcpy_nd (%arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64][%c1_i64, %c1_i64, %c64_i64, %c64_i64][%c0_i64, %c4096_i64, %c64_i64, %c1_i64]) {id = 4 : i64, metadata = @shim_out_0, issue_token = true} : memref<4x64x64xi8>
        aiex.npu.dma_memcpy_nd (%arg1[%c0_i64, %c1_i64, %c0_i64, %c0_i64][%c1_i64, %c1_i64, %c64_i64, %c64_i64][%c0_i64, %c4096_i64, %c64_i64, %c1_i64]) {id = 5 : i64, metadata = @shim_out_1, issue_token = true} : memref<4x64x64xi8>
        aiex.npu.dma_memcpy_nd (%arg1[%c0_i64, %c2_i64, %c0_i64, %c0_i64][%c1_i64, %c1_i64, %c64_i64, %c64_i64][%c0_i64, %c4096_i64, %c64_i64, %c1_i64]) {id = 6 : i64, metadata = @shim_out_2, issue_token = true} : memref<4x64x64xi8>
        aiex.npu.dma_memcpy_nd (%arg1[%c0_i64, %c3_i64, %c0_i64, %c0_i64][%c1_i64, %c1_i64, %c64_i64, %c64_i64][%c0_i64, %c4096_i64, %c64_i64, %c1_i64]) {id = 7 : i64, metadata = @shim_out_3, issue_token = true} : memref<4x64x64xi8>
        aiex.npu.dma_wait { symbol = @shim_out_0 }
        aiex.npu.dma_wait { symbol = @shim_out_1 }
        aiex.npu.dma_wait { symbol = @shim_out_2 }
        aiex.npu.dma_wait { symbol = @shim_out_3 }
      }
    }
  }
}
