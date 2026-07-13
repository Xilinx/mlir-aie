//===- aie2.mlir -----------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2024 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

module {
  aie.device(NPUDEVICE) @base {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_1 = aie.tile(0, 1)
    %tile_0_2 = aie.tile(0, 2)
    %tile_0_3 = aie.tile(0, 3)
    %tile_0_4 = aie.tile(0, 4)
    %tile_0_5 = aie.tile(0, 5)
  }
  aie.device(NPUDEVICE) @main {
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
      %c1_ul0 = arith.constant 1 : i32
      aie.use_lock(%tile_0_2_lock_3, AcquireGreaterEqual, %c1_ul0)
      %c1_ul1 = arith.constant 1 : i32
      aie.use_lock(%tile_0_2_lock_1, AcquireGreaterEqual, %c1_ul1)
      scf.for %arg1 = %c0 to %c64 step %c1 {
        scf.for %arg2 = %c0 to %c64 step %c1 {
          %0 = memref.load %tile_0_2_buff_0[%arg1, %arg2] : memref<64x64xi8>
          %1 = arith.addi %0, %c12_i8 : i8
          memref.store %1, %tile_0_2_buff_1[%arg1, %arg2] : memref<64x64xi8>
        }
      }
      %c1_ul2 = arith.constant 1 : i32
      aie.use_lock(%tile_0_2_lock_0, Release, %c1_ul2)
      %c1_ul3 = arith.constant 1 : i32
      aie.use_lock(%tile_0_2_lock_2, Release, %c1_ul3)
      aie.end
    }

    %mem_0_2 = aie.mem(%tile_0_2) {
      %0 = aie.dma(S2MM, 0) [{
        %c1_ul4 = arith.constant 1 : i32
        aie.use_lock(%tile_0_2_lock_0, AcquireGreaterEqual, %c1_ul4)
        aie.dma_bd(%tile_0_2_buff_0 : memref<64x64xi8>)
        %c1_ul5 = arith.constant 1 : i32
        aie.use_lock(%tile_0_2_lock_1, Release, %c1_ul5)
      }]
      %1 = aie.dma(MM2S, 0) [{
        %c1_ul6 = arith.constant 1 : i32
        aie.use_lock(%tile_0_2_lock_2, AcquireGreaterEqual, %c1_ul6)
        aie.dma_bd(%tile_0_2_buff_1 : memref<64x64xi8>)
        %c1_ul7 = arith.constant 1 : i32
        aie.use_lock(%tile_0_2_lock_3, Release, %c1_ul7)
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
      %c1_ul8 = arith.constant 1 : i32
      aie.use_lock(%tile_0_3_lock_3, AcquireGreaterEqual, %c1_ul8)
      %c1_ul9 = arith.constant 1 : i32
      aie.use_lock(%tile_0_3_lock_1, AcquireGreaterEqual, %c1_ul9)
      scf.for %arg1 = %c0 to %c64 step %c1 {
        scf.for %arg2 = %c0 to %c64 step %c1 {
          %0 = memref.load %tile_0_3_buff_0[%arg1, %arg2] : memref<64x64xi8>
          %1 = arith.addi %0, %c13_i8 : i8
          memref.store %1, %tile_0_3_buff_1[%arg1, %arg2] : memref<64x64xi8>
        }
      }
      %c1_ul10 = arith.constant 1 : i32
      aie.use_lock(%tile_0_3_lock_0, Release, %c1_ul10)
      %c1_ul11 = arith.constant 1 : i32
      aie.use_lock(%tile_0_3_lock_2, Release, %c1_ul11)
      aie.end
    }

    %mem_0_3 = aie.mem(%tile_0_3) {
      %0 = aie.dma(S2MM, 0) [{
        %c1_ul12 = arith.constant 1 : i32
        aie.use_lock(%tile_0_3_lock_0, AcquireGreaterEqual, %c1_ul12)
        aie.dma_bd(%tile_0_3_buff_0 : memref<64x64xi8>)
        %c1_ul13 = arith.constant 1 : i32
        aie.use_lock(%tile_0_3_lock_1, Release, %c1_ul13)
      }]
      %1 = aie.dma(MM2S, 0) [{
        %c1_ul14 = arith.constant 1 : i32
        aie.use_lock(%tile_0_3_lock_2, AcquireGreaterEqual, %c1_ul14)
        aie.dma_bd(%tile_0_3_buff_1 : memref<64x64xi8>)
        %c1_ul15 = arith.constant 1 : i32
        aie.use_lock(%tile_0_3_lock_3, Release, %c1_ul15)
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
      %c1_ul16 = arith.constant 1 : i32
      aie.use_lock(%tile_0_4_lock_3, AcquireGreaterEqual, %c1_ul16)
      %c1_ul17 = arith.constant 1 : i32
      aie.use_lock(%tile_0_4_lock_1, AcquireGreaterEqual, %c1_ul17)
      scf.for %arg1 = %c0 to %c64 step %c1 {
        scf.for %arg2 = %c0 to %c64 step %c1 {
          %0 = memref.load %tile_0_4_buff_0[%arg1, %arg2] : memref<64x64xi8>
          %1 = arith.addi %0, %c14_i8 : i8
          memref.store %1, %tile_0_4_buff_1[%arg1, %arg2] : memref<64x64xi8>
        }
      }
      %c1_ul18 = arith.constant 1 : i32
      aie.use_lock(%tile_0_4_lock_0, Release, %c1_ul18)
      %c1_ul19 = arith.constant 1 : i32
      aie.use_lock(%tile_0_4_lock_2, Release, %c1_ul19)
      aie.end
    }

    %mem_0_4 = aie.mem(%tile_0_4) {
      %0 = aie.dma(S2MM, 0) [{
        %c1_ul20 = arith.constant 1 : i32
        aie.use_lock(%tile_0_4_lock_0, AcquireGreaterEqual, %c1_ul20)
        aie.dma_bd(%tile_0_4_buff_0 : memref<64x64xi8>)
        %c1_ul21 = arith.constant 1 : i32
        aie.use_lock(%tile_0_4_lock_1, Release, %c1_ul21)
      }]
      %1 = aie.dma(MM2S, 0) [{
        %c1_ul22 = arith.constant 1 : i32
        aie.use_lock(%tile_0_4_lock_2, AcquireGreaterEqual, %c1_ul22)
        aie.dma_bd(%tile_0_4_buff_1 : memref<64x64xi8>)
        %c1_ul23 = arith.constant 1 : i32
        aie.use_lock(%tile_0_4_lock_3, Release, %c1_ul23)
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
      %c1_ul24 = arith.constant 1 : i32
      aie.use_lock(%tile_0_5_lock_3, AcquireGreaterEqual, %c1_ul24)
      %c1_ul25 = arith.constant 1 : i32
      aie.use_lock(%tile_0_5_lock_1, AcquireGreaterEqual, %c1_ul25)
      scf.for %arg1 = %c0 to %c64 step %c1 {
        scf.for %arg2 = %c0 to %c64 step %c1 {
          %0 = memref.load %tile_0_5_buff_0[%arg1, %arg2] : memref<64x64xi8>
          %1 = arith.addi %0, %c15_i8 : i8
          memref.store %1, %tile_0_5_buff_1[%arg1, %arg2] : memref<64x64xi8>
        }
      }
      %c1_ul26 = arith.constant 1 : i32
      aie.use_lock(%tile_0_5_lock_0, Release, %c1_ul26)
      %c1_ul27 = arith.constant 1 : i32
      aie.use_lock(%tile_0_5_lock_2, Release, %c1_ul27)
      aie.end
    }

    %mem_0_5 = aie.mem(%tile_0_5) {
      %0 = aie.dma(S2MM, 0) [{
        %c1_ul28 = arith.constant 1 : i32
        aie.use_lock(%tile_0_5_lock_0, AcquireGreaterEqual, %c1_ul28)
        aie.dma_bd(%tile_0_5_buff_0 : memref<64x64xi8>)
        %c1_ul29 = arith.constant 1 : i32
        aie.use_lock(%tile_0_5_lock_1, Release, %c1_ul29)
      }]
      %1 = aie.dma(MM2S, 0) [{
        %c1_ul30 = arith.constant 1 : i32
        aie.use_lock(%tile_0_5_lock_2, AcquireGreaterEqual, %c1_ul30)
        aie.dma_bd(%tile_0_5_buff_1 : memref<64x64xi8>)
        %c1_ul31 = arith.constant 1 : i32
        aie.use_lock(%tile_0_5_lock_3, Release, %c1_ul31)
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
        %c4_ul32 = arith.constant 4 : i32
        aie.use_lock(%memtile_lock_0, AcquireGreaterEqual, %c4_ul32)
        aie.dma_bd(%buff_0 : memref<4x64x64xi8>, 0, 16384)
        %c4_ul33 = arith.constant 4 : i32
        aie.use_lock(%memtile_lock_1, Release, %c4_ul33)
      }]
      %1 = aie.dma(MM2S, 0) [{
        %c1_ul34 = arith.constant 1 : i32
        aie.use_lock(%memtile_lock_1, AcquireGreaterEqual, %c1_ul34)
        aie.dma_bd(%buff_0 : memref<4x64x64xi8>, 0, 4096)
        %c1_ul35 = arith.constant 1 : i32
        aie.use_lock(%memtile_lock_0, Release, %c1_ul35)
      }]
      %2 = aie.dma(MM2S, 1) [{
        %c1_ul36 = arith.constant 1 : i32
        aie.use_lock(%memtile_lock_1, AcquireGreaterEqual, %c1_ul36)
        aie.dma_bd(%buff_0 : memref<4x64x64xi8>, 4096, 4096)
        %c1_ul37 = arith.constant 1 : i32
        aie.use_lock(%memtile_lock_0, Release, %c1_ul37)
      }]
      %3 = aie.dma(MM2S, 2) [{
        %c1_ul38 = arith.constant 1 : i32
        aie.use_lock(%memtile_lock_1, AcquireGreaterEqual, %c1_ul38)
        aie.dma_bd(%buff_0 : memref<4x64x64xi8>, 8192, 4096)
        %c1_ul39 = arith.constant 1 : i32
        aie.use_lock(%memtile_lock_0, Release, %c1_ul39)
      }]
      %4 = aie.dma(MM2S, 3) [{
        %c1_ul40 = arith.constant 1 : i32
        aie.use_lock(%memtile_lock_1, AcquireGreaterEqual, %c1_ul40)
        aie.dma_bd(%buff_0 : memref<4x64x64xi8>, 12288, 4096)
        %c1_ul41 = arith.constant 1 : i32
        aie.use_lock(%memtile_lock_0, Release, %c1_ul41)
      }]

      %5 = aie.dma(S2MM, 1) [{
        %c1_ul42 = arith.constant 1 : i32
        aie.use_lock(%memtile_lock_3, AcquireGreaterEqual, %c1_ul42)
        aie.dma_bd(%buff_1 : memref<4x64x64xi8>, 0, 4096)
        %c1_ul43 = arith.constant 1 : i32
        aie.use_lock(%memtile_lock_2, Release, %c1_ul43)
      }]
      %6 = aie.dma(S2MM, 2) [{
        %c1_ul44 = arith.constant 1 : i32
        aie.use_lock(%memtile_lock_3, AcquireGreaterEqual, %c1_ul44)
        aie.dma_bd(%buff_1 : memref<4x64x64xi8>, 4096, 4096)
        %c1_ul45 = arith.constant 1 : i32
        aie.use_lock(%memtile_lock_2, Release, %c1_ul45)
      }]
      %7 = aie.dma(S2MM, 3) [{
        %c1_ul46 = arith.constant 1 : i32
        aie.use_lock(%memtile_lock_3, AcquireGreaterEqual, %c1_ul46)
        aie.dma_bd(%buff_1 : memref<4x64x64xi8>, 8192, 4096)
        %c1_ul47 = arith.constant 1 : i32
        aie.use_lock(%memtile_lock_2, Release, %c1_ul47)
      }]
      %8 = aie.dma(S2MM, 4) [{
        %c1_ul48 = arith.constant 1 : i32
        aie.use_lock(%memtile_lock_3, AcquireGreaterEqual, %c1_ul48)
        aie.dma_bd(%buff_1 : memref<4x64x64xi8>, 12288, 4096)
        %c1_ul49 = arith.constant 1 : i32
        aie.use_lock(%memtile_lock_2, Release, %c1_ul49)
      }]
      %9 = aie.dma(MM2S, 4) [{
        %c4_ul50 = arith.constant 4 : i32
        aie.use_lock(%memtile_lock_2, AcquireGreaterEqual, %c4_ul50)
        aie.dma_bd(%buff_1 : memref<4x64x64xi8>, 0, 16384)
        %c4_ul51 = arith.constant 4 : i32
        aie.use_lock(%memtile_lock_3, Release, %c4_ul51)
      }]
      aie.end
    }

    aie.shim_dma_allocation @objFifo_in0 (%tile_0_0, MM2S, 0)
    aie.shim_dma_allocation @objFifo_out0 (%tile_0_0, S2MM, 0)

    aie.runtime_sequence @run(%arg0: memref<4x64x64xi8>, %arg1: memref<4x64x64xi8>) {
      %c0_i64 = arith.constant 0 : i64
      %c1_i64 = arith.constant 1 : i64
      %c4_i64 = arith.constant 4 : i64
      %c4096_i64 = arith.constant 4096 : i64
      %c64_i64 = arith.constant 64 : i64
      aiex.configure @main {
        aiex.npu.dma_memcpy_nd (%arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64][%c1_i64, %c4_i64, %c64_i64, %c64_i64][%c0_i64, %c4096_i64, %c64_i64, %c1_i64], packet = <pkt_id = 0, pkt_type = 0>) {id = 0 : i64, metadata = @objFifo_in0} : memref<4x64x64xi8>
        aiex.npu.dma_memcpy_nd (%arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64][%c1_i64, %c4_i64, %c64_i64, %c64_i64][%c0_i64, %c4096_i64, %c64_i64, %c1_i64]) {id = 1 : i64, metadata = @objFifo_out0, issue_token = true} : memref<4x64x64xi8>
        aiex.npu.dma_wait { symbol = @objFifo_out0 }
      }
    }
  }
}
