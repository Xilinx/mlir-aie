//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2022-2024 Advanced Micro Devices, Inc.
// Copyright (C) 2020-2022 Xilinx, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

module {
  aie.device(NPUDEVICE) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_1 = aie.tile(0, 1)
    %tile_0_2 = aie.tile(0, 2)

    %objFifo_in1_cons_buff_0 = aie.buffer(%tile_0_2) {sym_name = "objFifo_in1_cons_buff_0"} : memref<64x64xi8>
    %objFifo_in1_cons_buff_1 = aie.buffer(%tile_0_2) {sym_name = "objFifo_in1_cons_buff_1"} : memref<64x64xi8>
    %objFifo_out1_buff_0 = aie.buffer(%tile_0_2) {sym_name = "objFifo_out1_buff_0"} : memref<64x64xi8>
    %objFifo_out1_buff_1 = aie.buffer(%tile_0_2) {sym_name = "objFifo_out1_buff_1"} : memref<64x64xi8>

    %objFifo_in1_cons_prod_lock = aie.lock(%tile_0_2, 0) {init = 1 : i32, sym_name = "objFifo_in1_cons_prod_lock"}
    %objFifo_in1_cons_cons_lock = aie.lock(%tile_0_2, 1) {init = 0 : i32, sym_name = "objFifo_in1_cons_cons_lock"}
    %objFifo_out1_prod_lock = aie.lock(%tile_0_2, 2) {init = 1 : i32, sym_name = "objFifo_out1_prod_lock"}
    %objFifo_out1_cons_lock = aie.lock(%tile_0_2, 3) {init = 0 : i32, sym_name = "objFifo_out1_cons_lock"}

    aie.flow(%tile_0_0, DMA : 0, %tile_0_1, DMA : 0)
    aie.flow(%tile_0_1, DMA : 0, %tile_0_2, DMA : 0)
    aie.flow(%tile_0_1, DMA : 1, %tile_0_0, DMA : 0)
    aie.flow(%tile_0_2, DMA : 0, %tile_0_1, DMA : 1)

    %core_0_2 = aie.core(%tile_0_2) {
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c12_i8 = arith.constant 12 : i8
      %c2 = arith.constant 2 : index
      %c64 = arith.constant 64 : index
      %c1_ul1 = arith.constant 1 : i32
      aie.use_lock(%objFifo_in1_cons_cons_lock, AcquireGreaterEqual, %c1_ul1)
      %c1_ul2 = arith.constant 1 : i32
      aie.use_lock(%objFifo_out1_prod_lock, AcquireGreaterEqual, %c1_ul2)
      scf.for %arg1 = %c0 to %c64 step %c1 {
        scf.for %arg2 = %c0 to %c64 step %c1 {
          %0 = memref.load %objFifo_in1_cons_buff_0[%arg1, %arg2] : memref<64x64xi8>
          %1 = arith.addi %0, %c12_i8 : i8
          memref.store %1, %objFifo_out1_buff_0[%arg1, %arg2] : memref<64x64xi8>
        }
      }
      %c1_ul3 = arith.constant 1 : i32
      aie.use_lock(%objFifo_in1_cons_prod_lock, Release, %c1_ul3)
      %c1_ul4 = arith.constant 1 : i32
      aie.use_lock(%objFifo_out1_cons_lock, Release, %c1_ul4)
      aie.end
    }

    aie.shim_dma_allocation @objFifo_in0 (%tile_0_0, MM2S, 0)

    aie.runtime_sequence(%arg0: memref<61x56xi8>, %arg1: memref<32xi8>, %arg2: memref<64x64xi8>) {
      %c0_i64 = arith.constant 0 : i64
      %c1_i64 = arith.constant 1 : i64
      %c56_i64 = arith.constant 56 : i64
      %c61_i64 = arith.constant 61 : i64
      %c64_i64 = arith.constant 64 : i64
      aiex.npu.dma_memcpy_nd (%arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64][%c1_i64, %c1_i64, %c61_i64, %c56_i64][%c0_i64, %c0_i64, %c56_i64, %c1_i64]) {id = 0 : i64, metadata = @objFifo_in0} : memref<61x56xi8>
      aiex.npu.dma_memcpy_nd (%arg2[%c0_i64, %c0_i64, %c0_i64, %c0_i64][%c1_i64, %c1_i64, %c64_i64, %c64_i64][%c0_i64, %c0_i64, %c64_i64, %c1_i64]) {id = 1 : i64, metadata = @objFifo_out0, issue_token = true} : memref<64x64xi8>
      aiex.npu.dma_wait { symbol = @objFifo_out0 }
    }

    %memtile_dma_0_1 = aie.memtile_dma(%tile_0_1) {
      %objFifo_in0_cons_buff_0 = aie.buffer(%tile_0_1) {sym_name = "objFifo_in0_cons_buff_0"} : memref<64x64xi8>
      %objFifo_in0_cons_buff_1 = aie.buffer(%tile_0_1) {sym_name = "objFifo_in0_cons_buff_1"} : memref<64x64xi8>
      %objFifo_out0_buff_0 = aie.buffer(%tile_0_1) {sym_name = "objFifo_out0_buff_0"} : memref<64x64xi8>
      %objFifo_out0_buff_1 = aie.buffer(%tile_0_1) {sym_name = "objFifo_out0_buff_1"} : memref<64x64xi8>
      %objFifo_in0_cons_prod_lock = aie.lock(%tile_0_1, 0) {init = 1 : i32, sym_name = "objFifo_in0_cons_prod_lock"}
      %objFifo_in0_cons_cons_lock = aie.lock(%tile_0_1, 1) {init = 0 : i32, sym_name = "objFifo_in0_cons_cons_lock"}
      %objFifo_out0_prod_lock = aie.lock(%tile_0_1, 2) {init = 1 : i32, sym_name = "objFifo_out0_prod_lock"}
      %objFifo_out0_cons_lock = aie.lock(%tile_0_1, 3) {init = 0 : i32, sym_name = "objFifo_out0_cons_lock"}
      %0 = aie.dma(S2MM, 0) [{
        %c1_ul5 = arith.constant 1 : i32
        aie.use_lock(%objFifo_in0_cons_prod_lock, AcquireGreaterEqual, %c1_ul5)
        aie.dma_bd(%objFifo_in0_cons_buff_0 : memref<64x64xi8> offset = 0 len = 3416)
        %c1_ul6 = arith.constant 1 : i32
        aie.use_lock(%objFifo_in0_cons_cons_lock, Release, %c1_ul6)
      }]
      %1 = aie.dma(MM2S, 0) [{
        %c1_ul7 = arith.constant 1 : i32
        aie.use_lock(%objFifo_in0_cons_cons_lock, AcquireGreaterEqual, %c1_ul7)
        aie.dma_bd(%objFifo_in0_cons_buff_0 : memref<64x64xi8> offset = 0 len = 4096 sizes = [61, 56] strides = [56, 1] pad [<const_pad_before = 2, const_pad_after = 1>, <const_pad_before = 4, const_pad_after = 4>])
        %c1_ul8 = arith.constant 1 : i32
        aie.use_lock(%objFifo_in0_cons_prod_lock, Release, %c1_ul8)
      }]
      %2 = aie.dma(MM2S, 1) [{
        %c1_ul9 = arith.constant 1 : i32
        aie.use_lock(%objFifo_out0_cons_lock, AcquireGreaterEqual, %c1_ul9)
        aie.dma_bd(%objFifo_out0_buff_0 : memref<64x64xi8>)
        %c1_ul10 = arith.constant 1 : i32
        aie.use_lock(%objFifo_out0_prod_lock, Release, %c1_ul10)
      }]
      %3 = aie.dma(S2MM, 1) [{
        %c1_ul11 = arith.constant 1 : i32
        aie.use_lock(%objFifo_out0_prod_lock, AcquireGreaterEqual, %c1_ul11)
        aie.dma_bd(%objFifo_out0_buff_0 : memref<64x64xi8>)
        %c1_ul12 = arith.constant 1 : i32
        aie.use_lock(%objFifo_out0_cons_lock, Release, %c1_ul12)
      }]
      aie.end
    }

    aie.shim_dma_allocation @objFifo_out0 (%tile_0_0, S2MM, 0)

    %mem_0_2 = aie.mem(%tile_0_2) {
      %0 = aie.dma(S2MM, 0) [{
        %c1_ul13 = arith.constant 1 : i32
        aie.use_lock(%objFifo_in1_cons_prod_lock, AcquireGreaterEqual, %c1_ul13)
        aie.dma_bd(%objFifo_in1_cons_buff_0 : memref<64x64xi8>)
        %c1_ul14 = arith.constant 1 : i32
        aie.use_lock(%objFifo_in1_cons_cons_lock, Release, %c1_ul14)
      }]
      %1 = aie.dma(MM2S, 0) [{
        %c1_ul15 = arith.constant 1 : i32
        aie.use_lock(%objFifo_out1_cons_lock, AcquireGreaterEqual, %c1_ul15)
        aie.dma_bd(%objFifo_out1_buff_0 : memref<64x64xi8>)
        %c1_ul16 = arith.constant 1 : i32
        aie.use_lock(%objFifo_out1_prod_lock, Release, %c1_ul16)
      }]
      aie.end
    }
  }
}
