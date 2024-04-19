//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2020-2022, Xilinx Inc.
// Copyright (C) 2022, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

module {
  aie.device(npu) {
    memref.global "public" @inA : memref<64x32xi16>
    memref.global "public" @inA_cons : memref<64x32xi16>
    memref.global "public" @inB : memref<32x64xi16>
    memref.global "public" @inB_cons : memref<32x64xi16>
    memref.global "public" @aie.memA : memref<64x32xi16>
    memref.global "public" @aie.memA_cons : memref<64x32xi16>
    memref.global "public" @aie.memB : memref<32x64xi16>
    memref.global "public" @aie.memB_cons : memref<32x64xi16>
    memref.global "public" @aie.memC : memref<64x64xi16>
    memref.global "public" @aie.memC_cons : memref<64x64xi16>
    memref.global "public" @outC : memref<64x64xi16>
    memref.global "public" @outC_cons : memref<64x64xi16>

    func.func private @matmul_i16_i16(memref<64x32xi16>, memref<32x64xi16>, memref<64x64xi16>)
    func.func private @matmul_scalar_i16_i16(memref<64x32xi16>, memref<32x64xi16>, memref<64x64xi16>)
    func.func private @zero_i16(memref<64x64xi16>)
    func.func private @zero_scalar_i16(memref<64x64xi16>)

    %tile_0_0 = aie.tile(0, 0)
    %tile_0_1 = aie.tile(0, 1)
    %tile_0_2 = aie.tile(0, 2)

    %memA_cons_buff_0 = aie.buffer(%tile_0_2) {sym_name = "memA_cons_buff_0"} : memref<64x32xi16>
    %memA_cons_buff_1 = aie.buffer(%tile_0_2) {sym_name = "memA_cons_buff_1"} : memref<64x32xi16>
    %memB_cons_buff_0 = aie.buffer(%tile_0_2) {sym_name = "memB_cons_buff_0"} : memref<32x64xi16>
    %memB_cons_buff_1 = aie.buffer(%tile_0_2) {sym_name = "memB_cons_buff_1"} : memref<32x64xi16>
    %memC_buff_0 = aie.buffer(%tile_0_2) {sym_name = "memC_buff_0"} : memref<64x64xi16>
    %memC_buff_1 = aie.buffer(%tile_0_2) {sym_name = "memC_buff_1"} : memref<64x64xi16>

    %memA_cons_cons_lock = aie.lock(%tile_0_2, 1) {init = 0 : i32, sym_name = "memA_cons_cons_lock"}
    %memA_cons_prod_lock = aie.lock(%tile_0_2, 0) {init = 2 : i32, sym_name = "memA_cons_prod_lock"}
    %memB_cons_cons_lock = aie.lock(%tile_0_2, 3) {init = 0 : i32, sym_name = "memB_cons_cons_lock"}
    %memB_cons_prod_lock = aie.lock(%tile_0_2, 2) {init = 2 : i32, sym_name = "memB_cons_prod_lock"}
    %memC_cons_cons_lock = aie.lock(%tile_0_1, 5) {init = 0 : i32, sym_name = "memC_cons_cons_lock"}
    %memC_cons_lock = aie.lock(%tile_0_2, 5) {init = 0 : i32, sym_name = "memC_cons_lock"}
    %memC_cons_prod_lock = aie.lock(%tile_0_1, 4) {init = 2 : i32, sym_name = "memC_cons_prod_lock"}
    %memC_prod_lock = aie.lock(%tile_0_2, 4) {init = 2 : i32, sym_name = "memC_prod_lock"}

    aie.flow(%tile_0_0, DMA : 0, %tile_0_1, DMA : 0)
    aie.flow(%tile_0_1, DMA : 0, %tile_0_2, DMA : 0)
    aie.flow(%tile_0_0, DMA : 1, %tile_0_1, DMA : 1)
    aie.flow(%tile_0_1, DMA : 1, %tile_0_2, DMA : 1)
    aie.flow(%tile_0_2, DMA : 0, %tile_0_1, DMA : 2)
    aie.flow(%tile_0_1, DMA : 2, %tile_0_0, DMA : 0)

    %core_0_2 = aie.core(%tile_0_2) {
      %c4 = arith.constant 4 : index
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c2 = arith.constant 2 : index
        scf.for %arg1 = %c0 to %c4 step %c2 {
          aie.use_lock(%memC_prod_lock, AcquireGreaterEqual)
          func.call @zero_i16(%memC_buff_0) : (memref<64x64xi16>) -> ()
          %c2_0 = arith.constant 2 : index
          scf.for %arg2 = %c0 to %c4 step %c2_0 {
            aie.use_lock(%memA_cons_cons_lock, AcquireGreaterEqual)
            aie.use_lock(%memB_cons_cons_lock, AcquireGreaterEqual)
            func.call @matmul_i16_i16(%memA_cons_buff_0, %memB_cons_buff_0, %memC_buff_0) : (memref<64x32xi16>, memref<32x64xi16>, memref<64x64xi16>) -> ()
            aie.use_lock(%memA_cons_prod_lock, Release)
            aie.use_lock(%memB_cons_prod_lock, Release)

            aie.use_lock(%memA_cons_cons_lock, AcquireGreaterEqual)
            aie.use_lock(%memB_cons_cons_lock, AcquireGreaterEqual)
            func.call @matmul_i16_i16(%memA_cons_buff_1, %memB_cons_buff_1, %memC_buff_0) : (memref<64x32xi16>, memref<32x64xi16>, memref<64x64xi16>) -> ()
            aie.use_lock(%memA_cons_prod_lock, Release)
            aie.use_lock(%memB_cons_prod_lock, Release)
          }
          aie.use_lock(%memC_cons_lock, Release)
          aie.use_lock(%memC_prod_lock, AcquireGreaterEqual)
          func.call @zero_i16(%memC_buff_1) : (memref<64x64xi16>) -> ()
          %c2_1 = arith.constant 2 : index
          scf.for %arg2 = %c0 to %c4 step %c2_1 {
            aie.use_lock(%memA_cons_cons_lock, AcquireGreaterEqual)
            aie.use_lock(%memB_cons_cons_lock, AcquireGreaterEqual)
            func.call @matmul_i16_i16(%memA_cons_buff_0, %memB_cons_buff_0, %memC_buff_1) : (memref<64x32xi16>, memref<32x64xi16>, memref<64x64xi16>) -> ()
            aie.use_lock(%memA_cons_prod_lock, Release)
            aie.use_lock(%memB_cons_prod_lock, Release)

            aie.use_lock(%memA_cons_cons_lock, AcquireGreaterEqual)
            aie.use_lock(%memB_cons_cons_lock, AcquireGreaterEqual)
            func.call @matmul_i16_i16(%memA_cons_buff_1, %memB_cons_buff_1, %memC_buff_1) : (memref<64x32xi16>, memref<32x64xi16>, memref<64x64xi16>) -> ()
            aie.use_lock(%memA_cons_prod_lock, Release)
            aie.use_lock(%memB_cons_prod_lock, Release)
          }
          aie.use_lock(%memC_cons_lock, Release)
        }
      }
      aie.end
    } {link_with = "mm.o"}

    aie.shim_dma_allocation @inA(MM2S, 0, 0)

    func.func @sequence(%arg0: memref<8192xi32>, %arg1: memref<8192xi32>, %arg2: memref<8192xi32>) {
      %c2048_i64 = arith.constant 2048 : i64
      %c16_i64 = arith.constant 16 : i64
      %c4_i64 = arith.constant 4 : i64
      %c0_i64 = arith.constant 0 : i64
      %c2_i64 = arith.constant 2 : i64
      %c64_i64 = arith.constant 64 : i64
      %c32_i64 = arith.constant 32 : i64
      %c4096_i64 = arith.constant 4096 : i64
      aiex.npu.dma_memcpy_nd(0, 0, %arg2[%c0_i64, %c0_i64, %c0_i64, %c0_i64] [%c2_i64, %c2_i64, %c64_i64, %c32_i64] [%c4096_i64, %c32_i64, %c64_i64]) {id = 0 : i64, metadata = @outC} : memref<8192xi32>
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64] [%c2_i64, %c4_i64, %c64_i64, %c16_i64] [%c0_i64, %c16_i64, %c64_i64]) {id = 1 : i64, metadata = @inA} : memref<8192xi32>
      aiex.npu.dma_memcpy_nd(0, 0, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64] [%c2_i64, %c4_i64, %c32_i64, %c32_i64] [%c32_i64, %c2048_i64, %c64_i64]) {id = 2 : i64, metadata = @inB} : memref<8192xi32>
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[%c0_i64, %c0_i64, %c0_i64, %c4096_i64] [%c2_i64, %c4_i64, %c64_i64, %c16_i64] [%c0_i64, %c16_i64, %c64_i64]) {id = 3 : i64, metadata = @inA} : memref<8192xi32>
      aiex.npu.dma_memcpy_nd(0, 0, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64] [%c2_i64, %c4_i64, %c32_i64, %c32_i64] [%c32_i64, %c2048_i64, %c64_i64]) {id = 4 : i64, metadata = @inB} : memref<8192xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}
      return
    }

    %memtile_dma_0_1 = aie.memtile_dma(%tile_0_1) {
      %inA_cons_buff_0 = aie.buffer(%tile_0_1) {sym_name = "inA_cons_buff_0"} : memref<64x32xi16>
      %inA_cons_buff_1 = aie.buffer(%tile_0_1) {sym_name = "inA_cons_buff_1"} : memref<64x32xi16>
      %inB_cons_buff_0 = aie.buffer(%tile_0_1) {sym_name = "inB_cons_buff_0"} : memref<32x64xi16>
      %inB_cons_buff_1 = aie.buffer(%tile_0_1) {sym_name = "inB_cons_buff_1"} : memref<32x64xi16>
      %memC_cons_buff_0 = aie.buffer(%tile_0_1) {sym_name = "memC_cons_buff_0"} : memref<64x64xi16>
      %memC_cons_buff_1 = aie.buffer(%tile_0_1) {sym_name = "memC_cons_buff_1"} : memref<64x64xi16>
      %inA_cons_prod_lock = aie.lock(%tile_0_1, 0) {init = 2 : i32, sym_name = "inA_cons_prod_lock"}
      %inA_cons_cons_lock = aie.lock(%tile_0_1, 1) {init = 0 : i32, sym_name = "inA_cons_cons_lock"}
      %inB_cons_cons_lock = aie.lock(%tile_0_1, 3) {init = 0 : i32, sym_name = "inB_cons_cons_lock"}
      %inB_cons_prod_lock = aie.lock(%tile_0_1, 2) {init = 2 : i32, sym_name = "inB_cons_prod_lock"}
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%inA_cons_prod_lock, AcquireGreaterEqual)
      aie.dma_bd(%inA_cons_buff_0 : memref<64x32xi16>, 0, 2048)
      aie.use_lock(%inA_cons_cons_lock, Release)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%inA_cons_prod_lock, AcquireGreaterEqual)
      aie.dma_bd(%inA_cons_buff_1 : memref<64x32xi16>, 0, 2048)
      aie.use_lock(%inA_cons_cons_lock, Release)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%inA_cons_cons_lock, AcquireGreaterEqual)
      aie.dma_bd(%inA_cons_buff_0 : memref<64x32xi16>, 0, 2048, [<size = 16, stride = 128>, <size = 8, stride = 4>, <size = 4, stride = 32>, <size = 4, stride = 1>])
      aie.use_lock(%inA_cons_prod_lock, Release)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%inA_cons_cons_lock, AcquireGreaterEqual)
      aie.dma_bd(%inA_cons_buff_1 : memref<64x32xi16>, 0, 2048, [<size = 16, stride = 128>, <size = 8, stride = 4>, <size = 4, stride = 32>, <size = 4, stride = 1>])
      aie.use_lock(%inA_cons_prod_lock, Release)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb7, ^bb9)
    ^bb7:  // 2 preds: ^bb6, ^bb8
      aie.use_lock(%inB_cons_prod_lock, AcquireGreaterEqual)
      aie.dma_bd(%inB_cons_buff_0 : memref<32x64xi16>, 0, 2048)
      aie.use_lock(%inB_cons_cons_lock, Release)
      aie.next_bd ^bb8
    ^bb8:  // pred: ^bb7
      aie.use_lock(%inB_cons_prod_lock, AcquireGreaterEqual)
      aie.dma_bd(%inB_cons_buff_1 : memref<32x64xi16>, 0, 2048)
      aie.use_lock(%inB_cons_cons_lock, Release)
      aie.next_bd ^bb7
    ^bb9:  // pred: ^bb6
      %3 = aie.dma_start(MM2S, 1, ^bb10, ^bb12)
    ^bb10:  // 2 preds: ^bb9, ^bb11
      aie.use_lock(%inB_cons_cons_lock, AcquireGreaterEqual)
      aie.dma_bd(%inB_cons_buff_0 : memref<32x64xi16>, 0, 2048, [<size = 8, stride = 256>, <size = 16, stride = 4>, <size = 4, stride = 64>, <size = 4, stride = 1>])
      aie.use_lock(%inB_cons_prod_lock, Release)
      aie.next_bd ^bb11
    ^bb11:  // pred: ^bb10
      aie.use_lock(%inB_cons_cons_lock, AcquireGreaterEqual)
      aie.dma_bd(%inB_cons_buff_1 : memref<32x64xi16>, 0, 2048, [<size = 8, stride = 256>, <size = 16, stride = 4>, <size = 4, stride = 64>, <size = 4, stride = 1>])
      aie.use_lock(%inB_cons_prod_lock, Release)
      aie.next_bd ^bb10
    ^bb12:  // pred: ^bb9
      %4 = aie.dma_start(S2MM, 2, ^bb13, ^bb15)
    ^bb13:  // 2 preds: ^bb12, ^bb14
      aie.use_lock(%memC_cons_prod_lock, AcquireGreaterEqual)
      aie.dma_bd(%memC_cons_buff_0 : memref<64x64xi16>, 0, 4096)
      aie.use_lock(%memC_cons_cons_lock, Release)
      aie.next_bd ^bb14
    ^bb14:  // pred: ^bb13
      aie.use_lock(%memC_cons_prod_lock, AcquireGreaterEqual)
      aie.dma_bd(%memC_cons_buff_1 : memref<64x64xi16>, 0, 4096)
      aie.use_lock(%memC_cons_cons_lock, Release)
      aie.next_bd ^bb13
    ^bb15:  // pred: ^bb12
      %5 = aie.dma_start(MM2S, 2, ^bb16, ^bb18)
    ^bb16:  // 2 preds: ^bb15, ^bb17
      aie.use_lock(%memC_cons_cons_lock, AcquireGreaterEqual)
      aie.dma_bd(%memC_cons_buff_0 : memref<64x64xi16>, 0, 4096, [<size = 16, stride = 256>, <size = 4, stride = 4>, <size = 16, stride = 16>, <size = 4, stride = 1>])
      aie.use_lock(%memC_cons_prod_lock, Release)
      aie.next_bd ^bb17
    ^bb17:  // pred: ^bb16
      aie.use_lock(%memC_cons_cons_lock, AcquireGreaterEqual)
      aie.dma_bd(%memC_cons_buff_1 : memref<64x64xi16>, 0, 4096, [<size = 16, stride = 256>, <size = 4, stride = 4>, <size = 16, stride = 16>, <size = 4, stride = 1>])
      aie.use_lock(%memC_cons_prod_lock, Release)
      aie.next_bd ^bb16
    ^bb18:  // pred: ^bb15
      aie.end
    }
    aie.shim_dma_allocation @inB(MM2S, 1, 0)
    aie.shim_dma_allocation @outC(S2MM, 0, 0)
    %mem_0_2 = aie.mem(%tile_0_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%memA_cons_prod_lock, AcquireGreaterEqual)
      aie.dma_bd(%memA_cons_buff_0 : memref<64x32xi16>, 0, 2048)
      aie.use_lock(%memA_cons_cons_lock, Release)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%memA_cons_prod_lock, AcquireGreaterEqual)
      aie.dma_bd(%memA_cons_buff_1 : memref<64x32xi16>, 0, 2048)
      aie.use_lock(%memA_cons_cons_lock, Release)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%memB_cons_prod_lock, AcquireGreaterEqual)
      aie.dma_bd(%memB_cons_buff_0 : memref<32x64xi16>, 0, 2048)
      aie.use_lock(%memB_cons_cons_lock, Release)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%memB_cons_prod_lock, AcquireGreaterEqual)
      aie.dma_bd(%memB_cons_buff_1 : memref<32x64xi16>, 0, 2048)
      aie.use_lock(%memB_cons_cons_lock, Release)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb7, ^bb9)
    ^bb7:  // 2 preds: ^bb6, ^bb8
      aie.use_lock(%memC_cons_lock, AcquireGreaterEqual)
      aie.dma_bd(%memC_buff_0 : memref<64x64xi16>, 0, 4096)
      aie.use_lock(%memC_prod_lock, Release)
      aie.next_bd ^bb8
    ^bb8:  // pred: ^bb7
      aie.use_lock(%memC_cons_lock, AcquireGreaterEqual)
      aie.dma_bd(%memC_buff_1 : memref<64x64xi16>, 0, 4096)
      aie.use_lock(%memC_prod_lock, Release)
      aie.next_bd ^bb7
    ^bb9:  // pred: ^bb6
      aie.end
    }
  }
}
