//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2022-2024 Advanced Micro Devices, Inc.
// Copyright (C) 2020-2022 Xilinx, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

module {
  aie.device(npu1_1col) {
    func.func private @scale_int32(memref<1024xi32>, memref<1024xi32>) attributes {link_with = "scale.o"}

    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    %in_cons_buff_0 = aie.buffer(%tile_0_2) {sym_name = "in_cons_buff_0"} : memref<1024xi32>
    %in_cons_buff_1 = aie.buffer(%tile_0_2) {sym_name = "in_cons_buff_1"} : memref<1024xi32>
    %out_buff_0 = aie.buffer(%tile_0_2) {sym_name = "out_buff_0"} : memref<1024xi32>
    %out_buff_1 = aie.buffer(%tile_0_2) {sym_name = "out_buff_1"} : memref<1024xi32>

    %in_cons_cons_lock = aie.lock(%tile_0_2, 1) {init = 0 : i32, sym_name = "in_cons_cons_lock"}
    %in_cons_lock = aie.lock(%tile_0_0, 1) {init = 0 : i32, sym_name = "in_cons_lock"}
    %in_cons_prod_lock = aie.lock(%tile_0_2, 0) {init = 2 : i32, sym_name = "in_cons_prod_lock"}
    %in_prod_lock = aie.lock(%tile_0_0, 0) {init = 0 : i32, sym_name = "in_prod_lock"}
    %out_cons_cons_lock = aie.lock(%tile_0_0, 3) {init = 0 : i32, sym_name = "out_cons_cons_lock"}
    %out_cons_lock = aie.lock(%tile_0_2, 3) {init = 0 : i32, sym_name = "out_cons_lock"}
    %out_cons_prod_lock = aie.lock(%tile_0_0, 2) {init = 0 : i32, sym_name = "out_cons_prod_lock"}
    %out_prod_lock = aie.lock(%tile_0_2, 2) {init = 2 : i32, sym_name = "out_prod_lock"}

    aie.flow(%tile_0_0, DMA : 0, %tile_0_2, DMA : 0)
    aie.flow(%tile_0_2, DMA : 0, %tile_0_0, DMA : 0)

    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c4 = arith.constant 4 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index

      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        scf.for %arg1 = %c0 to %c4 step %c2 {
          %c1_ul0 = arith.constant 1 : i32
          aie.use_lock(%in_cons_cons_lock, AcquireGreaterEqual, %c1_ul0)
          %c1_ul1 = arith.constant 1 : i32
          aie.use_lock(%out_prod_lock, AcquireGreaterEqual, %c1_ul1)
          func.call @scale_int32(%in_cons_buff_0, %out_buff_0) : (memref<1024xi32>, memref<1024xi32>) -> ()
          %c1_ul2 = arith.constant 1 : i32
          aie.use_lock(%in_cons_prod_lock, Release, %c1_ul2)
          %c1_ul3 = arith.constant 1 : i32
          aie.use_lock(%out_cons_lock, Release, %c1_ul3)

          %c1_ul4 = arith.constant 1 : i32
          aie.use_lock(%out_prod_lock, AcquireGreaterEqual, %c1_ul4)
          %c1_ul5 = arith.constant 1 : i32
          aie.use_lock(%in_cons_cons_lock, AcquireGreaterEqual, %c1_ul5)
          func.call @scale_int32(%in_cons_buff_1, %out_buff_1) : (memref<1024xi32>, memref<1024xi32>) -> ()
          %c1_ul6 = arith.constant 1 : i32
          aie.use_lock(%in_cons_prod_lock, Release, %c1_ul6)
          %c1_ul7 = arith.constant 1 : i32
          aie.use_lock(%out_cons_lock, Release, %c1_ul7)
        }
      }
      aie.end
    }

    aie.shim_dma_allocation @in (%tile_0_0, MM2S, 0)

    aie.runtime_sequence(%arg0: memref<4096xi32>, %arg1: memref<4096xi32>, %arg2: memref<4096xi32>) {
      %c0_i64 = arith.constant 0 : i64
      %c1_i64 = arith.constant 1 : i64
      %c4096_i64 = arith.constant 4096 : i64
      aiex.npu.dma_memcpy_nd(%arg2[%c0_i64, %c0_i64, %c0_i64, %c0_i64] [%c1_i64, %c1_i64, %c1_i64, %c4096_i64] [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {id = 0 : i64, metadata = @out, issue_token = true} : memref<4096xi32>
      aiex.npu.dma_memcpy_nd(%arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64] [%c1_i64, %c1_i64, %c1_i64, %c4096_i64] [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {id = 1 : i64, metadata = @in} : memref<4096xi32>
      aiex.npu.dma_wait { symbol = @out }
    }

    aie.shim_dma_allocation @out (%tile_0_0, S2MM, 0)

    %mem_0_2 = aie.mem(%tile_0_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      %c1_ul8 = arith.constant 1 : i32
      aie.use_lock(%in_cons_prod_lock, AcquireGreaterEqual, %c1_ul8)
      aie.dma_bd(%in_cons_buff_0 : memref<1024xi32>, 0, 1024)
      %c1_ul9 = arith.constant 1 : i32
      aie.use_lock(%in_cons_cons_lock, Release, %c1_ul9)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      %c1_ul10 = arith.constant 1 : i32
      aie.use_lock(%in_cons_prod_lock, AcquireGreaterEqual, %c1_ul10)
      aie.dma_bd(%in_cons_buff_1 : memref<1024xi32>, 0, 1024)
      %c1_ul11 = arith.constant 1 : i32
      aie.use_lock(%in_cons_cons_lock, Release, %c1_ul11)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      %c1_ul12 = arith.constant 1 : i32
      aie.use_lock(%out_cons_lock, AcquireGreaterEqual, %c1_ul12)
      aie.dma_bd(%out_buff_0 : memref<1024xi32>, 0, 1024)
      %c1_ul13 = arith.constant 1 : i32
      aie.use_lock(%out_prod_lock, Release, %c1_ul13)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      %c1_ul14 = arith.constant 1 : i32
      aie.use_lock(%out_cons_lock, AcquireGreaterEqual, %c1_ul14)
      aie.dma_bd(%out_buff_1 : memref<1024xi32>, 0, 1024)
      %c1_ul15 = arith.constant 1 : i32
      aie.use_lock(%out_prod_lock, Release, %c1_ul15)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      aie.end
    }
  }
}
