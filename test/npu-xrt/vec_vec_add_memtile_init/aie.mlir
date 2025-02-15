//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2024 Advanced Micro Devices, Inc. or its affiliates
//
//===----------------------------------------------------------------------===//

module {
  aie.device(npu1_1col) {
    memref.global "public" @out_cons : memref<16xi32>
    memref.global "public" @out : memref<16xi32>
    memref.global "public" @in2_mem_cons : memref<256xi32>
    memref.global "public" @in2_mem : memref<256xi32>
    memref.global "public" @in1_cons : memref<16xi32>
    memref.global "public" @in1 : memref<16xi32>
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_1 = aie.tile(0, 1)
    %tile_0_2 = aie.tile(0, 2)
    %out_buff_0 = aie.buffer(%tile_0_2) {sym_name = "out_buff_0"} : memref<16xi32> 
    %out_buff_1 = aie.buffer(%tile_0_2) {sym_name = "out_buff_1"} : memref<16xi32> 
    %out_prod_lock = aie.lock(%tile_0_2, 4) {init = 2 : i32, sym_name = "out_prod_lock"}
    %out_cons_lock = aie.lock(%tile_0_2, 5) {init = 0 : i32, sym_name = "out_cons_lock"}
    %in2_mem_cons_buff_0 = aie.buffer(%tile_0_2) {sym_name = "in2_mem_cons_buff_0"} : memref<256xi32> 
    %in2_mem_cons_buff_1 = aie.buffer(%tile_0_2) {sym_name = "in2_mem_cons_buff_1"} : memref<256xi32> 
    %in2_mem_cons_prod_lock = aie.lock(%tile_0_2, 2) {init = 2 : i32, sym_name = "in2_mem_cons_prod_lock"}
    %in2_mem_cons_cons_lock = aie.lock(%tile_0_2, 3) {init = 0 : i32, sym_name = "in2_mem_cons_cons_lock"}
    %in2_mem_buff_0 = aie.buffer(%tile_0_1) {sym_name = "in2_mem_buff_0"} : memref<256xi32> = dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255]> 
    %in2_mem_buff_1 = aie.buffer(%tile_0_1) {sym_name = "in2_mem_buff_1"} : memref<256xi32> = dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255]>
    %in2_mem_prod_lock = aie.lock(%tile_0_1, 0) {init = 0 : i32, sym_name = "in2_mem_prod_lock"}
    %in2_mem_cons_lock = aie.lock(%tile_0_1, 1) {init = 2 : i32, sym_name = "in2_mem_cons_lock"}
    %in1_cons_buff_0 = aie.buffer(%tile_0_2) {sym_name = "in1_cons_buff_0"} : memref<16xi32> 
    %in1_cons_buff_1 = aie.buffer(%tile_0_2) {sym_name = "in1_cons_buff_1"} : memref<16xi32> 
    %in1_cons_prod_lock = aie.lock(%tile_0_2, 0) {init = 2 : i32, sym_name = "in1_cons_prod_lock"}
    %in1_cons_cons_lock = aie.lock(%tile_0_2, 1) {init = 0 : i32, sym_name = "in1_cons_cons_lock"}
    aie.flow(%tile_0_0, DMA : 0, %tile_0_2, DMA : 0)
    aie.flow(%tile_0_1, DMA : 0, %tile_0_2, DMA : 1)
    aie.flow(%tile_0_2, DMA : 0, %tile_0_0, DMA : 0)
    %core_0_2 = aie.core(%tile_0_2) {
      %c16 = arith.constant 16 : index
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c9223372036854775806 = arith.constant 9223372036854775806 : index
      %c2 = arith.constant 2 : index
      scf.for %arg0 = %c0 to %c9223372036854775806 step %c2 {
        aie.use_lock(%in2_mem_cons_cons_lock, AcquireGreaterEqual, 1)
        scf.for %arg1 = %c0 to %c16 step %c2 {
          aie.use_lock(%in1_cons_cons_lock, AcquireGreaterEqual, 1)
          aie.use_lock(%out_prod_lock, AcquireGreaterEqual, 1)
          scf.for %arg2 = %c0 to %c16 step %c1 {
            %1 = memref.load %in1_cons_buff_0[%arg2] : memref<16xi32>
            %2 = arith.muli %arg1, %c16 : index
            %3 = arith.addi %arg2, %2 : index
            %4 = memref.load %in2_mem_cons_buff_0[%3] : memref<256xi32>
            %5 = arith.addi %1, %4 : i32
            memref.store %5, %out_buff_0[%arg2] : memref<16xi32>
          }
          aie.use_lock(%in1_cons_prod_lock, Release, 1)
          aie.use_lock(%out_cons_lock, Release, 1)
          %0 = arith.addi %arg1, %c1 : index
          aie.use_lock(%in1_cons_cons_lock, AcquireGreaterEqual, 1)
          aie.use_lock(%out_prod_lock, AcquireGreaterEqual, 1)
          scf.for %arg2 = %c0 to %c16 step %c2 {
            %1 = memref.load %in1_cons_buff_1[%arg2] : memref<16xi32>
            %2 = arith.muli %0, %c16 : index
            %3 = arith.addi %arg2, %2 : index
            %4 = memref.load %in2_mem_cons_buff_0[%3] : memref<256xi32>
            %5 = arith.addi %1, %4 : i32
            memref.store %5, %out_buff_1[%arg2] : memref<16xi32>
            %6 = arith.addi %arg2, %c1 : index
            %7 = memref.load %in1_cons_buff_1[%6] : memref<16xi32>
            %8 = arith.muli %0, %c16 : index
            %9 = arith.addi %6, %8 : index
            %10 = memref.load %in2_mem_cons_buff_0[%9] : memref<256xi32>
            %11 = arith.addi %7, %10 : i32
            memref.store %11, %out_buff_1[%6] : memref<16xi32>
          }
          aie.use_lock(%in1_cons_prod_lock, Release, 1)
          aie.use_lock(%out_cons_lock, Release, 1)
        }
        aie.use_lock(%in2_mem_cons_prod_lock, Release, 1)
        aie.use_lock(%in2_mem_cons_cons_lock, AcquireGreaterEqual, 1)
        scf.for %arg1 = %c0 to %c16 step %c2 {
          aie.use_lock(%in1_cons_cons_lock, AcquireGreaterEqual, 1)
          aie.use_lock(%out_prod_lock, AcquireGreaterEqual, 1)
          scf.for %arg2 = %c0 to %c16 step %c1 {
            %1 = memref.load %in1_cons_buff_0[%arg2] : memref<16xi32>
            %2 = arith.muli %arg1, %c16 : index
            %3 = arith.addi %arg2, %2 : index
            %4 = memref.load %in2_mem_cons_buff_1[%3] : memref<256xi32>
            %5 = arith.addi %1, %4 : i32
            memref.store %5, %out_buff_0[%arg2] : memref<16xi32>
          }
          aie.use_lock(%in1_cons_prod_lock, Release, 1)
          aie.use_lock(%out_cons_lock, Release, 1)
          %0 = arith.addi %arg1, %c1 : index
          aie.use_lock(%in1_cons_cons_lock, AcquireGreaterEqual, 1)
          aie.use_lock(%out_prod_lock, AcquireGreaterEqual, 1)
          scf.for %arg2 = %c0 to %c16 step %c2 {
            %1 = memref.load %in1_cons_buff_1[%arg2] : memref<16xi32>
            %2 = arith.muli %0, %c16 : index
            %3 = arith.addi %arg2, %2 : index
            %4 = memref.load %in2_mem_cons_buff_1[%3] : memref<256xi32>
            %5 = arith.addi %1, %4 : i32
            memref.store %5, %out_buff_1[%arg2] : memref<16xi32>
            %6 = arith.addi %arg2, %c1 : index
            %7 = memref.load %in1_cons_buff_1[%6] : memref<16xi32>
            %8 = arith.muli %0, %c16 : index
            %9 = arith.addi %6, %8 : index
            %10 = memref.load %in2_mem_cons_buff_1[%9] : memref<256xi32>
            %11 = arith.addi %7, %10 : i32
            memref.store %11, %out_buff_1[%6] : memref<16xi32>
          }
          aie.use_lock(%in1_cons_prod_lock, Release, 1)
          aie.use_lock(%out_cons_lock, Release, 1)
        }
        aie.use_lock(%in2_mem_cons_prod_lock, Release, 1)
      }
      aie.use_lock(%in2_mem_cons_cons_lock, AcquireGreaterEqual, 1)
      scf.for %arg0 = %c0 to %c16 step %c2 {
        aie.use_lock(%in1_cons_cons_lock, AcquireGreaterEqual, 1)
        aie.use_lock(%out_prod_lock, AcquireGreaterEqual, 1)
        scf.for %arg1 = %c0 to %c16 step %c1 {
          %1 = memref.load %in1_cons_buff_0[%arg1] : memref<16xi32>
          %2 = arith.muli %arg0, %c16 : index
          %3 = arith.addi %arg1, %2 : index
          %4 = memref.load %in2_mem_cons_buff_0[%3] : memref<256xi32>
          %5 = arith.addi %1, %4 : i32
          memref.store %5, %out_buff_0[%arg1] : memref<16xi32>
        }
        aie.use_lock(%in1_cons_prod_lock, Release, 1)
        aie.use_lock(%out_cons_lock, Release, 1)
        %0 = arith.addi %arg0, %c1 : index
        aie.use_lock(%in1_cons_cons_lock, AcquireGreaterEqual, 1)
        aie.use_lock(%out_prod_lock, AcquireGreaterEqual, 1)
        scf.for %arg1 = %c0 to %c16 step %c2 {
          %1 = memref.load %in1_cons_buff_1[%arg1] : memref<16xi32>
          %2 = arith.muli %0, %c16 : index
          %3 = arith.addi %arg1, %2 : index
          %4 = memref.load %in2_mem_cons_buff_0[%3] : memref<256xi32>
          %5 = arith.addi %1, %4 : i32
          memref.store %5, %out_buff_1[%arg1] : memref<16xi32>
          %6 = arith.addi %arg1, %c1 : index
          %7 = memref.load %in1_cons_buff_1[%6] : memref<16xi32>
          %8 = arith.muli %0, %c16 : index
          %9 = arith.addi %6, %8 : index
          %10 = memref.load %in2_mem_cons_buff_0[%9] : memref<256xi32>
          %11 = arith.addi %7, %10 : i32
          memref.store %11, %out_buff_1[%6] : memref<16xi32>
        }
        aie.use_lock(%in1_cons_prod_lock, Release, 1)
        aie.use_lock(%out_cons_lock, Release, 1)
      }
      aie.use_lock(%in2_mem_cons_prod_lock, Release, 1)
      aie.end
    }
    aie.shim_dma_allocation @in1(MM2S, 0, 0)
    aiex.runtime_sequence(%arg0: memref<256xi32>, %arg1: memref<256xi32>, %arg2: memref<256xi32>) {
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 0][1, 1, 1, 256][0, 0, 0, 1]) {id = 1 : i64, metadata = @in1} : memref<256xi32>
      aiex.npu.dma_memcpy_nd(0, 0, %arg2[0, 0, 0, 0][1, 1, 1, 256][0, 0, 0, 1]) {id = 0 : i64, metadata = @out} : memref<256xi32>
      aiex.npu.dma_wait {symbol = @out}
    }
    %mem_0_2 = aie.mem(%tile_0_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%in1_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%in1_cons_buff_0 : memref<16xi32>, 0, 16)
      aie.use_lock(%in1_cons_cons_lock, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%in1_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%in1_cons_buff_1 : memref<16xi32>, 0, 16)
      aie.use_lock(%in1_cons_cons_lock, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%in2_mem_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%in2_mem_cons_buff_0 : memref<256xi32>, 0, 256)
      aie.use_lock(%in2_mem_cons_cons_lock, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%in2_mem_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%in2_mem_cons_buff_1 : memref<256xi32>, 0, 256)
      aie.use_lock(%in2_mem_cons_cons_lock, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb7, ^bb9)
    ^bb7:  // 2 preds: ^bb6, ^bb8
      aie.use_lock(%out_cons_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%out_buff_0 : memref<16xi32>, 0, 16)
      aie.use_lock(%out_prod_lock, Release, 1)
      aie.next_bd ^bb8
    ^bb8:  // pred: ^bb7
      aie.use_lock(%out_cons_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%out_buff_1 : memref<16xi32>, 0, 16)
      aie.use_lock(%out_prod_lock, Release, 1)
      aie.next_bd ^bb7
    ^bb9:  // pred: ^bb6
      aie.end
    }
    aie.shim_dma_allocation @out(S2MM, 0, 0)
    %memtile_dma_0_1 = aie.memtile_dma(%tile_0_1) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:
      aie.use_lock(%in2_mem_cons_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%in2_mem_buff_0 : memref<256xi32>, 0, 256)
      aie.use_lock(%in2_mem_prod_lock, Release, 1)
      aie.next_bd ^bb2
    ^bb2:
      aie.use_lock(%in2_mem_cons_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%in2_mem_buff_1 : memref<256xi32>, 0, 256)
      aie.use_lock(%in2_mem_prod_lock, Release, 1)
      aie.next_bd ^bb1
    ^bb3:
      aie.end
    }
  }
}

