// //===- aie.mlir ------------------------------------------------*- MLIR -*-===//
// //
// // This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// // See https://llvm.org/LICENSE.txt for license information.
// // SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// //
// // (c) Copyright 2024 Advanced Micro Devices, Inc. or its affiliates
// //
// //===----------------------------------------------------------------------===//

module {
  aie.device(npu1_1col) {
    memref.global "public" @out_cons : memref<1024xi32>
    memref.global "public" @out : memref<1024xi32>
    memref.global "public" @out3_cons : memref<512xi32>
    memref.global "public" @out3 : memref<512xi32>
    memref.global "public" @out2_cons : memref<512xi32>
    memref.global "public" @out2 : memref<512xi32>
    memref.global "public" @in3_cons : memref<512xi32>
    memref.global "public" @in3 : memref<512xi32>
    memref.global "public" @in2_cons : memref<512xi32>
    memref.global "public" @in2 : memref<512xi32>
    memref.global "public" @in_cons : memref<1024xi32>
    memref.global "public" @in : memref<1024xi32>
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_1 = aie.tile(0, 1)
    %tile_0_2 = aie.tile(0, 2)
    %tile_0_3 = aie.tile(0, 3)
    %out_cons_prod_lock = aie.lock(%tile_0_0, 2) {init = 0 : i32, sym_name = "out_cons_prod_lock"}
    %out_cons_cons_lock = aie.lock(%tile_0_0, 3) {init = 0 : i32, sym_name = "out_cons_cons_lock"}
    %out_buff_0 = aie.buffer(%tile_0_1) {sym_name = "out_buff_0"} : memref<4096xi32> 
    %out_prod_lock = aie.lock(%tile_0_1, 2) {init = 2 : i32, sym_name = "out_prod_lock"}
    %out_cons_lock = aie.lock(%tile_0_1, 3) {init = 0 : i32, sym_name = "out_cons_lock"}
    %out3_buff_0 = aie.buffer(%tile_0_3) {sym_name = "out3_buff_0"} : memref<512xi32> 
    %out3_buff_1 = aie.buffer(%tile_0_3) {sym_name = "out3_buff_1"} : memref<512xi32> 
    %out3_prod_lock = aie.lock(%tile_0_3, 2) {init = 2 : i32, sym_name = "out3_prod_lock"}
    %out3_cons_lock = aie.lock(%tile_0_3, 3) {init = 0 : i32, sym_name = "out3_cons_lock"}
    %out2_buff_0 = aie.buffer(%tile_0_2) {sym_name = "out2_buff_0"} : memref<512xi32> 
    %out2_buff_1 = aie.buffer(%tile_0_2) {sym_name = "out2_buff_1"} : memref<512xi32> 
    %out2_prod_lock = aie.lock(%tile_0_2, 2) {init = 2 : i32, sym_name = "out2_prod_lock"}
    %out2_cons_lock = aie.lock(%tile_0_2, 3) {init = 0 : i32, sym_name = "out2_cons_lock"}
    %in3_cons_buff_0 = aie.buffer(%tile_0_3) {sym_name = "in3_cons_buff_0"} : memref<512xi32> 
    %in3_cons_buff_1 = aie.buffer(%tile_0_3) {sym_name = "in3_cons_buff_1"} : memref<512xi32> 
    %in3_cons_prod_lock = aie.lock(%tile_0_3, 0) {init = 2 : i32, sym_name = "in3_cons_prod_lock"}
    %in3_cons_cons_lock = aie.lock(%tile_0_3, 1) {init = 0 : i32, sym_name = "in3_cons_cons_lock"}
    %in2_cons_buff_0 = aie.buffer(%tile_0_2) {sym_name = "in2_cons_buff_0"} : memref<512xi32> 
    %in2_cons_buff_1 = aie.buffer(%tile_0_2) {sym_name = "in2_cons_buff_1"} : memref<512xi32> 
    %in2_cons_prod_lock = aie.lock(%tile_0_2, 0) {init = 2 : i32, sym_name = "in2_cons_prod_lock"}
    %in2_cons_cons_lock = aie.lock(%tile_0_2, 1) {init = 0 : i32, sym_name = "in2_cons_cons_lock"}
    %in_cons_buff_0 = aie.buffer(%tile_0_1) {sym_name = "in_cons_buff_0"} : memref<1024xi32> 
    %in_cons_prod_lock = aie.lock(%tile_0_1, 0) {init = 8 : i32, sym_name = "in_cons_prod_lock"}
    %in_cons_cons_lock = aie.lock(%tile_0_1, 1) {init = 0 : i32, sym_name = "in_cons_cons_lock"}
    %in_prod_lock = aie.lock(%tile_0_0, 0) {init = 0 : i32, sym_name = "in_prod_lock"}
    %in_cons_lock = aie.lock(%tile_0_0, 1) {init = 0 : i32, sym_name = "in_cons_lock"}
    aie.flow(%tile_0_0, DMA : 0, %tile_0_1, DMA : 0)
    aie.flow(%tile_0_1, DMA : 0, %tile_0_2, DMA : 0)
    aie.flow(%tile_0_1, DMA : 1, %tile_0_3, DMA : 0)
    aie.flow(%tile_0_2, DMA : 0, %tile_0_1, DMA : 1)
    aie.flow(%tile_0_3, DMA : 0, %tile_0_1, DMA : 2)
    aie.flow(%tile_0_1, DMA : 2, %tile_0_0, DMA : 0)
    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      %c9223372036854775806 = arith.constant 9223372036854775806 : index
      %c2 = arith.constant 2 : index
      scf.for %arg0 = %c0 to %c9223372036854775806 step %c2 {
        aie.use_lock(%out2_prod_lock, AcquireGreaterEqual, 1)
        aie.use_lock(%in2_cons_cons_lock, AcquireGreaterEqual, 1)
        %c0_2 = arith.constant 0 : index
        %c512_3 = arith.constant 512 : index
        %c1_4 = arith.constant 1 : index
        scf.for %arg1 = %c0_2 to %c512_3 step %c1_4 {
          %0 = memref.load %in2_cons_buff_0[%arg1] : memref<512xi32>
          %c1_i32 = arith.constant 1 : i32
          %1 = arith.addi %0, %c1_i32 : i32
          memref.store %1, %out2_buff_0[%arg1] : memref<512xi32>
        }
        aie.use_lock(%in2_cons_prod_lock, Release, 1)
        aie.use_lock(%out2_cons_lock, Release, 1)
        aie.use_lock(%out2_prod_lock, AcquireGreaterEqual, 1)
        aie.use_lock(%in2_cons_cons_lock, AcquireGreaterEqual, 1)
        %c0_5 = arith.constant 0 : index
        %c512_6 = arith.constant 512 : index
        %c1_7 = arith.constant 1 : index
        scf.for %arg1 = %c0_5 to %c512_6 step %c1_7 {
          %0 = memref.load %in2_cons_buff_1[%arg1] : memref<512xi32>
          %c1_i32 = arith.constant 1 : i32
          %1 = arith.addi %0, %c1_i32 : i32
          memref.store %1, %out2_buff_1[%arg1] : memref<512xi32>
        }
        aie.use_lock(%in2_cons_prod_lock, Release, 1)
        aie.use_lock(%out2_cons_lock, Release, 1)
      }
      aie.use_lock(%out2_prod_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%in2_cons_cons_lock, AcquireGreaterEqual, 1)
      %c0_0 = arith.constant 0 : index
      %c512 = arith.constant 512 : index
      %c1_1 = arith.constant 1 : index
      scf.for %arg0 = %c0_0 to %c512 step %c1_1 {
        %0 = memref.load %in2_cons_buff_0[%arg0] : memref<512xi32>
        %c1_i32 = arith.constant 1 : i32
        %1 = arith.addi %0, %c1_i32 : i32
        memref.store %1, %out2_buff_0[%arg0] : memref<512xi32>
      }
      aie.use_lock(%in2_cons_prod_lock, Release, 1)
      aie.use_lock(%out2_cons_lock, Release, 1)
      aie.end
    }
    %core_0_3 = aie.core(%tile_0_3) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      %c9223372036854775806 = arith.constant 9223372036854775806 : index
      %c2 = arith.constant 2 : index
      scf.for %arg0 = %c0 to %c9223372036854775806 step %c2 {
        aie.use_lock(%out3_prod_lock, AcquireGreaterEqual, 1)
        aie.use_lock(%in3_cons_cons_lock, AcquireGreaterEqual, 1)
        %c0_2 = arith.constant 0 : index
        %c512_3 = arith.constant 512 : index
        %c1_4 = arith.constant 1 : index
        scf.for %arg1 = %c0_2 to %c512_3 step %c1_4 {
          %0 = memref.load %in3_cons_buff_0[%arg1] : memref<512xi32>
          %c2_i32 = arith.constant 2 : i32
          %1 = arith.addi %0, %c2_i32 : i32
          memref.store %1, %out3_buff_0[%arg1] : memref<512xi32>
        }
        aie.use_lock(%in3_cons_prod_lock, Release, 1)
        aie.use_lock(%out3_cons_lock, Release, 1)
        aie.use_lock(%out3_prod_lock, AcquireGreaterEqual, 1)
        aie.use_lock(%in3_cons_cons_lock, AcquireGreaterEqual, 1)
        %c0_5 = arith.constant 0 : index
        %c512_6 = arith.constant 512 : index
        %c1_7 = arith.constant 1 : index
        scf.for %arg1 = %c0_5 to %c512_6 step %c1_7 {
          %0 = memref.load %in3_cons_buff_1[%arg1] : memref<512xi32>
          %c2_i32 = arith.constant 2 : i32
          %1 = arith.addi %0, %c2_i32 : i32
          memref.store %1, %out3_buff_1[%arg1] : memref<512xi32>
        }
        aie.use_lock(%in3_cons_prod_lock, Release, 1)
        aie.use_lock(%out3_cons_lock, Release, 1)
      }
      aie.use_lock(%out3_prod_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%in3_cons_cons_lock, AcquireGreaterEqual, 1)
      %c0_0 = arith.constant 0 : index
      %c512 = arith.constant 512 : index
      %c1_1 = arith.constant 1 : index
      scf.for %arg0 = %c0_0 to %c512 step %c1_1 {
        %0 = memref.load %in3_cons_buff_0[%arg0] : memref<512xi32>
        %c2_i32 = arith.constant 2 : i32
        %1 = arith.addi %0, %c2_i32 : i32
        memref.store %1, %out3_buff_0[%arg0] : memref<512xi32>
      }
      aie.use_lock(%in3_cons_prod_lock, Release, 1)
      aie.use_lock(%out3_cons_lock, Release, 1)
      aie.end
    }
    aie.shim_dma_allocation @in(MM2S, 0, 0)
    func.func @sequence(%arg0: memref<1024xi32>, %arg1: memref<4096xi32>, %arg2: memref<4096xi32>) {
      aiex.npu.dma_memcpy_nd(0, 0, %arg2[0, 0, 0, 0][1, 1, 1, 4096][0, 0, 0]) {id = 0 : i64, metadata = @out} : memref<4096xi32>
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 0][1, 1, 1, 1024][0, 0, 0]) {id = 1 : i64, metadata = @in} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}
      return
    }
    %memtile_dma_0_1 = aie.memtile_dma(%tile_0_1) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%in_cons_prod_lock, AcquireGreaterEqual, 8)
      aie.dma_bd(%in_cons_buff_0 : memref<1024xi32>, 0, 1024)
      aie.use_lock(%in_cons_cons_lock, Release, 8)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb3, ^bb4, repeat_count = 1)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      aie.use_lock(%in_cons_cons_lock, AcquireGreaterEqual, 7)
      aie.dma_bd(%in_cons_buff_0 : memref<1024xi32>, 0, 512)
      aie.use_lock(%in_cons_prod_lock, Release, 1)
      aie.next_bd ^bb3
    ^bb4:  // pred: ^bb2
      %2 = aie.dma_start(MM2S, 1, ^bb5, ^bb6, repeat_count = 5)
    ^bb5:  // 2 preds: ^bb4, ^bb5
      aie.use_lock(%in_cons_cons_lock, AcquireGreaterEqual, 6)
      aie.dma_bd(%in_cons_buff_0 : memref<1024xi32>, 512, 512)
      aie.use_lock(%in_cons_prod_lock, Release, 1)
      aie.next_bd ^bb5
    ^bb6:  // pred: ^bb4
      %3 = aie.dma_start(S2MM, 1, ^bb7, ^bb8)
    ^bb7:  // 2 preds: ^bb6, ^bb7
      aie.use_lock(%out_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%out_buff_0 : memref<4096xi32>, 0, 1024)
      aie.use_lock(%out_cons_lock, Release, 1)
      aie.next_bd ^bb7
    ^bb8:  // pred: ^bb6
      %4 = aie.dma_start(S2MM, 2, ^bb9, ^bb10)
    ^bb9:  // 2 preds: ^bb8, ^bb9
      aie.use_lock(%out_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%out_buff_0 : memref<4096xi32>, 1024, 2048)
      aie.use_lock(%out_cons_lock, Release, 1)
      aie.next_bd ^bb9
    ^bb10:  // pred: ^bb8
      %5 = aie.dma_start(MM2S, 2, ^bb11, ^bb12)
    ^bb11:  // 2 preds: ^bb10, ^bb11
      aie.use_lock(%out_cons_lock, AcquireGreaterEqual, 2)
      aie.dma_bd(%out_buff_0 : memref<4096xi32>, 0, 4096)
      aie.use_lock(%out_prod_lock, Release, 2)
      aie.next_bd ^bb11
    ^bb12:  // pred: ^bb10
      aie.end
    }
    %mem_0_2 = aie.mem(%tile_0_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%in2_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%in2_cons_buff_0 : memref<512xi32>, 0, 512)
      aie.use_lock(%in2_cons_cons_lock, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%in2_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%in2_cons_buff_1 : memref<512xi32>, 0, 512)
      aie.use_lock(%in2_cons_cons_lock, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%out2_cons_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%out2_buff_0 : memref<512xi32>, 0, 512)
      aie.use_lock(%out2_prod_lock, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%out2_cons_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%out2_buff_1 : memref<512xi32>, 0, 512)
      aie.use_lock(%out2_prod_lock, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      aie.end
    }
    aie.shim_dma_allocation @out(S2MM, 0, 0)
    %mem_0_3 = aie.mem(%tile_0_3) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%in3_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%in3_cons_buff_0 : memref<512xi32>, 0, 512)
      aie.use_lock(%in3_cons_cons_lock, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%in3_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%in3_cons_buff_1 : memref<512xi32>, 0, 512)
      aie.use_lock(%in3_cons_cons_lock, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%out3_cons_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%out3_buff_0 : memref<512xi32>, 0, 512)
      aie.use_lock(%out3_prod_lock, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%out3_cons_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%out3_buff_1 : memref<512xi32>, 0, 512)
      aie.use_lock(%out3_prod_lock, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      aie.end
    }
  }
}

// module {
//   aie.device(npu1_1col) {
//     memref.global "public" @out_cons : memref<1024xi32>
//     memref.global "public" @out : memref<1024xi32>
//     memref.global "public" @out3_cons : memref<512xi32>
//     memref.global "public" @out3 : memref<512xi32>
//     memref.global "public" @out2_cons : memref<512xi32>
//     memref.global "public" @out2 : memref<512xi32>
//     memref.global "public" @in3_cons : memref<512xi32>
//     memref.global "public" @in3 : memref<512xi32>
//     memref.global "public" @in2_cons : memref<512xi32>
//     memref.global "public" @in2 : memref<512xi32>
//     memref.global "public" @in_cons : memref<1024xi32>
//     memref.global "public" @in : memref<1024xi32>
//     %tile_0_0 = aie.tile(0, 0)
//     %tile_0_1 = aie.tile(0, 1)
//     %tile_0_2 = aie.tile(0, 2)
//     %tile_0_3 = aie.tile(0, 3)
//     %out_cons_prod_lock = aie.lock(%tile_0_0, 2) {init = 0 : i32, sym_name = "out_cons_prod_lock"}
//     %out_cons_cons_lock = aie.lock(%tile_0_0, 3) {init = 0 : i32, sym_name = "out_cons_cons_lock"}
//     %out_buff_0 = aie.buffer(%tile_0_1) {sym_name = "out_buff_0"} : memref<1024xi32> 
//     %out_prod_lock = aie.lock(%tile_0_1, 2) {init = 2 : i32, sym_name = "out_prod_lock"}
//     %out_cons_lock = aie.lock(%tile_0_1, 3) {init = 0 : i32, sym_name = "out_cons_lock"}
//     %out3_buff_0 = aie.buffer(%tile_0_3) {sym_name = "out3_buff_0"} : memref<512xi32> 
//     %out3_prod_lock = aie.lock(%tile_0_3, 2) {init = 1 : i32, sym_name = "out3_prod_lock"}
//     %out3_cons_lock = aie.lock(%tile_0_3, 3) {init = 0 : i32, sym_name = "out3_cons_lock"}
//     %out2_buff_0 = aie.buffer(%tile_0_2) {sym_name = "out2_buff_0"} : memref<512xi32> 
//     %out2_prod_lock = aie.lock(%tile_0_2, 2) {init = 1 : i32, sym_name = "out2_prod_lock"}
//     %out2_cons_lock = aie.lock(%tile_0_2, 3) {init = 0 : i32, sym_name = "out2_cons_lock"}
//     %in3_cons_buff_0 = aie.buffer(%tile_0_3) {sym_name = "in3_cons_buff_0"} : memref<512xi32> 
//     %in3_cons_prod_lock = aie.lock(%tile_0_3, 0) {init = 1 : i32, sym_name = "in3_cons_prod_lock"}
//     %in3_cons_cons_lock = aie.lock(%tile_0_3, 1) {init = 0 : i32, sym_name = "in3_cons_cons_lock"}
//     %in2_cons_buff_0 = aie.buffer(%tile_0_2) {sym_name = "in2_cons_buff_0"} : memref<512xi32> 
//     %in2_cons_prod_lock = aie.lock(%tile_0_2, 0) {init = 1 : i32, sym_name = "in2_cons_prod_lock"}
//     %in2_cons_cons_lock = aie.lock(%tile_0_2, 1) {init = 0 : i32, sym_name = "in2_cons_cons_lock"}
//     %in_cons_buff_0 = aie.buffer(%tile_0_1) {sym_name = "in_cons_buff_0"} : memref<1024xi32> 
//     %in_cons_prod_lock = aie.lock(%tile_0_1, 0) {init = 8 : i32, sym_name = "in_cons_prod_lock"}
//     %in_cons_cons_lock = aie.lock(%tile_0_1, 1) {init = 0 : i32, sym_name = "in_cons_cons_lock"}
//     %in_prod_lock = aie.lock(%tile_0_0, 0) {init = 0 : i32, sym_name = "in_prod_lock"}
//     %in_cons_lock = aie.lock(%tile_0_0, 1) {init = 0 : i32, sym_name = "in_cons_lock"}
//     aie.flow(%tile_0_0, DMA : 0, %tile_0_1, DMA : 0)
//     aie.flow(%tile_0_1, DMA : 0, %tile_0_2, DMA : 0)
//     aie.flow(%tile_0_1, DMA : 1, %tile_0_3, DMA : 0)
//     aie.flow(%tile_0_2, DMA : 0, %tile_0_1, DMA : 1)
//     aie.flow(%tile_0_3, DMA : 0, %tile_0_1, DMA : 2)
//     aie.flow(%tile_0_1, DMA : 2, %tile_0_0, DMA : 0)
//     %core_0_2 = aie.core(%tile_0_2) {
//       %c0 = arith.constant 0 : index
//       %c9223372036854775807 = arith.constant 9223372036854775807 : index
//       %c1 = arith.constant 1 : index
//       %c1_0 = arith.constant 1 : index
//       scf.for %arg0 = %c0 to %c9223372036854775807 step %c1_0 {
//         aie.use_lock(%out2_prod_lock, AcquireGreaterEqual, 1)
//         aie.use_lock(%in2_cons_cons_lock, AcquireGreaterEqual, 1)
//         %c0_1 = arith.constant 0 : index
//         %c512 = arith.constant 512 : index
//         %c1_2 = arith.constant 1 : index
//         scf.for %arg1 = %c0_1 to %c512 step %c1_2 {
//           %0 = memref.load %in2_cons_buff_0[%arg1] : memref<512xi32>
//           %c1_i32 = arith.constant 1 : i32
//           %1 = arith.addi %0, %c1_i32 : i32
//           memref.store %1, %out2_buff_0[%arg1] : memref<512xi32>
//         }
//         aie.use_lock(%in2_cons_prod_lock, Release, 1)
//         aie.use_lock(%out2_cons_lock, Release, 1)
//       }
//       aie.end
//     }
//     %core_0_3 = aie.core(%tile_0_3) {
//       %c0 = arith.constant 0 : index
//       %c9223372036854775807 = arith.constant 9223372036854775807 : index
//       %c1 = arith.constant 1 : index
//       %c1_0 = arith.constant 1 : index
//       scf.for %arg0 = %c0 to %c9223372036854775807 step %c1_0 {
//         aie.use_lock(%out3_prod_lock, AcquireGreaterEqual, 1)
//         aie.use_lock(%in3_cons_cons_lock, AcquireGreaterEqual, 1)
//         %c0_1 = arith.constant 0 : index
//         %c512 = arith.constant 512 : index
//         %c1_2 = arith.constant 1 : index
//         scf.for %arg1 = %c0_1 to %c512 step %c1_2 {
//           %0 = memref.load %in3_cons_buff_0[%arg1] : memref<512xi32>
//           %c2_i32 = arith.constant 2 : i32
//           %1 = arith.addi %0, %c2_i32 : i32
//           memref.store %1, %out3_buff_0[%arg1] : memref<512xi32>
//         }
//         aie.use_lock(%in3_cons_prod_lock, Release, 1)
//         aie.use_lock(%out3_cons_lock, Release, 1)
//       }
//       aie.end
//     }
//     aie.shim_dma_allocation @in(MM2S, 0, 0)
//     func.func @sequence(%arg0: memref<1024xi32>, %arg1: memref<4096xi32>, %arg2: memref<4096xi32>) {
//       aiex.npu.dma_memcpy_nd(0, 0, %arg2[0, 0, 0, 0][1, 1, 1, 4096][0, 0, 0]) {id = 0 : i64, metadata = @out} : memref<4096xi32>
//       aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 0][1, 1, 1, 1024][0, 0, 0]) {id = 1 : i64, metadata = @in} : memref<1024xi32>
//       aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}
//       return
//     }
//     %memtile_dma_0_1 = aie.memtile_dma(%tile_0_1) {
//       %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
//     ^bb1:  // 2 preds: ^bb0, ^bb1
//       aie.use_lock(%in_cons_prod_lock, AcquireGreaterEqual, 8)
//       aie.dma_bd(%in_cons_buff_0 : memref<1024xi32>, 0, 1024)
//       aie.use_lock(%in_cons_cons_lock, Release, 8)
//       aie.next_bd ^bb1
//     ^bb2:  // pred: ^bb0
//       %1 = aie.dma_start(MM2S, 0, ^bb3, ^bb4, repeat_count = 1)
//     ^bb3:  // 2 preds: ^bb2, ^bb3
//       aie.use_lock(%in_cons_cons_lock, AcquireGreaterEqual, 5)
//       aie.dma_bd(%in_cons_buff_0 : memref<1024xi32>, 0, 512)
//       aie.use_lock(%in_cons_prod_lock, Release, 1)
//       aie.next_bd ^bb3
//     ^bb4:  // pred: ^bb2
//       %2 = aie.dma_start(MM2S, 1, ^bb5, ^bb6, repeat_count = 5)
//     ^bb5:  // 2 preds: ^bb4, ^bb5
//       aie.use_lock(%in_cons_cons_lock, AcquireGreaterEqual, 1)
//       aie.dma_bd(%in_cons_buff_0 : memref<1024xi32>, 512, 512)
//       aie.use_lock(%in_cons_prod_lock, Release, 1)
//       aie.next_bd ^bb5
//     ^bb6:  // pred: ^bb4
//       %3 = aie.dma_start(S2MM, 1, ^bb7, ^bb8)
//     ^bb7:  // 2 preds: ^bb6, ^bb7
//       aie.use_lock(%out_prod_lock, AcquireGreaterEqual, 1)
//       aie.dma_bd(%out_buff_0 : memref<1024xi32>, 0, 1024)
//       aie.use_lock(%out_cons_lock, Release, 1)
//       aie.next_bd ^bb7
//     ^bb8:  // pred: ^bb6
//       %4 = aie.dma_start(S2MM, 2, ^bb9, ^bb10)
//     ^bb9:  // 2 preds: ^bb8, ^bb9
//       aie.use_lock(%out_prod_lock, AcquireGreaterEqual, 1)
//       aie.dma_bd(%out_buff_0 : memref<1024xi32>, 1024, 3072)
//       aie.use_lock(%out_cons_lock, Release, 1)
//       aie.next_bd ^bb9
//     ^bb10:  // pred: ^bb8
//       %5 = aie.dma_start(MM2S, 2, ^bb11, ^bb12)
//     ^bb11:  // 2 preds: ^bb10, ^bb11
//       aie.use_lock(%out_cons_lock, AcquireGreaterEqual, 2)
//       aie.dma_bd(%out_buff_0 : memref<1024xi32>, 0, 1024)
//       aie.use_lock(%out_prod_lock, Release, 2)
//       aie.next_bd ^bb11
//     ^bb12:  // pred: ^bb10
//       aie.end
//     }
//     %mem_0_2 = aie.mem(%tile_0_2) {
//       %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
//     ^bb1:  // 2 preds: ^bb0, ^bb1
//       aie.use_lock(%in2_cons_prod_lock, AcquireGreaterEqual, 1)
//       aie.dma_bd(%in2_cons_buff_0 : memref<512xi32>, 0, 512)
//       aie.use_lock(%in2_cons_cons_lock, Release, 1)
//       aie.next_bd ^bb1
//     ^bb2:  // pred: ^bb0
//       %1 = aie.dma_start(MM2S, 0, ^bb3, ^bb4)
//     ^bb3:  // 2 preds: ^bb2, ^bb3
//       aie.use_lock(%out2_cons_lock, AcquireGreaterEqual, 1)
//       aie.dma_bd(%out2_buff_0 : memref<512xi32>, 0, 512)
//       aie.use_lock(%out2_prod_lock, Release, 1)
//       aie.next_bd ^bb3
//     ^bb4:  // pred: ^bb2
//       aie.end
//     }
//     aie.shim_dma_allocation @out(S2MM, 0, 0)
//     %mem_0_3 = aie.mem(%tile_0_3) {
//       %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
//     ^bb1:  // 2 preds: ^bb0, ^bb1
//       aie.use_lock(%in3_cons_prod_lock, AcquireGreaterEqual, 1)
//       aie.dma_bd(%in3_cons_buff_0 : memref<512xi32>, 0, 512)
//       aie.use_lock(%in3_cons_cons_lock, Release, 1)
//       aie.next_bd ^bb1
//     ^bb2:  // pred: ^bb0
//       %1 = aie.dma_start(MM2S, 0, ^bb3, ^bb4)
//     ^bb3:  // 2 preds: ^bb2, ^bb3
//       aie.use_lock(%out3_cons_lock, AcquireGreaterEqual, 1)
//       aie.dma_bd(%out3_buff_0 : memref<512xi32>, 0, 512)
//       aie.use_lock(%out3_prod_lock, Release, 1)
//       aie.next_bd ^bb3
//     ^bb4:  // pred: ^bb2
//       aie.end
//     }
//   }
// }
