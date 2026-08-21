//===- lock_analysis_test.mlir ----------------------------------*- MLIR -*-===//
//
// Copyright (C) 2024 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform --aie-objectFifo-unroll %s | FileCheck %s

// CHECK: module @lockAnalysis {
// CHECK:   aie.device(xcve2302) {
// CHECK-DAG:     %{{.*}}tile_1_2 = aie.tile(1, 2)
// CHECK-DAG:     %{{.*}}tile_3_3 = aie.tile(3, 3)
// CHECK-DAG:     %[[VAL_0:.*]] = aie.buffer(%{{.*}}tile_3_3) {sym_name = "of1_cons_buff_0"} : memref<16xi32>
// CHECK-DAG:     %[[VAL_1:.*]] = aie.buffer(%{{.*}}tile_3_3) {sym_name = "of1_cons_buff_1"} : memref<16xi32>
// CHECK-DAG:     %[[VAL_2:.*]] = aie.lock(%{{.*}}tile_3_3) {init = 2 : i32, sym_name = "of1_cons_prod_lock_0"}
// CHECK-DAG:     %[[VAL_3:.*]] = aie.lock(%{{.*}}tile_3_3) {init = 0 : i32, sym_name = "of1_cons_cons_lock_0"}
// CHECK-DAG:     %[[VAL_4:.*]] = aie.buffer(%{{.*}}tile_1_2) {sym_name = "of1_buff_0"} : memref<16xi32>
// CHECK-DAG:     %[[VAL_5:.*]] = aie.buffer(%{{.*}}tile_1_2) {sym_name = "of1_buff_1"} : memref<16xi32>
// CHECK-DAG:     %[[VAL_6:.*]] = aie.lock(%{{.*}}tile_1_2) {init = 2 : i32, sym_name = "of1_prod_lock_0"}
// CHECK-DAG:     %[[VAL_7:.*]] = aie.lock(%{{.*}}tile_1_2) {init = 0 : i32, sym_name = "of1_cons_lock_0"}
// CHECK-DAG:     %test_buff = aie.buffer(%{{.*}}tile_1_2) {sym_name = "test_buff"} : memref<16xi32>
// CHECK-DAG:     aie.flow(%{{.*}}tile_1_2, DMA : 0, %{{.*}}tile_3_3, DMA : 0)
// CHECK:     %mem_1_2 = aie.mem(%{{.*}}tile_1_2) {
// CHECK-DAG:       %test_prod_lock = aie.lock(%{{.*}}tile_1_2, 0) {init = 1 : i32, sym_name = "test_prod_lock"}
// CHECK-DAG:       %test_cons_lock = aie.lock(%{{.*}}tile_1_2, 1) {init = 0 : i32, sym_name = "test_cons_lock"}
// CHECK:       %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
// CHECK:     ^bb1:  // 2 preds: ^bb0, ^bb1
// CHECK:       aie.use_lock(%test_prod_lock, AcquireGreaterEqual, %{{.*}})
// CHECK:       aie.dma_bd(%test_buff : memref<16xi32> offset = {{.*}} len = {{.*}})
// CHECK:       aie.use_lock(%test_cons_lock, Release, %{{.*}})
// CHECK:       aie.next_bd ^bb1
// CHECK:     ^bb2:  // pred: ^bb0
// CHECK:       %1 = aie.dma_start(MM2S, 0, ^bb3, ^bb5)
// CHECK:     ^bb3:  // 2 preds: ^bb2, ^bb4
// CHECK:       aie.use_lock(%[[VAL_7]], AcquireGreaterEqual, %{{.*}})
// CHECK:       aie.dma_bd(%[[VAL_4]] : memref<16xi32> offset = {{.*}} len = {{.*}})
// CHECK:       aie.use_lock(%[[VAL_6]], Release, %{{.*}})
// CHECK:       aie.next_bd ^bb4
// CHECK:     ^bb4:  // pred: ^bb3
// CHECK:       aie.use_lock(%[[VAL_7]], AcquireGreaterEqual, %{{.*}})
// CHECK:       aie.dma_bd(%[[VAL_5]] : memref<16xi32> offset = {{.*}} len = {{.*}})
// CHECK:       aie.use_lock(%[[VAL_6]], Release, %{{.*}})
// CHECK:       aie.next_bd ^bb3
// CHECK:     ^bb5:  // pred: ^bb2
// CHECK:       aie.end
// CHECK:     }
// CHECK:     %mem_3_3 = aie.mem(%{{.*}}tile_3_3) {
// CHECK:       %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:     ^bb1:  // 2 preds: ^bb0, ^bb2
// CHECK:       aie.use_lock(%[[VAL_2]], AcquireGreaterEqual, %{{.*}})
// CHECK:       aie.dma_bd(%[[VAL_0]] : memref<16xi32> offset = {{.*}} len = {{.*}})
// CHECK:       aie.use_lock(%[[VAL_3]], Release, %{{.*}})
// CHECK:       aie.next_bd ^bb2
// CHECK:     ^bb2:  // pred: ^bb1
// CHECK:       aie.use_lock(%[[VAL_2]], AcquireGreaterEqual, %{{.*}})
// CHECK:       aie.dma_bd(%[[VAL_1]] : memref<16xi32> offset = {{.*}} len = {{.*}})
// CHECK:       aie.use_lock(%[[VAL_3]], Release, %{{.*}})
// CHECK:       aie.next_bd ^bb1
// CHECK:     ^bb3:  // pred: ^bb0
// CHECK:       aie.end
// CHECK:     }
// CHECK:   }
// CHECK: }

module @lockAnalysis {
   aie.device(xcve2302) {
      %tile12 = aie.tile(1, 2)
      %tile33 = aie.tile(3, 3)

      %test_buff = aie.buffer(%tile12) {sym_name = "test_buff"} : memref<16xi32>

      aie.objectfifo @of1 (%tile12, {%tile33}, 2 : i32) : !aie.objectfifo<memref<16xi32>>

      %mem_1_2 = aie.mem(%tile12) {
         %test_prod_lock = aie.lock(%tile12, 0) {init = 1 : i32, sym_name = "test_prod_lock"}
         %test_cons_lock = aie.lock(%tile12, 1) {init = 0 : i32, sym_name = "test_cons_lock"}
         %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
         ^bb1:
            %c1_ul1 = arith.constant 1 : i32
            aie.use_lock(%test_prod_lock, AcquireGreaterEqual, %c1_ul1)
            aie.dma_bd(%test_buff : memref<16xi32> offset = 0 len = 16)
            %c1_ul2 = arith.constant 1 : i32
            aie.use_lock(%test_cons_lock, Release, %c1_ul2)
            aie.next_bd ^bb1
         ^bb2:
            aie.end
      }
   }
}
