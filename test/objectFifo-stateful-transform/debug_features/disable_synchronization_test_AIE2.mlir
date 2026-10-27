//===- disable_synchronization_test_AIE2.mlir -------------------*- MLIR -*-===//
//
// Copyright (C) 2024 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform="skip-verify=true" --aie-objectFifo-unroll %s | FileCheck %s

// CHECK: module @disable_sync {
// CHECK:   aie.device(xcve2302) {
// CHECK-DAG:     %{{.*}}tile_1_2 = aie.tile(1, 2)
// CHECK-DAG:     %{{.*}}tile_3_3 = aie.tile(3, 3)
// CHECK-DAG:     aie.buffer @[[VAL_0:of1_cons_buff_0]](%{{.*}}tile_3_3) : memref<16xi32>
// CHECK-DAG:     aie.buffer @[[VAL_1:of1_cons_buff_1]](%{{.*}}tile_3_3) : memref<16xi32>
// CHECK-DAG:     aie.buffer @[[VAL_2:of1_buff_0]](%{{.*}}tile_1_2) : memref<16xi32>
// CHECK-DAG:     aie.buffer @[[VAL_3:of1_buff_1]](%{{.*}}tile_1_2) : memref<16xi32>
// CHECK-DAG:     aie.buffer @[[VAL_4:of0_buff_0]](%{{.*}}tile_1_2) : memref<16xi32>
// CHECK-DAG:     aie.flow(%{{.*}}tile_1_2, DMA : 0, %{{.*}}tile_3_3, DMA : 0)
// CHECK:     %mem_1_2 = aie.mem(%{{.*}}tile_1_2) {
// CHECK:       %[[CHECK_SSA_0:.*]] = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
// CHECK:     ^bb1:  // 2 preds: ^bb0, ^bb2
// CHECK:       %[[VAL_2_ACCESS_0:.*]] = aie.get_buffer @[[VAL_2]] : memref<16xi32>
// CHECK:       aie.dma_bd(%[[VAL_2_ACCESS_0]] : memref<16xi32> offset = {{.*}} len = {{.*}})
// CHECK:       aie.next_bd ^bb2
// CHECK:     ^bb2:  // pred: ^bb1
// CHECK:       %[[VAL_3_ACCESS_1:.*]] = aie.get_buffer @[[VAL_3]] : memref<16xi32>
// CHECK:       aie.dma_bd(%[[VAL_3_ACCESS_1]] : memref<16xi32> offset = {{.*}} len = {{.*}})
// CHECK:       aie.next_bd ^bb1
// CHECK:     ^bb3:  // pred: ^bb0
// CHECK:       aie.end
// CHECK:     }
// CHECK:     %mem_3_3 = aie.mem(%{{.*}}tile_3_3) {
// CHECK:       %[[CHECK_SSA_1:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:     ^bb1:  // 2 preds: ^bb0, ^bb2
// CHECK:       %[[VAL_0_ACCESS_2:.*]] = aie.get_buffer @[[VAL_0]] : memref<16xi32>
// CHECK:       aie.dma_bd(%[[VAL_0_ACCESS_2]] : memref<16xi32> offset = {{.*}} len = {{.*}})
// CHECK:       aie.next_bd ^bb2
// CHECK:     ^bb2:  // pred: ^bb1
// CHECK:       %[[VAL_1_ACCESS_3:.*]] = aie.get_buffer @[[VAL_1]] : memref<16xi32>
// CHECK:       aie.dma_bd(%[[VAL_1_ACCESS_3]] : memref<16xi32> offset = {{.*}} len = {{.*}})
// CHECK:       aie.next_bd ^bb1
// CHECK:     ^bb3:  // pred: ^bb0
// CHECK:       aie.end
// CHECK:     }
// CHECK:   }
// CHECK: }

module @disable_sync {
 aie.device(xcve2302) {
    %tile12 = aie.tile(1, 2)
    %tile13 = aie.tile(1, 3)
    %tile33 = aie.tile(3, 3)

    aie.objectfifo @of0 (%tile12, {%tile13}, 1 : i32) { disable_synchronization = true } : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @of1 (%tile12, {%tile33}, 2 : i32) { disable_synchronization = true } : !aie.objectfifo<memref<16xi32>>
 }
}
