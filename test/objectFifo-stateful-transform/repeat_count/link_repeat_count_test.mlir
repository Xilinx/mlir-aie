//===- link_repeat_count_test.mlir ------------------------------*- MLIR -*-===//
//
// Copyright (C) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform="skip-verify=true" --aie-objectFifo-unroll %s | FileCheck %s

// CHECK: module @memtileRepeat {
// CHECK:   aie.device(npu1) {
// CHECK-DAG: aie.buffer @[[OF2_BUF:of2_buff_0]](%{{.*}}tile_3_3) : memref<32xi32>
// CHECK-DAG: aie.lock @[[OF2_PROD:of2_prod_lock_0]](%{{.*}}tile_3_3) {init = 3 : i32}
// CHECK-DAG: aie.lock @[[OF2_CONS:of2_cons_lock_0]](%{{.*}}tile_3_3) {init = 0 : i32}
// CHECK-DAG: aie.buffer @[[OF1C_BUF:of1_cons_buff_0]](%{{.*}}tile_1_2) : memref<16xi32>
// CHECK-DAG: aie.lock @[[OF1C_PROD:of1_cons_prod_lock_0]](%{{.*}}tile_1_2) {init = 1 : i32}
// CHECK-DAG: aie.lock @[[OF1C_CONS:of1_cons_cons_lock_0]](%{{.*}}tile_1_2) {init = 0 : i32}
// CHECK-DAG: aie.buffer @[[OF2C_BUF:of2_cons_buff_0]](%{{.*}}tile_2_1) : memref<32xi32>
// CHECK-DAG: aie.lock @[[OF2C_PROD:of2_cons_prod_lock_0]](%{{.*}}tile_2_1) {init = 1 : i32}
// CHECK-DAG: aie.lock @[[OF2C_CONS:of2_cons_cons_lock_0]](%{{.*}}tile_2_1) {init = 0 : i32}
// CHECK-DAG: aie.buffer @[[OF0C_BUF:of0_cons_buff_0]](%{{.*}}tile_1_1) : memref<32xi32>
// CHECK-DAG: aie.lock @[[OF0C_PROD:of0_cons_prod_lock_0]](%{{.*}}tile_1_1) {init = 3 : i32}
// CHECK-DAG: aie.lock @[[OF0C_CONS:of0_cons_cons_lock_0]](%{{.*}}tile_1_1) {init = 0 : i32}
// CHECK-DAG:     aie.shim_dma_allocation @of3_shim_alloc(%shim_noc_tile_1_0, S2MM, 0)
// CHECK:     %memtile_dma_1_1 = aie.memtile_dma(%{{.*}}tile_1_1) {
// CHECK:       %[[CHECK_SSA_0:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
// CHECK:     ^bb1:
// CHECK:       aie.use_lock(@[[OF0C_PROD]], AcquireGreaterEqual, %{{.*}})
// CHECK:       %[[OF0C_BUF_ACCESS_0:.*]] = aie.get_buffer @[[OF0C_BUF]] : memref<32xi32>
// CHECK:       aie.dma_bd(%[[OF0C_BUF_ACCESS_0]] : memref<32xi32> offset = {{.*}} len = {{.*}})
// CHECK:       aie.use_lock(@[[OF0C_CONS]], Release, %{{.*}})
// CHECK:       aie.next_bd ^bb1
// CHECK:     ^bb2:
// CHECK:       %[[CHECK_SSA_1:.*]] = aie.dma_start(MM2S, 0, ^bb3, ^bb4, repeat_count = 2)
// CHECK:     ^bb3:
// CHECK:       aie.use_lock(@[[OF0C_CONS]], AcquireGreaterEqual, %{{.*}})
// CHECK:       %[[OF0C_BUF_ACCESS_1:.*]] = aie.get_buffer @[[OF0C_BUF]] : memref<32xi32>
// CHECK:       aie.dma_bd(%[[OF0C_BUF_ACCESS_1]] : memref<32xi32> offset = {{.*}} len = {{.*}})
// CHECK:       aie.use_lock(@[[OF0C_PROD]], Release, %{{.*}})
// CHECK:       aie.next_bd ^bb3
// CHECK:     ^bb4:
// CHECK:       aie.end
// CHECK:     }
// CHECK:     %mem_1_2 = aie.mem(%{{.*}}tile_1_2) {
// CHECK:       %[[CHECK_SSA_2:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
// CHECK:     ^bb1:
// CHECK:       aie.use_lock(@[[OF1C_PROD]], AcquireGreaterEqual, %{{.*}})
// CHECK:       %[[OF1C_BUF_ACCESS_2:.*]] = aie.get_buffer @[[OF1C_BUF]] : memref<16xi32>
// CHECK:       aie.dma_bd(%[[OF1C_BUF_ACCESS_2]] : memref<16xi32> offset = {{.*}} len = {{.*}})
// CHECK:       aie.use_lock(@[[OF1C_CONS]], Release, %{{.*}})
// CHECK:       aie.next_bd ^bb1
// CHECK:     ^bb2:
// CHECK:       aie.end
// CHECK:     }
// CHECK:     %mem_3_3 = aie.mem(%{{.*}}tile_3_3) {
// CHECK:       %[[CHECK_SSA_3:.*]] = aie.dma_start(MM2S, 0, ^bb1, ^bb2, repeat_count = 2)
// CHECK:     ^bb1:
// CHECK:       aie.use_lock(@[[OF2_CONS]], AcquireGreaterEqual, %{{.*}})
// CHECK:       %[[OF2_BUF_ACCESS_3:.*]] = aie.get_buffer @[[OF2_BUF]] : memref<32xi32>
// CHECK:       aie.dma_bd(%[[OF2_BUF_ACCESS_3]] : memref<32xi32> offset = {{.*}} len = {{.*}})
// CHECK:       aie.use_lock(@[[OF2_PROD]], Release, %{{.*}})
// CHECK:       aie.next_bd ^bb1
// CHECK:     ^bb2:
// CHECK:       aie.end
// CHECK:     }
// CHECK:     %memtile_dma_2_1 = aie.memtile_dma(%{{.*}}tile_2_1) {
// CHECK:       %[[CHECK_SSA_4:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
// CHECK:     ^bb1:
// CHECK:       aie.use_lock(@[[OF2C_PROD]], AcquireGreaterEqual, %{{.*}})
// CHECK:       %[[OF2C_BUF_ACCESS_4:.*]] = aie.get_buffer @[[OF2C_BUF]] : memref<32xi32>
// CHECK:       aie.dma_bd(%[[OF2C_BUF_ACCESS_4]] : memref<32xi32> offset = {{.*}} len = {{.*}})
// CHECK:       aie.use_lock(@[[OF2C_CONS]], Release, %{{.*}})
// CHECK:       aie.next_bd ^bb1
// CHECK:     ^bb2:
// CHECK:       %[[CHECK_SSA_5:.*]] = aie.dma_start(MM2S, 0, ^bb3, ^bb4)
// CHECK:     ^bb3:
// CHECK:       aie.use_lock(@[[OF2C_CONS]], AcquireGreaterEqual, %{{.*}})
// CHECK:       %[[OF2C_BUF_ACCESS_5:.*]] = aie.get_buffer @[[OF2C_BUF]] : memref<32xi32>
// CHECK:       aie.dma_bd(%[[OF2C_BUF_ACCESS_5]] : memref<32xi32> offset = {{.*}} len = {{.*}})
// CHECK:       aie.use_lock(@[[OF2C_PROD]], Release, %{{.*}})
// CHECK:       aie.next_bd ^bb3
// CHECK:     ^bb4:
// CHECK:       aie.end
// CHECK:     }
// CHECK:   }
// CHECK: }

module @memtileRepeat {
 aie.device(npu1) {
    %tile10 = aie.tile(1, 0)
    %tile11 = aie.tile(1, 1)
    %tile21 = aie.tile(2, 1)
    %tile12 = aie.tile(1, 2)
    %tile33 = aie.tile(3, 3)

    aie.objectfifo @of0 (%tile10, {%tile11}, 1 : i32) : !aie.objectfifo<memref<32xi32>>
    aie.objectfifo @of1 (%tile11, {%tile12}, 1 : i32) {repeat_count = 3 : i32} : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo.link [@of0] -> [@of1] ([] [])

    aie.objectfifo @of2 (%tile33, {%tile21}, 1 : i32) {repeat_count = 3 : i32} : !aie.objectfifo<memref<32xi32>>
    aie.objectfifo @of3 (%tile21, {%tile10}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo.link [@of2] -> [@of3] ([] [])
 }
}
