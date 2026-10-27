//===- link_join_repeat_count_test.mlir -------------------------*- MLIR -*-===//
//
// Copyright (C) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform="skip-verify=true" --aie-objectFifo-unroll %s | FileCheck %s

// CHECK: module @memtileRepeat {
// CHECK:   aie.device(npu1) {
// CHECK-DAG:     %{{.*}}tile_1_0 = aie.tile(1, 0)
// CHECK-DAG:     %{{.*}}tile_1_1 = aie.tile(1, 1)
// CHECK-DAG:     %{{.*}}tile_1_2 = aie.tile(1, 2)
// CHECK-DAG:     %{{.*}}tile_3_3 = aie.tile(3, 3)
// CHECK-DAG: aie.buffer @[[OF2_BUF:of2_buff_0]]({{.*}}) : memref<32xi32>
// CHECK-DAG: aie.lock @[[OF2_PROD0:of2_prod_lock_0]]({{.*}}) {init = 1 : i32}
// CHECK-DAG: aie.lock @[[OF2_CONS0:of2_cons_lock_0]]({{.*}}) {init = 0 : i32}
// CHECK-DAG: aie.lock @[[OF2_PROD1:of2_prod_lock_1]]({{.*}}) {init = 1 : i32}
// CHECK-DAG: aie.lock @[[OF2_CONS1:of2_cons_lock_1]]({{.*}}) {init = 0 : i32}
// CHECK-DAG: aie.buffer @[[OF1_BUF:of1_buff_0]]({{.*}}) : memref<16xi32>
// CHECK-DAG: aie.lock @[[OF1_PROD:of1_prod_lock_0]]({{.*}}) {init = 3 : i32}
// CHECK-DAG: aie.lock @[[OF1_CONS:of1_cons_lock_0]]({{.*}}) {init = 0 : i32}
// CHECK-DAG: aie.buffer @[[OF0_BUF:of0_buff_0]]({{.*}}) : memref<16xi32>
// CHECK-DAG: aie.lock @[[OF0_PROD:of0_prod_lock_0]]({{.*}}) {init = 3 : i32}
// CHECK-DAG: aie.lock @[[OF0_CONS:of0_cons_lock_0]]({{.*}}) {init = 0 : i32}
// CHECK-DAG:     aie.flow(%{{.*}}tile_1_2, DMA : 0, %{{.*}}tile_1_1, DMA : 0)
// CHECK-DAG:     aie.flow(%{{.*}}tile_3_3, DMA : 0, %{{.*}}tile_1_1, DMA : 1)
// CHECK-DAG:     aie.flow(%{{.*}}tile_1_1, DMA : 0, %{{.*}}tile_1_0, DMA : 0)
// CHECK-DAG:     aie.shim_dma_allocation @of2_shim_alloc(%shim_noc_tile_1_0, S2MM, 0)
// CHECK:     %mem_1_2 = aie.mem(%{{.*}}tile_1_2) {
// CHECK:       %[[CHECK_SSA_0:.*]] = aie.dma_start(MM2S, 0, ^bb1, ^bb2, repeat_count = 2)
// CHECK:     ^bb1:
// CHECK:       aie.use_lock(@[[OF0_CONS]], AcquireGreaterEqual, %{{.*}})
// CHECK:       %[[OF0_BUF_ACCESS_0:.*]] = aie.get_buffer @[[OF0_BUF]] : memref<16xi32>
// CHECK:       aie.dma_bd(%[[OF0_BUF_ACCESS_0]] : memref<16xi32> offset = {{.*}} len = {{.*}})
// CHECK:       aie.use_lock(@[[OF0_PROD]], Release, %{{.*}})
// CHECK:       aie.next_bd ^bb1
// CHECK:     ^bb2:
// CHECK:       aie.end
// CHECK:     }
// CHECK:     %memtile_dma_1_1 = aie.memtile_dma(%{{.*}}tile_1_1) {
// CHECK:       %[[CHECK_SSA_1:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
// CHECK:     ^bb1:
// CHECK:       aie.use_lock(@[[OF2_PROD0]], AcquireGreaterEqual, %{{.*}})
// CHECK:       %[[OF2_BUF_ACCESS_1:.*]] = aie.get_buffer @[[OF2_BUF]] : memref<32xi32>
// CHECK:       aie.dma_bd(%[[OF2_BUF_ACCESS_1]] : memref<32xi32> offset = {{.*}} len = {{.*}})
// CHECK:       aie.use_lock(@[[OF2_CONS0]], Release, %{{.*}})
// CHECK:       aie.next_bd ^bb1
// CHECK:     ^bb2:
// CHECK:       %[[CHECK_SSA_2:.*]] = aie.dma_start(S2MM, 1, ^bb3, ^bb4)
// CHECK:     ^bb3:
// CHECK:       aie.use_lock(@[[OF2_PROD1]], AcquireGreaterEqual, %{{.*}})
// CHECK:       %[[OF2_BUF_ACCESS_2:.*]] = aie.get_buffer @[[OF2_BUF]] : memref<32xi32>
// CHECK:       aie.dma_bd(%[[OF2_BUF_ACCESS_2]] : memref<32xi32> offset = {{.*}} len = {{.*}})
// CHECK:       aie.use_lock(@[[OF2_CONS1]], Release, %{{.*}})
// CHECK:       aie.next_bd ^bb3
// CHECK:     ^bb4:
// CHECK:       %[[CHECK_SSA_3:.*]] = aie.dma_start(MM2S, 0, ^bb5, ^bb7)
// CHECK:     ^bb5:
// CHECK:       aie.use_lock(@[[OF2_CONS0]], AcquireGreaterEqual, %{{.*}})
// CHECK:       %[[OF2_BUF_ACCESS_3:.*]] = aie.get_buffer @[[OF2_BUF]] : memref<32xi32>
// CHECK:       aie.dma_bd(%[[OF2_BUF_ACCESS_3]] : memref<32xi32> offset = {{.*}} len = {{.*}})
// CHECK:       aie.use_lock(@[[OF2_PROD0]], Release, %{{.*}})
// CHECK:       aie.next_bd ^bb6
// CHECK:     ^bb6:
// CHECK:       aie.use_lock(@[[OF2_CONS1]], AcquireGreaterEqual, %{{.*}})
// CHECK:       %[[OF2_BUF_ACCESS_4:.*]] = aie.get_buffer @[[OF2_BUF]] : memref<32xi32>
// CHECK:       aie.dma_bd(%[[OF2_BUF_ACCESS_4]] : memref<32xi32> offset = {{.*}} len = {{.*}})
// CHECK:       aie.use_lock(@[[OF2_PROD1]], Release, %{{.*}})
// CHECK:       aie.next_bd ^bb5
// CHECK:     ^bb7:
// CHECK:       aie.end
// CHECK:     }
// CHECK:     %mem_3_3 = aie.mem(%{{.*}}tile_3_3) {
// CHECK:       %[[CHECK_SSA_4:.*]] = aie.dma_start(MM2S, 0, ^bb1, ^bb2, repeat_count = 2)
// CHECK:     ^bb1:
// CHECK:       aie.use_lock(@[[OF1_CONS]], AcquireGreaterEqual, %{{.*}})
// CHECK:       %[[OF1_BUF_ACCESS_5:.*]] = aie.get_buffer @[[OF1_BUF]] : memref<16xi32>
// CHECK:       aie.dma_bd(%[[OF1_BUF_ACCESS_5]] : memref<16xi32> offset = {{.*}} len = {{.*}})
// CHECK:       aie.use_lock(@[[OF1_PROD]], Release, %{{.*}})
// CHECK:       aie.next_bd ^bb1
// CHECK:     ^bb2:
// CHECK:       aie.end
// CHECK:     }
// CHECK:   }
// CHECK: }

module @memtileRepeat {
 aie.device(npu1) {
    %tile10 = aie.tile(1, 0)
    %tile11 = aie.tile(1, 1)
    %tile12 = aie.tile(1, 2)
    %tile33 = aie.tile(3, 3)

    aie.objectfifo @of0 (%tile12, {%tile11}, 1 : i32) {repeat_count = 3 : i32} : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @of1 (%tile33, {%tile11}, 1 : i32) {repeat_count = 3 : i32} : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @of2 (%tile11, {%tile10}, 1 : i32) : !aie.objectfifo<memref<32xi32>>
    aie.objectfifo.link [@of0, @of1] -> [@of2] ([0, 16] [])
 }
}
