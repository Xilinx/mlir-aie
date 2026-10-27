//===- shim_dma_alloc_test.mlir ---------------------------------*- MLIR -*-===//
//
// Copyright (C) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform="packet-sw-objFifos=true skip-verify=true" --aie-objectFifo-unroll %s | FileCheck %s

// CHECK: module @shim_dma_alloc {
// CHECK:   aie.device(xcve2302) {
// CHECK-DAG:     %[[VAL_0:.*]] = aie.tile(1, 0)
// CHECK-DAG:     %[[VAL_2:.*]] = aie.tile(3, 3)
// CHECK-DAG:     aie.buffer @[[VAL_3:of1_cons_buff_0]](%[[VAL_2]]) : memref<16xi32>
// CHECK-DAG:     aie.buffer @[[VAL_4:of1_cons_buff_1]](%[[VAL_2]]) : memref<16xi32>
// CHECK-DAG:     aie.lock @[[VAL_5:of1_cons_prod_lock_0]](%[[VAL_2]]) {init = 2 : i32}
// CHECK-DAG:     aie.lock @[[VAL_6:of1_cons_cons_lock_0]](%[[VAL_2]]) {init = 0 : i32}
// CHECK-DAG:     aie.packet_flow(0) {
// CHECK:        aie.packet_source<%[[VAL_0]], DMA : 0>
// CHECK:        aie.packet_dest<%[[VAL_2]], DMA : 0>
// CHECK:     }
// CHECK-DAG:     aie.shim_dma_allocation @of1_shim_alloc(%[[VAL_0]], MM2S, 0, <pkt_type = 0, pkt_id = 0>)
// CHECK:     %[[VAL_12:.*]] = aie.mem(%[[VAL_2]]) {
// CHECK:       aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:     ^bb1:  // 2 preds: ^bb0, ^bb2
// CHECK:       aie.use_lock(@[[VAL_5]], AcquireGreaterEqual, %{{.*}})
// CHECK:       %[[VAL_3_ACCESS_0:.*]] = aie.get_buffer @[[VAL_3]] : memref<16xi32>
// CHECK:       aie.dma_bd(%[[VAL_3_ACCESS_0]] : memref<16xi32> offset = {{.*}} len = {{.*}})
// CHECK:       aie.use_lock(@[[VAL_6]], Release, %{{.*}})
// CHECK:       aie.next_bd ^bb2
// CHECK:     ^bb2:  // pred: ^bb1
// CHECK:       aie.use_lock(@[[VAL_5]], AcquireGreaterEqual, %{{.*}})
// CHECK:       %[[VAL_4_ACCESS_1:.*]] = aie.get_buffer @[[VAL_4]] : memref<16xi32>
// CHECK:       aie.dma_bd(%[[VAL_4_ACCESS_1]] : memref<16xi32> offset = {{.*}} len = {{.*}})
// CHECK:       aie.use_lock(@[[VAL_6]], Release, %{{.*}})
// CHECK:       aie.next_bd ^bb1
// CHECK:     ^bb3:  // pred: ^bb0
// CHECK:       aie.end
// CHECK:     }
// CHECK:   }
// CHECK: }

module @shim_dma_alloc {
 aie.device(xcve2302) {
    %tile10 = aie.tile(1, 0)
    %tile33 = aie.tile(3, 3)

    aie.objectfifo @of1 (%tile10, {%tile33}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
 }
}
