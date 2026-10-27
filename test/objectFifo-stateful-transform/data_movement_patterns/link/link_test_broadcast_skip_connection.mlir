//===- link_test_broadcast_skip_connection.mlir -----------------*- MLIR -*-===//
//
// Copyright (C) 2023 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Date: June 28th 2023
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform="skip-verify=true" --aie-objectFifo-unroll %s | FileCheck %s

// CHECK-LABEL:   aie.device(xcve2302) {
// CHECK-DAG:           %[[VAL_0:.*]] = aie.tile(2, 0)
// CHECK-DAG:           %[[VAL_1:.*]] = aie.tile(2, 1)
// CHECK-DAG:           %[[VAL_2:.*]] = aie.tile(2, 2)
// CHECK-DAG:           %[[VAL_3:.*]] = aie.tile(3, 3)
// CHECK-DAG:           aie.buffer @[[VAL_4:skip_connection_cons_buff_0]](%[[VAL_3]]) : memref<16xi32>
// CHECK-DAG:           aie.buffer @[[VAL_5:skip_connection_cons_buff_1]](%[[VAL_3]]) : memref<16xi32>
// CHECK-DAG:           aie.lock @[[VAL_6:skip_connection_cons_prod_lock_0]](%[[VAL_3]]) {init = 2 : i32}
// CHECK-DAG:           aie.lock @[[VAL_7:skip_connection_cons_cons_lock_0]](%[[VAL_3]]) {init = 0 : i32}
// CHECK-DAG:           aie.buffer @[[VAL_8:skip_connection_buff_0]](%[[VAL_2]]) : memref<16xi32>
// CHECK-DAG:           aie.buffer @[[VAL_9:skip_connection_buff_1]](%[[VAL_2]]) : memref<16xi32>
// CHECK-DAG:           aie.lock @[[VAL_10:skip_connection_prod_lock_0]](%[[VAL_2]]) {init = 2 : i32}
// CHECK-DAG:           aie.lock @[[VAL_11:skip_connection_cons_lock_0]](%[[VAL_2]]) {init = 0 : i32}
// CHECK-DAG:           aie.buffer @[[VAL_12:link2_0_cons_buff_0]](%[[VAL_2]]) : memref<16xi32>
// CHECK-DAG:           aie.buffer @[[VAL_13:link2_0_cons_buff_1]](%[[VAL_2]]) : memref<16xi32>
// CHECK-DAG:           aie.lock @[[VAL_14:link2_0_cons_prod_lock_0]](%[[VAL_2]]) {init = 2 : i32}
// CHECK-DAG:           aie.lock @[[VAL_15:link2_0_cons_cons_lock_0]](%[[VAL_2]]) {init = 0 : i32}
// CHECK-DAG:           aie.buffer @[[VAL_16:link2_1_cons_buff_0]](%[[VAL_3]]) : memref<16xi32>
// CHECK-DAG:           aie.buffer @[[VAL_17:link2_1_cons_buff_1]](%[[VAL_3]]) : memref<16xi32>
// CHECK-DAG:           aie.buffer @[[VAL_18:link2_1_cons_buff_2]](%[[VAL_3]]) : memref<16xi32>
// CHECK-DAG:           aie.lock @[[VAL_19:link2_1_cons_prod_lock_0]](%[[VAL_3]]) {init = 3 : i32}
// CHECK-DAG:           aie.lock @[[VAL_20:link2_1_cons_cons_lock_0]](%[[VAL_3]]) {init = 0 : i32}
// CHECK-DAG:           aie.buffer @[[VAL_21:link1_cons_buff_0]](%[[VAL_1]]) : memref<48xi32>
// CHECK-DAG:           aie.buffer @[[VAL_22:link1_cons_buff_1]](%[[VAL_1]]) : memref<48xi32>
// CHECK-DAG:           aie.lock @[[VAL_23:link1_cons_prod_lock_0]](%[[VAL_1]]) {init = 2 : i32}
// CHECK-DAG:           aie.lock @[[VAL_24:link1_cons_cons_lock_0]](%[[VAL_1]]) {init = 0 : i32}
// CHECK-DAG:           aie.flow(%[[VAL_0]], DMA : 0, %[[VAL_1]], DMA : 0)
// CHECK-DAG:           aie.flow(%[[VAL_1]], DMA : 0, %[[VAL_3]], DMA : 1)
// CHECK-DAG:           aie.flow(%[[VAL_1]], DMA : 0, %[[VAL_2]], DMA : 0)
// CHECK-DAG:           aie.flow(%[[VAL_2]], DMA : 0, %[[VAL_3]], DMA : 0)
// CHECK-DAG:           aie.shim_dma_allocation @link1_shim_alloc(%[[VAL_0]], MM2S, 0)
// CHECK:           %[[VAL_27:.*]] = aie.memtile_dma(%[[VAL_1]]) {
// CHECK:             %[[VAL_28:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:  // 2 preds: ^bb0, ^bb2
// CHECK:             aie.use_lock(@[[VAL_23]], AcquireGreaterEqual, %{{.*}})
// CHECK:             %[[VAL_21_ACCESS_0:.*]] = aie.get_buffer @[[VAL_21]] : memref<48xi32>
// CHECK:             aie.dma_bd(%[[VAL_21_ACCESS_0]] : memref<48xi32> offset = {{.*}} len = {{.*}})
// CHECK:             aie.use_lock(@[[VAL_24]], Release, %{{.*}})
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:  // pred: ^bb1
// CHECK:             aie.use_lock(@[[VAL_23]], AcquireGreaterEqual, %{{.*}})
// CHECK:             %[[VAL_22_ACCESS_1:.*]] = aie.get_buffer @[[VAL_22]] : memref<48xi32>
// CHECK:             aie.dma_bd(%[[VAL_22_ACCESS_1]] : memref<48xi32> offset = {{.*}} len = {{.*}})
// CHECK:             aie.use_lock(@[[VAL_24]], Release, %{{.*}})
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:  // pred: ^bb0
// CHECK:             %[[VAL_29:.*]] = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
// CHECK:           ^bb4:  // 2 preds: ^bb3, ^bb5
// CHECK:             aie.use_lock(@[[VAL_24]], AcquireGreaterEqual, %{{.*}})
// CHECK:             %[[VAL_21_ACCESS_2:.*]] = aie.get_buffer @[[VAL_21]] : memref<48xi32>
// CHECK:             aie.dma_bd(%[[VAL_21_ACCESS_2]] : memref<48xi32> offset = {{.*}} len = {{.*}})
// CHECK:             aie.use_lock(@[[VAL_23]], Release, %{{.*}})
// CHECK:             aie.next_bd ^bb5
// CHECK:           ^bb5:  // pred: ^bb4
// CHECK:             aie.use_lock(@[[VAL_24]], AcquireGreaterEqual, %{{.*}})
// CHECK:             %[[VAL_22_ACCESS_3:.*]] = aie.get_buffer @[[VAL_22]] : memref<48xi32>
// CHECK:             aie.dma_bd(%[[VAL_22_ACCESS_3]] : memref<48xi32> offset = {{.*}} len = {{.*}})
// CHECK:             aie.use_lock(@[[VAL_23]], Release, %{{.*}})
// CHECK:             aie.next_bd ^bb4
// CHECK:           ^bb6:  // pred: ^bb3
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_30:.*]] = aie.mem(%[[VAL_2]]) {
// CHECK:             %[[VAL_31:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:  // 2 preds: ^bb0, ^bb2
// CHECK:             aie.use_lock(@[[VAL_14]], AcquireGreaterEqual, %{{.*}})
// CHECK:             %[[VAL_12_ACCESS_4:.*]] = aie.get_buffer @[[VAL_12]] : memref<16xi32>
// CHECK:             aie.dma_bd(%[[VAL_12_ACCESS_4]] : memref<16xi32> offset = {{.*}} len = {{.*}})
// CHECK:             aie.use_lock(@[[VAL_15]], Release, %{{.*}})
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:  // pred: ^bb1
// CHECK:             aie.use_lock(@[[VAL_14]], AcquireGreaterEqual, %{{.*}})
// CHECK:             %[[VAL_13_ACCESS_5:.*]] = aie.get_buffer @[[VAL_13]] : memref<16xi32>
// CHECK:             aie.dma_bd(%[[VAL_13_ACCESS_5]] : memref<16xi32> offset = {{.*}} len = {{.*}})
// CHECK:             aie.use_lock(@[[VAL_15]], Release, %{{.*}})
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:  // pred: ^bb0
// CHECK:             %[[VAL_32:.*]] = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
// CHECK:           ^bb4:  // 2 preds: ^bb3, ^bb5
// CHECK:             aie.use_lock(@[[VAL_11]], AcquireGreaterEqual, %{{.*}})
// CHECK:             %[[VAL_8_ACCESS_6:.*]] = aie.get_buffer @[[VAL_8]] : memref<16xi32>
// CHECK:             aie.dma_bd(%[[VAL_8_ACCESS_6]] : memref<16xi32> offset = {{.*}} len = {{.*}})
// CHECK:             aie.use_lock(@[[VAL_10]], Release, %{{.*}})
// CHECK:             aie.next_bd ^bb5
// CHECK:           ^bb5:  // pred: ^bb4
// CHECK:             aie.use_lock(@[[VAL_11]], AcquireGreaterEqual, %{{.*}})
// CHECK:             %[[VAL_9_ACCESS_7:.*]] = aie.get_buffer @[[VAL_9]] : memref<16xi32>
// CHECK:             aie.dma_bd(%[[VAL_9_ACCESS_7]] : memref<16xi32> offset = {{.*}} len = {{.*}})
// CHECK:             aie.use_lock(@[[VAL_10]], Release, %{{.*}})
// CHECK:             aie.next_bd ^bb4
// CHECK:           ^bb6:  // pred: ^bb3
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_33:.*]] = aie.mem(%[[VAL_3]]) {
// CHECK:             %[[VAL_35:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(@[[VAL_6]], AcquireGreaterEqual, %{{.*}})
// CHECK:             %[[VAL_4_ACCESS_8:.*]] = aie.get_buffer @[[VAL_4]] : memref<16xi32>
// CHECK:             aie.dma_bd(%[[VAL_4_ACCESS_8]] : memref<16xi32> offset = {{.*}} len = {{.*}})
// CHECK:             aie.use_lock(@[[VAL_7]], Release, %{{.*}})
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(@[[VAL_6]], AcquireGreaterEqual, %{{.*}})
// CHECK:             %[[VAL_5_ACCESS_9:.*]] = aie.get_buffer @[[VAL_5]] : memref<16xi32>
// CHECK:             aie.dma_bd(%[[VAL_5_ACCESS_9]] : memref<16xi32> offset = {{.*}} len = {{.*}})
// CHECK:             aie.use_lock(@[[VAL_7]], Release, %{{.*}})
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:
// CHECK:             %[[VAL_34:.*]] = aie.dma_start(S2MM, 1, ^bb4, ^bb7)
// CHECK:           ^bb4:
// CHECK:             aie.use_lock(@[[VAL_19]], AcquireGreaterEqual, %{{.*}})
// CHECK:             %[[VAL_16_ACCESS_10:.*]] = aie.get_buffer @[[VAL_16]] : memref<16xi32>
// CHECK:             aie.dma_bd(%[[VAL_16_ACCESS_10]] : memref<16xi32> offset = {{.*}} len = {{.*}})
// CHECK:             aie.use_lock(@[[VAL_20]], Release, %{{.*}})
// CHECK:             aie.next_bd ^bb5
// CHECK:           ^bb5:
// CHECK:             aie.use_lock(@[[VAL_19]], AcquireGreaterEqual, %{{.*}})
// CHECK:             %[[VAL_17_ACCESS_11:.*]] = aie.get_buffer @[[VAL_17]] : memref<16xi32>
// CHECK:             aie.dma_bd(%[[VAL_17_ACCESS_11]] : memref<16xi32> offset = {{.*}} len = {{.*}})
// CHECK:             aie.use_lock(@[[VAL_20]], Release, %{{.*}})
// CHECK:             aie.next_bd ^bb6
// CHECK:           ^bb6:
// CHECK:             aie.use_lock(@[[VAL_19]], AcquireGreaterEqual, %{{.*}})
// CHECK:             %[[VAL_18_ACCESS_12:.*]] = aie.get_buffer @[[VAL_18]] : memref<16xi32>
// CHECK:             aie.dma_bd(%[[VAL_18_ACCESS_12]] : memref<16xi32> offset = {{.*}} len = {{.*}})
// CHECK:             aie.use_lock(@[[VAL_20]], Release, %{{.*}})
// CHECK:             aie.next_bd ^bb4
// CHECK:           ^bb7:
// CHECK:             aie.end
// CHECK:           }
// CHECK:         }

module @link_test_broadcast_skip_connection {
    aie.device(xcve2302) {
        %tile20 = aie.tile(2, 0)
        %tile21 = aie.tile(2, 1)
        %tile22 = aie.tile(2, 2)
        %tile33 = aie.tile(3, 3)

        aie.objectfifo @link1 (%tile20, {%tile21}, 2 : i32) : !aie.objectfifo<memref<48xi32>>
        aie.objectfifo @link2 (%tile21, {%tile22, %tile33}, [2, 2, 3]) : !aie.objectfifo<memref<16xi32>>

        aie.objectfifo @skip_connection (%tile22, {%tile33}, 2 : i32) : !aie.objectfifo<memref<16xi32>>

        aie.objectfifo.link [@link1] -> [@link2] ([] [])
    }
}
