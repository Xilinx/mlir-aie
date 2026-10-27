//===- no_register_external_buffers_test.mlir -------------------*- MLIR -*-===//
//
// Copyright (C) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform="skip-verify=true" --aie-objectFifo-unroll %s | FileCheck %s

// CHECK:           aie.device(xcvc1902) {
// CHECK-DAG:             %{{.*}}tile_7_2 = aie.tile(7, 2)
// CHECK-DAG:             %{{.*}}tile_7_0 = aie.tile(7, 0)
// CHECK-DAG:             aie.buffer @[[VAL_0:ext_of_cons_buff_0]](%{{.*}}tile_7_2) : memref<16xi32>
// CHECK-DAG:             aie.buffer @[[VAL_1:ext_of_cons_buff_1]](%{{.*}}tile_7_2) : memref<16xi32>
// CHECK-DAG:             aie.buffer @[[VAL_2:ext_of_cons_buff_2]](%{{.*}}tile_7_2) : memref<16xi32>
// CHECK-DAG:             aie.lock @[[VAL_3:ext_of_cons_lock_0]](%{{.*}}tile_7_2) {init = 0 : i32}
// CHECK-DAG:             aie.lock @[[VAL_4:ext_of_cons_lock_1]](%{{.*}}tile_7_2) {init = 0 : i32}
// CHECK-DAG:             aie.lock @[[VAL_5:ext_of_cons_lock_2]](%{{.*}}tile_7_2) {init = 0 : i32}
// CHECK-DAG:             aie.flow(%{{.*}}tile_7_0, DMA : 0, %{{.*}}tile_7_2, DMA : 0)
// CHECK-DAG:             aie.shim_dma_allocation @ext_of_shim_alloc(%shim_noc_tile_7_0, MM2S, 0)
// CHECK:             %mem_7_2 = aie.mem(%{{.*}}tile_7_2) {
// CHECK:               %[[CHECK_SSA_0:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb4)
// CHECK:             ^bb1:  // 2 preds: ^bb0, ^bb3
// CHECK:               aie.use_lock(@[[VAL_3]], Acquire, %{{.*}})
// CHECK:               %[[VAL_0_ACCESS_0:.*]] = aie.get_buffer @[[VAL_0]] : memref<16xi32>
// CHECK:               aie.dma_bd(%[[VAL_0_ACCESS_0]] : memref<16xi32> offset = {{.*}} len = {{.*}})
// CHECK:               aie.use_lock(@[[VAL_3]], Release, %{{.*}})
// CHECK:               aie.next_bd ^bb2
// CHECK:             ^bb2:  // pred: ^bb1
// CHECK:               aie.use_lock(@[[VAL_4]], Acquire, %{{.*}})
// CHECK:               %[[VAL_1_ACCESS_1:.*]] = aie.get_buffer @[[VAL_1]] : memref<16xi32>
// CHECK:               aie.dma_bd(%[[VAL_1_ACCESS_1]] : memref<16xi32> offset = {{.*}} len = {{.*}})
// CHECK:               aie.use_lock(@[[VAL_4]], Release, %{{.*}})
// CHECK:               aie.next_bd ^bb3
// CHECK:             ^bb3:  // pred: ^bb2
// CHECK:               aie.use_lock(@[[VAL_5]], Acquire, %{{.*}})
// CHECK:               %[[VAL_2_ACCESS_2:.*]] = aie.get_buffer @[[VAL_2]] : memref<16xi32>
// CHECK:               aie.dma_bd(%[[VAL_2_ACCESS_2]] : memref<16xi32> offset = {{.*}} len = {{.*}})
// CHECK:               aie.use_lock(@[[VAL_5]], Release, %{{.*}})
// CHECK:               aie.next_bd ^bb1
// CHECK:             ^bb4:  // pred: ^bb0
// CHECK:               aie.end
// CHECK:             }
// CHECK:           }

module @no_register_external_buffers {
 aie.device(xcvc1902) {
    %tile72 = aie.tile(7, 2)
    %tile70 = aie.tile(7, 0)

    aie.objectfifo @ext_of (%tile70, {%tile72}, 3 : i32) : !aie.objectfifo<memref<16xi32>>
 }
}
