//===- link_via_shared_mem3.mlir --------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2024-2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform="skip-verify=true" --aie-objectFifo-unroll %s | FileCheck %s

// CHECK:  aie.device(xcve2302) {
// CHECK-DAG:    %{{.*}}tile_2_0 = aie.tile(2, 0)
// CHECK-DAG:    %{{.*}}tile_1_2 = aie.tile(1, 2)
// CHECK-DAG:    %{{.*}}tile_2_2 = aie.tile(2, 2)
// CHECK-DAG:    aie.buffer @[[VAL_0:of1_cons_buff_0]](%{{.*}}tile_1_2) : memref<16xi32>
// CHECK-DAG:    aie.buffer @[[VAL_1:of1_cons_buff_1]](%{{.*}}tile_1_2) : memref<16xi32>
// CHECK-DAG:    aie.lock @[[VAL_2:of1_cons_prod_lock_0]](%{{.*}}tile_1_2) {init = 2 : i32}
// CHECK-DAG:    aie.lock @[[VAL_3:of1_cons_cons_lock_0]](%{{.*}}tile_1_2) {init = 0 : i32}
// CHECK-DAG:    aie.flow(%{{.*}}tile_2_0, DMA : 0, %{{.*}}tile_1_2, DMA : 0)
// CHECK:    %[[VAL_6:.*]] = aie.mem(%{{.*}}tile_1_2) {
// CHECK:      %[[VAL_7:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:    ^bb1:
// CHECK:      aie.use_lock(@[[VAL_2]], AcquireGreaterEqual, %{{.*}})
// CHECK:      %[[VAL_0_ACCESS_0:.*]] = aie.get_buffer @[[VAL_0]] : memref<16xi32>
// CHECK:      aie.dma_bd(%[[VAL_0_ACCESS_0]] : memref<16xi32> offset = {{.*}} len = {{.*}})
// CHECK:      aie.use_lock(@[[VAL_3]], Release, %{{.*}})
// CHECK:      aie.next_bd ^bb2
// CHECK:    ^bb2:
// CHECK:      aie.use_lock(@[[VAL_2]], AcquireGreaterEqual, %{{.*}})
// CHECK:      %[[VAL_1_ACCESS_1:.*]] = aie.get_buffer @[[VAL_1]] : memref<16xi32>
// CHECK:      aie.dma_bd(%[[VAL_1_ACCESS_1]] : memref<16xi32> offset = {{.*}} len = {{.*}})
// CHECK:      aie.use_lock(@[[VAL_3]], Release, %{{.*}})
// CHECK:      aie.next_bd ^bb1
// CHECK:    ^bb3:
// CHECK:      aie.end
// CHECK:    }
// CHECK:  }

// In this design, the allocate operation applies to tile_2_2, to which tile_1_2
// does not have direct shared memory access: buffers and locks are created on
// both tiles, following default behaviour of a link.

module @link_AIE2 {
    aie.device(xcve2302) {
        %tile20 = aie.tile(2, 0)
        %tile12 = aie.tile(1, 2)
        %tile22 = aie.tile(2, 2)

        aie.objectfifo @of1 (%tile20, {%tile12}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
        aie.objectfifo @of2 (%tile12, {%tile22}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
        aie.objectfifo.allocate @of2 (%tile22)

        aie.objectfifo.link [@of1] -> [@of2] ([] [])
    }
}
