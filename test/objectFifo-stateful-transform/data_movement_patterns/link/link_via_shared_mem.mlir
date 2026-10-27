//===- link_via_shared_mem.mlir --------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2024 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Date: October 1st 2024
//
//===----------------------------------------------------------------------===//

//RUN: aie-opt --aie-objectFifo-stateful-transform --aie-objectFifo-unroll %s | FileCheck %s

//CHECK: module @link_AIE2 {
//CHECK:   aie.device(xcve2302) {
//CHECK-DAG:    %{{.*}}tile_2_0 = aie.tile(2, 0)
//CHECK-DAG:    %{{.*}}tile_1_2 = aie.tile(1, 2)
//CHECK-DAG:    %{{.*}}tile_2_2 = aie.tile(2, 2)
//CHECK-DAG:    aie.buffer @[[VAL_0:of2_cons_buff_0]](%{{.*}}tile_2_2) : memref<16xi32>
//CHECK-DAG:    aie.buffer @[[VAL_1:of2_cons_buff_1]](%{{.*}}tile_2_2) : memref<16xi32>
//CHECK-DAG:    aie.lock @[[VAL_2:of2_cons_prod_lock_0]](%{{.*}}tile_2_2) {init = 2 : i32}
//CHECK-DAG:    aie.lock @[[VAL_3:of2_cons_cons_lock_0]](%{{.*}}tile_2_2) {init = 0 : i32}
//CHECK-DAG:    aie.buffer @[[VAL_4:of1_cons_buff_0]](%{{.*}}tile_1_2) : memref<16xi32>
//CHECK-DAG:    aie.buffer @[[VAL_5:of1_cons_buff_1]](%{{.*}}tile_1_2) : memref<16xi32>
//CHECK-DAG:    aie.lock @[[VAL_6:of1_cons_prod_lock_0]](%{{.*}}tile_1_2) {init = 2 : i32}
//CHECK-DAG:    aie.lock @[[VAL_7:of1_cons_cons_lock_0]](%{{.*}}tile_1_2) {init = 0 : i32}
//CHECK-DAG:    aie.flow(%{{.*}}tile_2_0, DMA : 0, %{{.*}}tile_1_2, DMA : 0)
//CHECK-DAG:    aie.flow(%{{.*}}tile_1_2, DMA : 0, %{{.*}}tile_2_2, DMA : 0)
//CHECK:    func.func @some_work(%arg0: memref<16xi32>) {
//CHECK:      return
//CHECK:    }
//CHECK:    %core_2_2 = aie.core(%{{.*}}tile_2_2) {
//CHECK:      aie.use_lock(@[[VAL_3]], AcquireGreaterEqual, %{{.*}})
//CHECK:      func.call @some_work(%{{[^ ,)]+}}) : (memref<16xi32>) -> ()
//CHECK:      aie.use_lock(@[[VAL_2]], Release, %{{.*}})
//CHECK:      aie.end
//CHECK:    }
//CHECK-DAG:    aie.shim_dma_allocation @of1_shim_alloc(%shim_noc_tile_2_0, MM2S, 0)
//CHECK:    %mem_1_2 = aie.mem(%{{.*}}tile_1_2) {
//CHECK:      %[[CHECK_SSA_0:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
//CHECK:    ^bb1:
//CHECK:      aie.use_lock(@[[VAL_6]], AcquireGreaterEqual, %{{.*}})
//CHECK:      %[[VAL_4_ACCESS_0:.*]] = aie.get_buffer @[[VAL_4]] : memref<16xi32>
//CHECK:      aie.dma_bd(%[[VAL_4_ACCESS_0]] : memref<16xi32> offset = {{.*}} len = {{.*}})
//CHECK:      aie.use_lock(@[[VAL_7]], Release, %{{.*}})
//CHECK:      aie.next_bd ^bb2
//CHECK:    ^bb2:
//CHECK:      aie.use_lock(@[[VAL_6]], AcquireGreaterEqual, %{{.*}})
//CHECK:      %[[VAL_5_ACCESS_1:.*]] = aie.get_buffer @[[VAL_5]] : memref<16xi32>
//CHECK:      aie.dma_bd(%[[VAL_5_ACCESS_1]] : memref<16xi32> offset = {{.*}} len = {{.*}})
//CHECK:      aie.use_lock(@[[VAL_7]], Release, %{{.*}})
//CHECK:      aie.next_bd ^bb1
//CHECK:    ^bb3:
//CHECK:      %[[CHECK_SSA_1:.*]] = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
//CHECK:    ^bb4:
//CHECK:      aie.use_lock(@[[VAL_7]], AcquireGreaterEqual, %{{.*}})
//CHECK:      %[[VAL_4_ACCESS_2:.*]] = aie.get_buffer @[[VAL_4]] : memref<16xi32>
//CHECK:      aie.dma_bd(%[[VAL_4_ACCESS_2]] : memref<16xi32> offset = {{.*}} len = {{.*}})
//CHECK:      aie.use_lock(@[[VAL_6]], Release, %{{.*}})
//CHECK:      aie.next_bd ^bb5
//CHECK:    ^bb5:
//CHECK:      aie.use_lock(@[[VAL_7]], AcquireGreaterEqual, %{{.*}})
//CHECK:      %[[VAL_5_ACCESS_3:.*]] = aie.get_buffer @[[VAL_5]] : memref<16xi32>
//CHECK:      aie.dma_bd(%[[VAL_5_ACCESS_3]] : memref<16xi32> offset = {{.*}} len = {{.*}})
//CHECK:      aie.use_lock(@[[VAL_6]], Release, %{{.*}})
//CHECK:      aie.next_bd ^bb4
//CHECK:    ^bb6:
//CHECK:      aie.end
//CHECK:    }
//CHECK:    %mem_2_2 = aie.mem(%{{.*}}tile_2_2) {
//CHECK:      %[[CHECK_SSA_2:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
//CHECK:    ^bb1:
//CHECK:      aie.use_lock(@[[VAL_2]], AcquireGreaterEqual, %{{.*}})
//CHECK:      %[[VAL_0_ACCESS_4:.*]] = aie.get_buffer @[[VAL_0]] : memref<16xi32>
//CHECK:      aie.dma_bd(%[[VAL_0_ACCESS_4]] : memref<16xi32> offset = {{.*}} len = {{.*}})
//CHECK:      aie.use_lock(@[[VAL_3]], Release, %{{.*}})
//CHECK:      aie.next_bd ^bb2
//CHECK:    ^bb2:
//CHECK:      aie.use_lock(@[[VAL_2]], AcquireGreaterEqual, %{{.*}})
//CHECK:      %[[VAL_1_ACCESS_5:.*]] = aie.get_buffer @[[VAL_1]] : memref<16xi32>
//CHECK:      aie.dma_bd(%[[VAL_1_ACCESS_5]] : memref<16xi32> offset = {{.*}} len = {{.*}})
//CHECK:      aie.use_lock(@[[VAL_3]], Release, %{{.*}})
//CHECK:      aie.next_bd ^bb1
//CHECK:    ^bb3:
//CHECK:      aie.end
//CHECK:    }
//CHECK:  }

// In this design, there is no allocate operation: buffers and locks are created
// on both tiles, following default behaviour of a link.

module @link_AIE2 {
    aie.device(xcve2302) {
        %tile20 = aie.tile(2, 0)
        %tile12 = aie.tile(1, 2)
        %tile22 = aie.tile(2, 2)

        aie.objectfifo @of1 (%tile20, {%tile12}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
        aie.objectfifo @of2 (%tile12, {%tile22}, 2 : i32) : !aie.objectfifo<memref<16xi32>>

        aie.objectfifo.link [@of1] -> [@of2] ([] [])

        func.func @some_work(%lineOut : memref<16xi32>) -> () {
            return
        }

        %core22 = aie.core(%tile22) {
            %elem0 = aie.objectfifo.acquire @of2 (Consume, 1) : memref<16xi32>
            func.call @some_work(%elem0) : (memref<16xi32>) -> ()
            aie.objectfifo.release @of2 (Consume, 1)
            aie.end
        }
    }
}
