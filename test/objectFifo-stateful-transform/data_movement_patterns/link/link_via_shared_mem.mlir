//===- link_via_shared_mem.mlir --------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
//
// Date: October 1st 2024
//
//===----------------------------------------------------------------------===//

//RUN: aie-opt --aie-objectFifo-stateful-transform %s | FileCheck %s

//CHECK: module @link_AIE2 {
//CHECK:   aie.device(xcve2302) {
//CHECK:    memref.global "public" @of2_cons : memref<16xi32>
//CHECK:    memref.global "public" @of2 : memref<16xi32>
//CHECK:    memref.global "public" @of1_cons : memref<16xi32>
//CHECK:    memref.global "public" @of1 : memref<16xi32>
//CHECK:    %{{.*}}tile_2_0 = aie.tile(2, 0)
//CHECK:    %{{.*}}tile_1_2 = aie.tile(1, 2)
//CHECK:    %{{.*}}tile_2_2 = aie.tile(2, 2)
//CHECK:    %[[VAL_0:.*]] = aie.buffer(%{{.*}}tile_2_2) {sym_name = "of2_cons_buff_0"} : memref<16xi32>
//CHECK:    %[[VAL_1:.*]] = aie.buffer(%{{.*}}tile_2_2) {sym_name = "of2_cons_buff_1"} : memref<16xi32>
//CHECK:    %[[VAL_2:.*]] = aie.lock(%{{.*}}tile_2_2, 0) {init = 2 : i32, sym_name = "of2_cons_prod_lock_0"}
//CHECK:    %[[VAL_3:.*]] = aie.lock(%{{.*}}tile_2_2, 1) {init = 0 : i32, sym_name = "of2_cons_cons_lock_0"}
//CHECK:    %[[VAL_4:.*]] = aie.buffer(%{{.*}}tile_1_2) {sym_name = "of1_cons_buff_0"} : memref<16xi32>
//CHECK:    %[[VAL_5:.*]] = aie.buffer(%{{.*}}tile_1_2) {sym_name = "of1_cons_buff_1"} : memref<16xi32>
//CHECK:    %[[VAL_6:.*]] = aie.lock(%{{.*}}tile_1_2, 0) {init = 2 : i32, sym_name = "of1_cons_prod_lock_0"}
//CHECK:    %[[VAL_7:.*]] = aie.lock(%{{.*}}tile_1_2, 1) {init = 0 : i32, sym_name = "of1_cons_cons_lock_0"}
//CHECK:    aie.flow(%{{.*}}tile_2_0, DMA : 0, %{{.*}}tile_1_2, DMA : 0)
//CHECK:    aie.flow(%{{.*}}tile_1_2, DMA : 0, %{{.*}}tile_2_2, DMA : 0)
//CHECK:    func.func @some_work(%arg0: memref<16xi32>) {
//CHECK:      return
//CHECK:    }
//CHECK:    %core_2_2 = aie.core(%{{.*}}tile_2_2) {
//CHECK:      aie.use_lock(%[[VAL_3]], AcquireGreaterEqual, 1)
//CHECK:      func.call @some_work(%[[VAL_0]]) : (memref<16xi32>) -> ()
//CHECK:      aie.use_lock(%[[VAL_2]], Release, 1)
//CHECK:      aie.end
//CHECK:    }
//CHECK:    aie.shim_dma_allocation @of1(MM2S, 0, 2)
//CHECK:    %mem_1_2 = aie.mem(%{{.*}}tile_1_2) {
//CHECK:      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
//CHECK:    ^bb1:
//CHECK:      aie.use_lock(%[[VAL_6]], AcquireGreaterEqual, 1)
//CHECK:      aie.dma_bd(%[[VAL_4]] : memref<16xi32>, 0, 16)
//CHECK:      aie.use_lock(%[[VAL_7]], Release, 1)
//CHECK:      aie.next_bd ^bb2
//CHECK:    ^bb2:
//CHECK:      aie.use_lock(%[[VAL_6]], AcquireGreaterEqual, 1)
//CHECK:      aie.dma_bd(%[[VAL_5]] : memref<16xi32>, 0, 16)
//CHECK:      aie.use_lock(%[[VAL_7]], Release, 1)
//CHECK:      aie.next_bd ^bb1
//CHECK:    ^bb3:
//CHECK:      %1 = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
//CHECK:    ^bb4:
//CHECK:      aie.use_lock(%[[VAL_7]], AcquireGreaterEqual, 1)
//CHECK:      aie.dma_bd(%[[VAL_4]] : memref<16xi32>, 0, 16)
//CHECK:      aie.use_lock(%[[VAL_6]], Release, 1)
//CHECK:      aie.next_bd ^bb5
//CHECK:    ^bb5:
//CHECK:      aie.use_lock(%[[VAL_7]], AcquireGreaterEqual, 1)
//CHECK:      aie.dma_bd(%[[VAL_5]] : memref<16xi32>, 0, 16)
//CHECK:      aie.use_lock(%[[VAL_6]], Release, 1)
//CHECK:      aie.next_bd ^bb4
//CHECK:    ^bb6:
//CHECK:      aie.end
//CHECK:    }
//CHECK:    %mem_2_2 = aie.mem(%{{.*}}tile_2_2) {
//CHECK:      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
//CHECK:    ^bb1:
//CHECK:      aie.use_lock(%[[VAL_2]], AcquireGreaterEqual, 1)
//CHECK:      aie.dma_bd(%[[VAL_0]] : memref<16xi32>, 0, 16)
//CHECK:      aie.use_lock(%[[VAL_3]], Release, 1)
//CHECK:      aie.next_bd ^bb2
//CHECK:    ^bb2:
//CHECK:      aie.use_lock(%[[VAL_2]], AcquireGreaterEqual, 1)
//CHECK:      aie.dma_bd(%[[VAL_1]] : memref<16xi32>, 0, 16)
//CHECK:      aie.use_lock(%[[VAL_3]], Release, 1)
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
            %subview = aie.objectfifo.acquire @of2 (Consume, 1) : !aie.objectfifosubview<memref<16xi32>>
            %elem0 = aie.objectfifo.subview.access %subview[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
            func.call @some_work(%elem0) : (memref<16xi32>) -> ()
            aie.objectfifo.release @of2 (Consume, 1)
            aie.end
        }
    }
}
