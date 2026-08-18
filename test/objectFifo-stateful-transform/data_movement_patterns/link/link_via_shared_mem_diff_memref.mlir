//===- link_via_shared_mem_diff_memref.mlir --------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2024 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Date: October 1st 2024
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform --aie-objectFifo-unroll %s | FileCheck %s

// CHECK-LABEL:   aie.device(xcve2302) {
// CHECK:           %[[VAL_0:.*]] = aie.tile(2, 0)
// CHECK:           %[[VAL_1:.*]] = aie.tile(1, 2)
// CHECK:           %[[VAL_2:.*]] = aie.buffer(%[[VAL_1]]) {sym_name = "of1_cons_buff_0"} : memref<32xi32>
// CHECK:           %[[VAL_3:.*]] = aie.buffer(%[[VAL_1]]) {sym_name = "of1_cons_buff_1"} : memref<32xi32>
// CHECK:           %[[VAL_4:.*]] = aie.lock(%[[VAL_1]]) {init = 2 : i32, sym_name = "of1_cons_prod_lock_0"}
// CHECK:           %[[VAL_5:.*]] = aie.lock(%[[VAL_1]]) {init = 0 : i32, sym_name = "of1_cons_cons_lock_0"}
// CHECK:           %[[VAL_6:.*]] = aie.tile(2, 2)
// CHECK:           %[[VAL_7:.*]] = aie.buffer(%[[VAL_6]]) {sym_name = "of2_cons_buff_0"} : memref<16xi32>
// CHECK:           %[[VAL_8:.*]] = aie.buffer(%[[VAL_6]]) {sym_name = "of2_cons_buff_1"} : memref<16xi32>
// CHECK:           %[[VAL_9:.*]] = aie.lock(%[[VAL_6]]) {init = 2 : i32, sym_name = "of2_cons_prod_lock_0"}
// CHECK:           %[[VAL_10:.*]] = aie.lock(%[[VAL_6]]) {init = 0 : i32, sym_name = "of2_cons_cons_lock_0"}
// CHECK:           aie.flow(%[[VAL_0]], DMA : 0, %[[VAL_1]], DMA : 0)
// CHECK:           aie.flow(%[[VAL_1]], DMA : 0, %[[VAL_6]], DMA : 0)
// CHECK:           %[[VAL_11:.*]] = aie.core(%[[VAL_6]]) {
// CHECK:             %[[VAL_12:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_10]], AcquireGreaterEqual, %[[VAL_12]])
// CHECK:             aie.use_lock(%[[VAL_9]], Release, %[[VAL_12]])
// CHECK:             aie.end
// CHECK:           }
// CHECK:           aie.shim_dma_allocation @of1_shim_alloc(%[[VAL_0]], MM2S, 0)
// CHECK:           %[[VAL_13:.*]] = aie.mem(%[[VAL_1]]) {
// CHECK:             %[[VAL_14:.*]] = arith.constant 1 : i32
// CHECK:             %[[VAL_15:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[VAL_4]], AcquireGreaterEqual, %[[VAL_14]])
// CHECK:             aie.dma_bd(%[[VAL_2]] : memref<32xi32> offset = 0 len = 32)
// CHECK:             aie.use_lock(%[[VAL_5]], Release, %[[VAL_14]])
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[VAL_4]], AcquireGreaterEqual, %[[VAL_14]])
// CHECK:             aie.dma_bd(%[[VAL_3]] : memref<32xi32> offset = 0 len = 32)
// CHECK:             aie.use_lock(%[[VAL_5]], Release, %[[VAL_14]])
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:
// CHECK:             %[[VAL_16:.*]] = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
// CHECK:           ^bb4:
// CHECK:             aie.use_lock(%[[VAL_5]], AcquireGreaterEqual, %[[VAL_14]])
// CHECK:             aie.dma_bd(%[[VAL_2]] : memref<32xi32> offset = 0 len = 32)
// CHECK:             aie.use_lock(%[[VAL_4]], Release, %[[VAL_14]])
// CHECK:             aie.next_bd ^bb5
// CHECK:           ^bb5:
// CHECK:             aie.use_lock(%[[VAL_5]], AcquireGreaterEqual, %[[VAL_14]])
// CHECK:             aie.dma_bd(%[[VAL_3]] : memref<32xi32> offset = 0 len = 32)
// CHECK:             aie.use_lock(%[[VAL_4]], Release, %[[VAL_14]])
// CHECK:             aie.next_bd ^bb4
// CHECK:           ^bb6:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_17:.*]] = aie.mem(%[[VAL_6]]) {
// CHECK:             %[[VAL_18:.*]] = arith.constant 1 : i32
// CHECK:             %[[VAL_19:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[VAL_9]], AcquireGreaterEqual, %[[VAL_18]])
// CHECK:             aie.dma_bd(%[[VAL_7]] : memref<16xi32> offset = 0 len = 16)
// CHECK:             aie.use_lock(%[[VAL_10]], Release, %[[VAL_18]])
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[VAL_9]], AcquireGreaterEqual, %[[VAL_18]])
// CHECK:             aie.dma_bd(%[[VAL_8]] : memref<16xi32> offset = 0 len = 16)
// CHECK:             aie.use_lock(%[[VAL_10]], Release, %[[VAL_18]])
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:
// CHECK:             aie.end
// CHECK:           }
// CHECK:         }

module @link_AIE2 {
    aie.device(xcve2302) {
        %tile20 = aie.tile(2, 0)
        %tile12 = aie.tile(1, 2)
        %tile22 = aie.tile(2, 2)

        aie.objectfifo @of1 (%tile20, {%tile12}, 2 : i32) : !aie.objectfifo<memref<32xi32>>
        aie.objectfifo @of2 (%tile12, {%tile22}, 2 : i32) : !aie.objectfifo<memref<16xi32>>

        aie.objectfifo.link [@of1] -> [@of2] ([] [])

                %core = aie.core(%tile22) {
                    %object = aie.objectfifo.acquire @of2(Consume) : memref<16xi32>
                    aie.objectfifo.release @of2(Consume) [1]
                    aie.end
                }
    }
}
