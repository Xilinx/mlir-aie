//===- link_test_AIE1.mlir --------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2023 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Date: May 9th 2023
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform --aie-objectFifo-unroll %s | FileCheck %s

// CHECK-LABEL:   aie.device(xcvc1902) {
// CHECK:           %[[VAL_0:.*]] = aie.tile(2, 0)
// CHECK:           %[[VAL_1:.*]] = aie.tile(1, 2)
// CHECK:           %[[VAL_2:.*]] = aie.tile(2, 2)
// CHECK:           %[[VAL_3:.*]] = aie.buffer(%[[VAL_2]]) {sym_name = "of2_cons_buff_0"} : memref<16xi32>
// CHECK:           %[[VAL_4:.*]] = aie.buffer(%[[VAL_2]]) {sym_name = "of2_cons_buff_1"} : memref<16xi32>
// CHECK:           %[[VAL_5:.*]] = aie.lock(%[[VAL_2]]) {init = 0 : i32, sym_name = "of2_cons_lock_0"}
// CHECK:           %[[VAL_6:.*]] = aie.lock(%[[VAL_2]]) {init = 0 : i32, sym_name = "of2_cons_lock_1"}
// CHECK:           %[[VAL_7:.*]] = aie.lock(%[[VAL_0]]) {init = 0 : i32, sym_name = "of1_lock_0"}
// CHECK:           %[[VAL_8:.*]] = aie.buffer(%[[VAL_1]]) {sym_name = "of1_cons_buff_0"} : memref<16xi32>
// CHECK:           %[[VAL_9:.*]] = aie.buffer(%[[VAL_1]]) {sym_name = "of1_cons_buff_1"} : memref<16xi32>
// CHECK:           %[[VAL_10:.*]] = aie.lock(%[[VAL_1]]) {init = 0 : i32, sym_name = "of1_cons_lock_0"}
// CHECK:           %[[VAL_11:.*]] = aie.lock(%[[VAL_1]]) {init = 0 : i32, sym_name = "of1_cons_lock_1"}
// CHECK:           aie.flow(%[[VAL_0]], DMA : 0, %[[VAL_1]], DMA : 0)
// CHECK:           aie.flow(%[[VAL_1]], DMA : 0, %[[VAL_2]], DMA : 0)
// CHECK:           %[[VAL_12:.*]] = aie.external_buffer {sym_name = "ext_buff_in"} : memref<16xi32>
// CHECK:           aie.shim_dma_allocation @of1_shim_alloc(%[[VAL_0]], MM2S, 0)
// CHECK:           %[[VAL_13:.*]] = aie.shim_dma(%[[VAL_0]]) {
// CHECK:             %[[VAL_14:.*]] = arith.constant 1 : i32
// CHECK:             %[[VAL_15:.*]] = arith.constant 0 : i32
// CHECK:             %[[VAL_16:.*]] = aie.dma_start(MM2S, 0, ^bb1, ^bb2)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[VAL_7]], Acquire, %[[VAL_14]])
// CHECK:             aie.dma_bd(%[[VAL_12]] : memref<16xi32> offset = 0 len = 16)
// CHECK:             aie.use_lock(%[[VAL_7]], Release, %[[VAL_15]])
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb2:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_17:.*]] = aie.mem(%[[VAL_1]]) {
// CHECK:             %[[VAL_18:.*]] = arith.constant 0 : i32
// CHECK:             %[[VAL_19:.*]] = arith.constant 1 : i32
// CHECK:             %[[VAL_20:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[VAL_10]], Acquire, %[[VAL_18]])
// CHECK:             aie.dma_bd(%[[VAL_8]] : memref<16xi32> offset = 0 len = 16)
// CHECK:             aie.use_lock(%[[VAL_10]], Release, %[[VAL_19]])
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[VAL_11]], Acquire, %[[VAL_18]])
// CHECK:             aie.dma_bd(%[[VAL_9]] : memref<16xi32> offset = 0 len = 16)
// CHECK:             aie.use_lock(%[[VAL_11]], Release, %[[VAL_19]])
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:
// CHECK:             %[[VAL_21:.*]] = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
// CHECK:           ^bb4:
// CHECK:             aie.use_lock(%[[VAL_10]], Acquire, %[[VAL_19]])
// CHECK:             aie.dma_bd(%[[VAL_8]] : memref<16xi32> offset = 0 len = 16)
// CHECK:             aie.use_lock(%[[VAL_10]], Release, %[[VAL_18]])
// CHECK:             aie.next_bd ^bb5
// CHECK:           ^bb5:
// CHECK:             aie.use_lock(%[[VAL_11]], Acquire, %[[VAL_19]])
// CHECK:             aie.dma_bd(%[[VAL_9]] : memref<16xi32> offset = 0 len = 16)
// CHECK:             aie.use_lock(%[[VAL_11]], Release, %[[VAL_18]])
// CHECK:             aie.next_bd ^bb4
// CHECK:           ^bb6:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_22:.*]] = aie.mem(%[[VAL_2]]) {
// CHECK:             %[[VAL_23:.*]] = arith.constant 0 : i32
// CHECK:             %[[VAL_24:.*]] = arith.constant 1 : i32
// CHECK:             %[[VAL_25:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[VAL_5]], Acquire, %[[VAL_23]])
// CHECK:             aie.dma_bd(%[[VAL_3]] : memref<16xi32> offset = 0 len = 16)
// CHECK:             aie.use_lock(%[[VAL_5]], Release, %[[VAL_24]])
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[VAL_6]], Acquire, %[[VAL_23]])
// CHECK:             aie.dma_bd(%[[VAL_4]] : memref<16xi32> offset = 0 len = 16)
// CHECK:             aie.use_lock(%[[VAL_6]], Release, %[[VAL_24]])
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:
// CHECK:             aie.end
// CHECK:           }
// CHECK:         }

module @link_AIE1 {
    aie.device(xcvc1902) {
        %tile20 = aie.tile(2, 0)
        %tile12 = aie.tile(1, 2)
        %tile22 = aie.tile(2, 2)

        aie.objectfifo @of1 (%tile20, {%tile12}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
        aie.objectfifo @of2 (%tile12, {%tile22}, 2 : i32) : !aie.objectfifo<memref<16xi32>>

        aie.objectfifo.link [@of1] -> [@of2] ([] [])

        %ext_buff_in = aie.external_buffer {sym_name = "ext_buff_in"} : memref<16xi32>
        aie.objectfifo.register_external_buffers @of1 (%tile20, {%ext_buff_in}) : (memref<16xi32>)
    }
}
