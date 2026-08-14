//===- plio_test.mlir -------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2024 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform --aie-objectFifo-unroll %s | FileCheck %s

// CHECK-LABEL:   aie.device(xcve2302) {
// CHECK:           %[[VAL_0:.*]] = aie.tile(2, 0)
// CHECK:           %[[VAL_1:.*]] = aie.tile(2, 2)
// CHECK:           %[[VAL_2:.*]] = aie.tile(2, 3)
// CHECK:           %[[VAL_3:.*]] = aie.buffer(%[[VAL_2]]) {sym_name = "of_2_1_cons_buff_0"} : memref<64xi16>
// CHECK:           %[[VAL_4:.*]] = aie.buffer(%[[VAL_2]]) {sym_name = "of_2_1_cons_buff_1"} : memref<64xi16>
// CHECK:           %[[VAL_5:.*]] = aie.lock(%[[VAL_2]]) {init = 2 : i32, sym_name = "of_2_1_cons_prod_lock_0"}
// CHECK:           %[[VAL_6:.*]] = aie.lock(%[[VAL_2]]) {init = 0 : i32, sym_name = "of_2_1_cons_cons_lock_0"}
// CHECK:           %[[VAL_7:.*]] = aie.buffer(%[[VAL_1]]) {sym_name = "of_2_buff_0"} : memref<64xi16>
// CHECK:           %[[VAL_8:.*]] = aie.buffer(%[[VAL_1]]) {sym_name = "of_2_buff_1"} : memref<64xi16>
// CHECK:           %[[VAL_9:.*]] = aie.lock(%[[VAL_1]]) {init = 2 : i32, sym_name = "of_2_prod_lock_0"}
// CHECK:           %[[VAL_10:.*]] = aie.lock(%[[VAL_1]]) {init = 0 : i32, sym_name = "of_2_cons_lock_0"}
// CHECK:           %[[VAL_11:.*]] = aie.buffer(%[[VAL_1]]) {sym_name = "of_1_buff_0"} : memref<64xi16>
// CHECK:           %[[VAL_12:.*]] = aie.buffer(%[[VAL_1]]) {sym_name = "of_1_buff_1"} : memref<64xi16>
// CHECK:           %[[VAL_13:.*]] = aie.lock(%[[VAL_1]]) {init = 2 : i32, sym_name = "of_1_prod_lock_0"}
// CHECK:           %[[VAL_14:.*]] = aie.lock(%[[VAL_1]]) {init = 0 : i32, sym_name = "of_1_cons_lock_0"}
// CHECK:           %[[VAL_15:.*]] = aie.buffer(%[[VAL_1]]) {sym_name = "of_0_cons_buff_0"} : memref<64xi16>
// CHECK:           %[[VAL_16:.*]] = aie.buffer(%[[VAL_1]]) {sym_name = "of_0_cons_buff_1"} : memref<64xi16>
// CHECK:           %[[VAL_17:.*]] = aie.lock(%[[VAL_1]]) {init = 2 : i32, sym_name = "of_0_cons_prod_lock_0"}
// CHECK:           %[[VAL_18:.*]] = aie.lock(%[[VAL_1]]) {init = 0 : i32, sym_name = "of_0_cons_cons_lock_0"}
// CHECK:           aie.flow(%[[VAL_0]], PLIO : 0, %[[VAL_1]], DMA : 0)
// CHECK:           aie.flow(%[[VAL_1]], DMA : 0, %[[VAL_0]], PLIO : 0)
// CHECK:           aie.flow(%[[VAL_1]], DMA : 1, %[[VAL_0]], PLIO : 1)
// CHECK:           aie.flow(%[[VAL_1]], DMA : 1, %[[VAL_2]], DMA : 0)
// CHECK:           aie.shim_dma_allocation @of_0_shim_alloc(%[[VAL_0]], MM2S, 0) {plio = true}
// CHECK:           aie.shim_dma_allocation @of_1_shim_alloc(%[[VAL_0]], S2MM, 0) {plio = true}
// CHECK:           aie.shim_dma_allocation @of_2_shim_alloc(%[[VAL_0]], S2MM, 1) {plio = true}
// CHECK:           %[[VAL_19:.*]] = aie.mem(%[[VAL_1]]) {
// CHECK:             %[[VAL_20:.*]] = arith.constant 1 : i32
// CHECK:             %[[VAL_21:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[VAL_17]], AcquireGreaterEqual, %[[VAL_20]])
// CHECK:             aie.dma_bd(%[[VAL_15]] : memref<64xi16> offset = 0 len = 64)
// CHECK:             aie.use_lock(%[[VAL_18]], Release, %[[VAL_20]])
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[VAL_17]], AcquireGreaterEqual, %[[VAL_20]])
// CHECK:             aie.dma_bd(%[[VAL_16]] : memref<64xi16> offset = 0 len = 64)
// CHECK:             aie.use_lock(%[[VAL_18]], Release, %[[VAL_20]])
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:
// CHECK:             %[[VAL_22:.*]] = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
// CHECK:           ^bb4:
// CHECK:             aie.use_lock(%[[VAL_14]], AcquireGreaterEqual, %[[VAL_20]])
// CHECK:             aie.dma_bd(%[[VAL_11]] : memref<64xi16> offset = 0 len = 64)
// CHECK:             aie.use_lock(%[[VAL_13]], Release, %[[VAL_20]])
// CHECK:             aie.next_bd ^bb5
// CHECK:           ^bb5:
// CHECK:             aie.use_lock(%[[VAL_14]], AcquireGreaterEqual, %[[VAL_20]])
// CHECK:             aie.dma_bd(%[[VAL_12]] : memref<64xi16> offset = 0 len = 64)
// CHECK:             aie.use_lock(%[[VAL_13]], Release, %[[VAL_20]])
// CHECK:             aie.next_bd ^bb4
// CHECK:           ^bb6:
// CHECK:             %[[VAL_23:.*]] = aie.dma_start(MM2S, 1, ^bb7, ^bb9)
// CHECK:           ^bb7:
// CHECK:             aie.use_lock(%[[VAL_10]], AcquireGreaterEqual, %[[VAL_20]])
// CHECK:             aie.dma_bd(%[[VAL_7]] : memref<64xi16> offset = 0 len = 64)
// CHECK:             aie.use_lock(%[[VAL_9]], Release, %[[VAL_20]])
// CHECK:             aie.next_bd ^bb8
// CHECK:           ^bb8:
// CHECK:             aie.use_lock(%[[VAL_10]], AcquireGreaterEqual, %[[VAL_20]])
// CHECK:             aie.dma_bd(%[[VAL_8]] : memref<64xi16> offset = 0 len = 64)
// CHECK:             aie.use_lock(%[[VAL_9]], Release, %[[VAL_20]])
// CHECK:             aie.next_bd ^bb7
// CHECK:           ^bb9:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_24:.*]] = aie.mem(%[[VAL_2]]) {
// CHECK:             %[[VAL_25:.*]] = arith.constant 1 : i32
// CHECK:             %[[VAL_26:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[VAL_5]], AcquireGreaterEqual, %[[VAL_25]])
// CHECK:             aie.dma_bd(%[[VAL_3]] : memref<64xi16> offset = 0 len = 64)
// CHECK:             aie.use_lock(%[[VAL_6]], Release, %[[VAL_25]])
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[VAL_5]], AcquireGreaterEqual, %[[VAL_25]])
// CHECK:             aie.dma_bd(%[[VAL_4]] : memref<64xi16> offset = 0 len = 64)
// CHECK:             aie.use_lock(%[[VAL_6]], Release, %[[VAL_25]])
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:
// CHECK:             aie.end
// CHECK:           }
// CHECK:         }

module @plio {
    aie.device(xcve2302) {
        %tile20 = aie.tile(2, 0)
        %tile22 = aie.tile(2, 2)
        %tile23 = aie.tile(2, 3)

        aie.objectfifo @of_0 (%tile20, {%tile22}, 2 : i32) {plio = true} : !aie.objectfifo<memref<64xi16>>
        aie.objectfifo @of_1 (%tile22, {%tile20}, 2 : i32) {plio = true} : !aie.objectfifo<memref<64xi16>>
        aie.objectfifo @of_2 (%tile22, {%tile20, %tile23}, 2 : i32) {plio = true} : !aie.objectfifo<memref<64xi16>>
    }
}
