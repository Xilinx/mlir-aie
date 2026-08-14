//===- link_repeat_count_test.mlir ------------------------------*- MLIR -*-===//
//
// Copyright (C) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform --aie-objectFifo-unroll %s | FileCheck %s

// CHECK-LABEL:   aie.device(npu1) {
// CHECK:           %[[VAL_0:.*]] = aie.tile(1, 0)
// CHECK:           %[[VAL_1:.*]] = aie.tile(1, 1)
// CHECK:           %[[VAL_2:.*]] = aie.tile(2, 1)
// CHECK:           %[[VAL_3:.*]] = aie.tile(1, 2)
// CHECK:           %[[VAL_4:.*]] = aie.tile(3, 3)
// CHECK:           %[[VAL_5:.*]] = aie.buffer(%[[VAL_4]]) {sym_name = "of2_buff_0"} : memref<32xi32>
// CHECK:           %[[VAL_6:.*]] = aie.lock(%[[VAL_4]]) {init = 3 : i32, sym_name = "of2_prod_lock_0"}
// CHECK:           %[[VAL_7:.*]] = aie.lock(%[[VAL_4]]) {init = 0 : i32, sym_name = "of2_cons_lock_0"}
// CHECK:           %[[VAL_8:.*]] = aie.buffer(%[[VAL_2]]) {sym_name = "of2_cons_buff_0"} : memref<32xi32>
// CHECK:           %[[VAL_9:.*]] = aie.lock(%[[VAL_2]]) {init = 1 : i32, sym_name = "of2_cons_prod_lock_0"}
// CHECK:           %[[VAL_10:.*]] = aie.lock(%[[VAL_2]]) {init = 0 : i32, sym_name = "of2_cons_cons_lock_0"}
// CHECK:           %[[VAL_11:.*]] = aie.buffer(%[[VAL_3]]) {sym_name = "of1_cons_buff_0"} : memref<16xi32>
// CHECK:           %[[VAL_12:.*]] = aie.lock(%[[VAL_3]]) {init = 1 : i32, sym_name = "of1_cons_prod_lock_0"}
// CHECK:           %[[VAL_13:.*]] = aie.lock(%[[VAL_3]]) {init = 0 : i32, sym_name = "of1_cons_cons_lock_0"}
// CHECK:           %[[VAL_14:.*]] = aie.buffer(%[[VAL_1]]) {sym_name = "of0_cons_buff_0"} : memref<32xi32>
// CHECK:           %[[VAL_15:.*]] = aie.lock(%[[VAL_1]]) {init = 3 : i32, sym_name = "of0_cons_prod_lock_0"}
// CHECK:           %[[VAL_16:.*]] = aie.lock(%[[VAL_1]]) {init = 0 : i32, sym_name = "of0_cons_cons_lock_0"}
// CHECK:           aie.flow(%[[VAL_0]], DMA : 0, %[[VAL_1]], DMA : 0)
// CHECK:           aie.flow(%[[VAL_1]], DMA : 0, %[[VAL_3]], DMA : 0)
// CHECK:           aie.flow(%[[VAL_4]], DMA : 0, %[[VAL_2]], DMA : 0)
// CHECK:           aie.flow(%[[VAL_2]], DMA : 0, %[[VAL_0]], DMA : 0)
// CHECK:           aie.shim_dma_allocation @of0_shim_alloc(%[[VAL_0]], MM2S, 0)
// CHECK:           aie.shim_dma_allocation @of3_shim_alloc(%[[VAL_0]], S2MM, 0)
// CHECK:           %[[VAL_17:.*]] = aie.memtile_dma(%[[VAL_1]]) {
// CHECK:             %[[VAL_18:.*]] = arith.constant 3 : i32
// CHECK:             %[[VAL_19:.*]] = arith.constant 1 : i32
// CHECK:             %[[VAL_20:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[VAL_15]], AcquireGreaterEqual, %[[VAL_18]])
// CHECK:             aie.dma_bd(%[[VAL_14]] : memref<32xi32> offset = 0 len = 32)
// CHECK:             aie.use_lock(%[[VAL_16]], Release, %[[VAL_18]])
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb2:
// CHECK:             %[[VAL_21:.*]] = aie.dma_start(MM2S, 0, ^bb3, ^bb4, repeat_count = 2)
// CHECK:           ^bb3:
// CHECK:             aie.use_lock(%[[VAL_16]], AcquireGreaterEqual, %[[VAL_19]])
// CHECK:             aie.dma_bd(%[[VAL_14]] : memref<32xi32> offset = 0 len = 32)
// CHECK:             aie.use_lock(%[[VAL_15]], Release, %[[VAL_19]])
// CHECK:             aie.next_bd ^bb3
// CHECK:           ^bb4:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_22:.*]] = aie.mem(%[[VAL_3]]) {
// CHECK:             %[[VAL_23:.*]] = arith.constant 1 : i32
// CHECK:             %[[VAL_24:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[VAL_12]], AcquireGreaterEqual, %[[VAL_23]])
// CHECK:             aie.dma_bd(%[[VAL_11]] : memref<16xi32> offset = 0 len = 16)
// CHECK:             aie.use_lock(%[[VAL_13]], Release, %[[VAL_23]])
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb2:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_25:.*]] = aie.mem(%[[VAL_4]]) {
// CHECK:             %[[VAL_26:.*]] = arith.constant 1 : i32
// CHECK:             %[[VAL_27:.*]] = aie.dma_start(MM2S, 0, ^bb1, ^bb2, repeat_count = 2)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[VAL_7]], AcquireGreaterEqual, %[[VAL_26]])
// CHECK:             aie.dma_bd(%[[VAL_5]] : memref<32xi32> offset = 0 len = 32)
// CHECK:             aie.use_lock(%[[VAL_6]], Release, %[[VAL_26]])
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb2:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_28:.*]] = aie.memtile_dma(%[[VAL_2]]) {
// CHECK:             %[[VAL_29:.*]] = arith.constant 1 : i32
// CHECK:             %[[VAL_30:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[VAL_9]], AcquireGreaterEqual, %[[VAL_29]])
// CHECK:             aie.dma_bd(%[[VAL_8]] : memref<32xi32> offset = 0 len = 32)
// CHECK:             aie.use_lock(%[[VAL_10]], Release, %[[VAL_29]])
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb2:
// CHECK:             %[[VAL_31:.*]] = aie.dma_start(MM2S, 0, ^bb3, ^bb4)
// CHECK:           ^bb3:
// CHECK:             aie.use_lock(%[[VAL_10]], AcquireGreaterEqual, %[[VAL_29]])
// CHECK:             aie.dma_bd(%[[VAL_8]] : memref<32xi32> offset = 0 len = 32)
// CHECK:             aie.use_lock(%[[VAL_9]], Release, %[[VAL_29]])
// CHECK:             aie.next_bd ^bb3
// CHECK:           ^bb4:
// CHECK:             aie.end
// CHECK:           }
// CHECK:         }

module @memtileRepeat {
 aie.device(npu1) {
    %tile10 = aie.tile(1, 0)
    %tile11 = aie.tile(1, 1)
    %tile21 = aie.tile(2, 1)
    %tile12 = aie.tile(1, 2)
    %tile33 = aie.tile(3, 3)

    aie.objectfifo @of0 (%tile10, {%tile11}, 1 : i32) : !aie.objectfifo<memref<32xi32>>
    aie.objectfifo @of1 (%tile11, {%tile12}, 1 : i32) {repeat_count = 3 : i32} : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo.link [@of0] -> [@of1] ([] [])

    aie.objectfifo @of2 (%tile33, {%tile21}, 1 : i32) {repeat_count = 3 : i32} : !aie.objectfifo<memref<32xi32>>
    aie.objectfifo @of3 (%tile21, {%tile10}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo.link [@of2] -> [@of3] ([] [])
 }
}
