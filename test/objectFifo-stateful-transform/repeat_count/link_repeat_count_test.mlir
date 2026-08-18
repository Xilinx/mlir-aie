//===- link_repeat_count_test.mlir ------------------------------*- MLIR -*-===//
//
// Copyright (C) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform="skip-verify=true" --aie-objectFifo-unroll %s | FileCheck %s

// CHECK-LABEL:   aie.device(npu1) {
// CHECK:           %[[VAL_0:.*]] = aie.tile(1, 0)
// CHECK:           %[[VAL_1:.*]] = aie.tile(1, 1)
// CHECK:           %[[VAL_2:.*]] = aie.buffer(%[[VAL_1]]) {sym_name = "of0_cons_buff_0"} : memref<32xi32>
// CHECK:           %[[VAL_3:.*]] = aie.lock(%[[VAL_1]]) {init = 3 : i32, sym_name = "of0_cons_prod_lock_0"}
// CHECK:           %[[VAL_4:.*]] = aie.lock(%[[VAL_1]]) {init = 0 : i32, sym_name = "of0_cons_cons_lock_0"}
// CHECK:           %[[VAL_5:.*]] = aie.lock(%[[VAL_1]]) {init = 3 : i32, sym_name = "of0_cons_prod_lock_1"}
// CHECK:           %[[VAL_6:.*]] = aie.lock(%[[VAL_1]]) {init = 0 : i32, sym_name = "of0_cons_cons_lock_1"}
// CHECK:           %[[VAL_7:.*]] = aie.tile(2, 1)
// CHECK:           %[[VAL_8:.*]] = aie.buffer(%[[VAL_7]]) {sym_name = "of2_cons_buff_0"} : memref<32xi32>
// CHECK:           %[[VAL_9:.*]] = aie.lock(%[[VAL_7]]) {init = 1 : i32, sym_name = "of2_cons_prod_lock_0"}
// CHECK:           %[[VAL_10:.*]] = aie.lock(%[[VAL_7]]) {init = 0 : i32, sym_name = "of2_cons_cons_lock_0"}
// CHECK:           %[[VAL_11:.*]] = aie.lock(%[[VAL_7]]) {init = 1 : i32, sym_name = "of2_cons_prod_lock_1"}
// CHECK:           %[[VAL_12:.*]] = aie.lock(%[[VAL_7]]) {init = 0 : i32, sym_name = "of2_cons_cons_lock_1"}
// CHECK:           %[[VAL_13:.*]] = aie.tile(1, 2)
// CHECK:           %[[VAL_14:.*]] = aie.buffer(%[[VAL_13]]) {sym_name = "of1_cons_buff_0"} : memref<16xi32>
// CHECK:           %[[VAL_15:.*]] = aie.lock(%[[VAL_13]]) {init = 1 : i32, sym_name = "of1_cons_prod_lock_0"}
// CHECK:           %[[VAL_16:.*]] = aie.lock(%[[VAL_13]]) {init = 0 : i32, sym_name = "of1_cons_cons_lock_0"}
// CHECK:           %[[VAL_17:.*]] = aie.tile(3, 3)
// CHECK:           %[[VAL_18:.*]] = aie.buffer(%[[VAL_17]]) {sym_name = "of2_buff_0"} : memref<32xi32>
// CHECK:           %[[VAL_19:.*]] = aie.lock(%[[VAL_17]]) {init = 3 : i32, sym_name = "of2_prod_lock_0"}
// CHECK:           %[[VAL_20:.*]] = aie.lock(%[[VAL_17]]) {init = 0 : i32, sym_name = "of2_cons_lock_0"}
// CHECK:           aie.flow(%[[VAL_0]], DMA : 0, %[[VAL_1]], DMA : 0)
// CHECK:           aie.flow(%[[VAL_1]], DMA : 0, %[[VAL_13]], DMA : 0)
// CHECK:           aie.flow(%[[VAL_17]], DMA : 0, %[[VAL_7]], DMA : 0)
// CHECK:           aie.flow(%[[VAL_7]], DMA : 0, %[[VAL_0]], DMA : 0)
// CHECK:           aie.shim_dma_allocation @of0_shim_alloc(%[[VAL_0]], MM2S, 0)
// CHECK:           aie.shim_dma_allocation @of3_shim_alloc(%[[VAL_0]], S2MM, 0)
// CHECK:           %[[VAL_21:.*]] = aie.memtile_dma(%[[VAL_1]]) {
// CHECK:             %[[VAL_22:.*]] = arith.constant 3 : i32
// CHECK:             %[[VAL_23:.*]] = arith.constant 1 : i32
// CHECK:             %[[VAL_24:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[VAL_3]], AcquireGreaterEqual, %[[VAL_22]])
// CHECK:             aie.dma_bd(%[[VAL_2]] : memref<32xi32> offset = 0 len = 16)
// CHECK:             aie.use_lock(%[[VAL_4]], Release, %[[VAL_22]])
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[VAL_5]], AcquireGreaterEqual, %[[VAL_22]])
// CHECK:             aie.dma_bd(%[[VAL_2]] : memref<32xi32> offset = 16 len = 16)
// CHECK:             aie.use_lock(%[[VAL_6]], Release, %[[VAL_22]])
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:
// CHECK:             %[[VAL_25:.*]] = aie.dma_start(MM2S, 0, ^bb4, ^bb5, repeat_count = 2)
// CHECK:           ^bb4:
// CHECK:             aie.use_lock(%[[VAL_4]], AcquireGreaterEqual, %[[VAL_23]])
// CHECK:             aie.dma_bd(%[[VAL_2]] : memref<32xi32> offset = 0 len = 16)
// CHECK:             aie.use_lock(%[[VAL_3]], Release, %[[VAL_23]])
// CHECK:             aie.next_bd ^bb4
// CHECK:           ^bb5:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_26:.*]] = aie.mem(%[[VAL_13]]) {
// CHECK:             %[[VAL_27:.*]] = arith.constant 1 : i32
// CHECK:             %[[VAL_28:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[VAL_15]], AcquireGreaterEqual, %[[VAL_27]])
// CHECK:             aie.dma_bd(%[[VAL_14]] : memref<16xi32> offset = 0 len = 16)
// CHECK:             aie.use_lock(%[[VAL_16]], Release, %[[VAL_27]])
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb2:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_29:.*]] = aie.mem(%[[VAL_17]]) {
// CHECK:             %[[VAL_30:.*]] = arith.constant 1 : i32
// CHECK:             %[[VAL_31:.*]] = aie.dma_start(MM2S, 0, ^bb1, ^bb2, repeat_count = 2)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[VAL_20]], AcquireGreaterEqual, %[[VAL_30]])
// CHECK:             aie.dma_bd(%[[VAL_18]] : memref<32xi32> offset = 0 len = 32)
// CHECK:             aie.use_lock(%[[VAL_19]], Release, %[[VAL_30]])
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb2:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_32:.*]] = aie.memtile_dma(%[[VAL_7]]) {
// CHECK:             %[[VAL_33:.*]] = arith.constant 1 : i32
// CHECK:             %[[VAL_34:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[VAL_9]], AcquireGreaterEqual, %[[VAL_33]])
// CHECK:             aie.dma_bd(%[[VAL_8]] : memref<32xi32> offset = 0 len = 16)
// CHECK:             aie.use_lock(%[[VAL_10]], Release, %[[VAL_33]])
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[VAL_11]], AcquireGreaterEqual, %[[VAL_33]])
// CHECK:             aie.dma_bd(%[[VAL_8]] : memref<32xi32> offset = 16 len = 16)
// CHECK:             aie.use_lock(%[[VAL_12]], Release, %[[VAL_33]])
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:
// CHECK:             %[[VAL_35:.*]] = aie.dma_start(MM2S, 0, ^bb4, ^bb5)
// CHECK:           ^bb4:
// CHECK:             aie.use_lock(%[[VAL_10]], AcquireGreaterEqual, %[[VAL_33]])
// CHECK:             aie.dma_bd(%[[VAL_8]] : memref<32xi32> offset = 0 len = 16)
// CHECK:             aie.use_lock(%[[VAL_9]], Release, %[[VAL_33]])
// CHECK:             aie.next_bd ^bb4
// CHECK:           ^bb5:
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
