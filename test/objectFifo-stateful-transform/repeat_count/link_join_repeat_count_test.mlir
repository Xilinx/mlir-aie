//===- link_join_repeat_count_test.mlir -------------------------*- MLIR -*-===//
//
// Copyright (C) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform="skip-verify=true" --aie-objectFifo-unroll %s | FileCheck %s

// CHECK-LABEL:   aie.device(npu1) {
// CHECK:           %[[VAL_0:.*]] = aie.tile(1, 0)
// CHECK:           %[[VAL_1:.*]] = aie.tile(1, 1)
// CHECK:           %[[VAL_2:.*]] = aie.buffer(%[[VAL_1]]) {sym_name = "of2_buff_0"} : memref<32xi32>
// CHECK:           %[[VAL_3:.*]] = aie.lock(%[[VAL_1]]) {init = 1 : i32, sym_name = "of2_prod_lock_0"}
// CHECK:           %[[VAL_4:.*]] = aie.lock(%[[VAL_1]]) {init = 0 : i32, sym_name = "of2_cons_lock_0"}
// CHECK:           %[[VAL_5:.*]] = aie.lock(%[[VAL_1]]) {init = 1 : i32, sym_name = "of2_prod_lock_1"}
// CHECK:           %[[VAL_6:.*]] = aie.lock(%[[VAL_1]]) {init = 0 : i32, sym_name = "of2_cons_lock_1"}
// CHECK:           %[[VAL_7:.*]] = aie.tile(1, 2)
// CHECK:           %[[VAL_8:.*]] = aie.buffer(%[[VAL_7]]) {sym_name = "of0_buff_0"} : memref<16xi32>
// CHECK:           %[[VAL_9:.*]] = aie.lock(%[[VAL_7]]) {init = 3 : i32, sym_name = "of0_prod_lock_0"}
// CHECK:           %[[VAL_10:.*]] = aie.lock(%[[VAL_7]]) {init = 0 : i32, sym_name = "of0_cons_lock_0"}
// CHECK:           %[[VAL_11:.*]] = aie.tile(3, 3)
// CHECK:           %[[VAL_12:.*]] = aie.buffer(%[[VAL_11]]) {sym_name = "of1_buff_0"} : memref<16xi32>
// CHECK:           %[[VAL_13:.*]] = aie.lock(%[[VAL_11]]) {init = 3 : i32, sym_name = "of1_prod_lock_0"}
// CHECK:           %[[VAL_14:.*]] = aie.lock(%[[VAL_11]]) {init = 0 : i32, sym_name = "of1_cons_lock_0"}
// CHECK:           aie.flow(%[[VAL_7]], DMA : 0, %[[VAL_1]], DMA : 0)
// CHECK:           aie.flow(%[[VAL_11]], DMA : 0, %[[VAL_1]], DMA : 1)
// CHECK:           aie.flow(%[[VAL_1]], DMA : 0, %[[VAL_0]], DMA : 0)
// CHECK:           aie.shim_dma_allocation @of2_shim_alloc(%[[VAL_0]], S2MM, 0)
// CHECK:           %[[VAL_15:.*]] = aie.mem(%[[VAL_7]]) {
// CHECK:             %[[VAL_16:.*]] = arith.constant 1 : i32
// CHECK:             %[[VAL_17:.*]] = aie.dma_start(MM2S, 0, ^bb1, ^bb2, repeat_count = 2)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[VAL_10]], AcquireGreaterEqual, %[[VAL_16]])
// CHECK:             aie.dma_bd(%[[VAL_8]] : memref<16xi32> offset = 0 len = 16)
// CHECK:             aie.use_lock(%[[VAL_9]], Release, %[[VAL_16]])
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb2:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_18:.*]] = aie.memtile_dma(%[[VAL_1]]) {
// CHECK:             %[[VAL_19:.*]] = arith.constant 1 : i32
// CHECK:             %[[VAL_20:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[VAL_3]], AcquireGreaterEqual, %[[VAL_19]])
// CHECK:             aie.dma_bd(%[[VAL_2]] : memref<32xi32> offset = 0 len = 16)
// CHECK:             aie.use_lock(%[[VAL_4]], Release, %[[VAL_19]])
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb2:
// CHECK:             %[[VAL_21:.*]] = aie.dma_start(S2MM, 1, ^bb3, ^bb4)
// CHECK:           ^bb3:
// CHECK:             aie.use_lock(%[[VAL_5]], AcquireGreaterEqual, %[[VAL_19]])
// CHECK:             aie.dma_bd(%[[VAL_2]] : memref<32xi32> offset = 16 len = 16)
// CHECK:             aie.use_lock(%[[VAL_6]], Release, %[[VAL_19]])
// CHECK:             aie.next_bd ^bb3
// CHECK:           ^bb4:
// CHECK:             %[[VAL_22:.*]] = aie.dma_start(MM2S, 0, ^bb5, ^bb7)
// CHECK:           ^bb5:
// CHECK:             aie.use_lock(%[[VAL_4]], AcquireGreaterEqual, %[[VAL_19]])
// CHECK:             aie.dma_bd(%[[VAL_2]] : memref<32xi32> offset = 0 len = 16)
// CHECK:             aie.use_lock(%[[VAL_3]], Release, %[[VAL_19]])
// CHECK:             aie.next_bd ^bb6
// CHECK:           ^bb6:
// CHECK:             aie.use_lock(%[[VAL_6]], AcquireGreaterEqual, %[[VAL_19]])
// CHECK:             aie.dma_bd(%[[VAL_2]] : memref<32xi32> offset = 16 len = 16)
// CHECK:             aie.use_lock(%[[VAL_5]], Release, %[[VAL_19]])
// CHECK:             aie.next_bd ^bb5
// CHECK:           ^bb7:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_23:.*]] = aie.mem(%[[VAL_11]]) {
// CHECK:             %[[VAL_24:.*]] = arith.constant 1 : i32
// CHECK:             %[[VAL_25:.*]] = aie.dma_start(MM2S, 0, ^bb1, ^bb2, repeat_count = 2)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[VAL_14]], AcquireGreaterEqual, %[[VAL_24]])
// CHECK:             aie.dma_bd(%[[VAL_12]] : memref<16xi32> offset = 0 len = 16)
// CHECK:             aie.use_lock(%[[VAL_13]], Release, %[[VAL_24]])
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb2:
// CHECK:             aie.end
// CHECK:           }
// CHECK:         }

module @memtileRepeat {
 aie.device(npu1) {
    %tile10 = aie.tile(1, 0)
    %tile11 = aie.tile(1, 1)
    %tile12 = aie.tile(1, 2)
    %tile33 = aie.tile(3, 3)

    aie.objectfifo @of0 (%tile12, {%tile11}, 1 : i32) {repeat_count = 3 : i32} : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @of1 (%tile33, {%tile11}, 1 : i32) {repeat_count = 3 : i32} : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @of2 (%tile11, {%tile10}, 1 : i32) : !aie.objectfifo<memref<32xi32>>
    aie.objectfifo.link [@of0, @of1] -> [@of2] ([0, 16] [])
 }
}
