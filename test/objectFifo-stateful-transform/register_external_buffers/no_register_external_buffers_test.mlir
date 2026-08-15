//===- no_register_external_buffers_test.mlir -------------------*- MLIR -*-===//
//
// Copyright (C) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform --aie-objectFifo-unroll %s | FileCheck %s

// CHECK-LABEL:   aie.device(xcvc1902) {
// CHECK:           %[[VAL_0:.*]] = aie.tile(7, 2)
// CHECK:           %[[VAL_1:.*]] = aie.buffer(%[[VAL_0]]) {sym_name = "ext_of_cons_buff_0"} : memref<16xi32>
// CHECK:           %[[VAL_2:.*]] = aie.buffer(%[[VAL_0]]) {sym_name = "ext_of_cons_buff_1"} : memref<16xi32>
// CHECK:           %[[VAL_3:.*]] = aie.buffer(%[[VAL_0]]) {sym_name = "ext_of_cons_buff_2"} : memref<16xi32>
// CHECK:           %[[VAL_4:.*]] = aie.lock(%[[VAL_0]]) {init = 0 : i32, sym_name = "ext_of_cons_lock_0"}
// CHECK:           %[[VAL_5:.*]] = aie.lock(%[[VAL_0]]) {init = 0 : i32, sym_name = "ext_of_cons_lock_1"}
// CHECK:           %[[VAL_6:.*]] = aie.lock(%[[VAL_0]]) {init = 0 : i32, sym_name = "ext_of_cons_lock_2"}
// CHECK:           %[[VAL_7:.*]] = aie.tile(7, 0)
// CHECK:           aie.flow(%[[VAL_7]], DMA : 0, %[[VAL_0]], DMA : 0)
// CHECK:           aie.shim_dma_allocation @ext_of_shim_alloc(%[[VAL_7]], MM2S, 0)
// CHECK:           %[[VAL_8:.*]] = aie.mem(%[[VAL_0]]) {
// CHECK:             %[[VAL_9:.*]] = arith.constant 0 : i32
// CHECK:             %[[VAL_10:.*]] = arith.constant 1 : i32
// CHECK:             %[[VAL_11:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb4)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[VAL_4]], Acquire, %[[VAL_9]])
// CHECK:             aie.dma_bd(%[[VAL_1]] : memref<16xi32> offset = 0 len = 16)
// CHECK:             aie.use_lock(%[[VAL_4]], Release, %[[VAL_10]])
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[VAL_5]], Acquire, %[[VAL_9]])
// CHECK:             aie.dma_bd(%[[VAL_2]] : memref<16xi32> offset = 0 len = 16)
// CHECK:             aie.use_lock(%[[VAL_5]], Release, %[[VAL_10]])
// CHECK:             aie.next_bd ^bb3
// CHECK:           ^bb3:
// CHECK:             aie.use_lock(%[[VAL_6]], Acquire, %[[VAL_9]])
// CHECK:             aie.dma_bd(%[[VAL_3]] : memref<16xi32> offset = 0 len = 16)
// CHECK:             aie.use_lock(%[[VAL_6]], Release, %[[VAL_10]])
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb4:
// CHECK:             aie.end
// CHECK:           }
// CHECK:         }

module @no_register_external_buffers {
 aie.device(xcvc1902) {
    %tile72 = aie.tile(7, 2)
    %tile70 = aie.tile(7, 0)

    aie.objectfifo @ext_of (%tile70, {%tile72}, 3 : i32) : !aie.objectfifo<memref<16xi32>>
 }
}
