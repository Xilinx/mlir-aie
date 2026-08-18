//===- disable_synchronization_test_AIE1.mlir -------------------*- MLIR -*-===//
//
// Copyright (C) 2024 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform="skip-verify=true" --aie-objectFifo-unroll %s | FileCheck %s

// CHECK-LABEL:   aie.device(xcvc1902) {
// CHECK:           %[[VAL_0:.*]] = aie.tile(1, 2)
// CHECK:           %[[VAL_1:.*]] = aie.buffer(%[[VAL_0]]) {sym_name = "of0_buff_0"} : memref<16xi32>
// CHECK:           %[[VAL_2:.*]] = aie.buffer(%[[VAL_0]]) {sym_name = "of1_buff_0"} : memref<16xi32>
// CHECK:           %[[VAL_3:.*]] = aie.buffer(%[[VAL_0]]) {sym_name = "of1_buff_1"} : memref<16xi32>
// CHECK:           %[[VAL_4:.*]] = aie.tile(3, 3)
// CHECK:           %[[VAL_5:.*]] = aie.buffer(%[[VAL_4]]) {sym_name = "of1_cons_buff_0"} : memref<16xi32>
// CHECK:           %[[VAL_6:.*]] = aie.buffer(%[[VAL_4]]) {sym_name = "of1_cons_buff_1"} : memref<16xi32>
// CHECK:           aie.flow(%[[VAL_0]], DMA : 0, %[[VAL_4]], DMA : 0)
// CHECK:           %[[VAL_7:.*]] = aie.mem(%[[VAL_0]]) {
// CHECK:             %[[VAL_8:.*]] = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:
// CHECK:             aie.dma_bd(%[[VAL_2]] : memref<16xi32> offset = 0 len = 16)
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.dma_bd(%[[VAL_3]] : memref<16xi32> offset = 0 len = 16)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_9:.*]] = aie.mem(%[[VAL_4]]) {
// CHECK:             %[[VAL_10:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:
// CHECK:             aie.dma_bd(%[[VAL_5]] : memref<16xi32> offset = 0 len = 16)
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.dma_bd(%[[VAL_6]] : memref<16xi32> offset = 0 len = 16)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:
// CHECK:             aie.end
// CHECK:           }
// CHECK:         }

module @disable_sync {
 aie.device(xcvc1902) {
    %tile12 = aie.tile(1, 2)
    %tile13 = aie.tile(1, 3)
    %tile33 = aie.tile(3, 3)

    aie.objectfifo @of0 (%tile12, {%tile13}, 1 : i32) { disable_synchronization = true } : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @of1 (%tile12, {%tile33}, 2 : i32) { disable_synchronization = true } : !aie.objectfifo<memref<16xi32>>
 }
}
