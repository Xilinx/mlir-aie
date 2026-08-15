//===- pass_flag_test.mlir --------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform="packet-sw-objFifos=true" --aie-objectFifo-unroll %s | FileCheck %s

// CHECK-LABEL:   aie.device(xcve2302) {
// CHECK:           %[[VAL_0:.*]] = aie.tile(1, 2)
// CHECK:           %[[VAL_1:.*]] = aie.buffer(%[[VAL_0]]) {sym_name = "of1_buff_0"} : memref<16xi32>
// CHECK:           %[[VAL_2:.*]] = aie.buffer(%[[VAL_0]]) {sym_name = "of1_buff_1"} : memref<16xi32>
// CHECK:           %[[VAL_3:.*]] = aie.lock(%[[VAL_0]]) {init = 2 : i32, sym_name = "of1_prod_lock_0"}
// CHECK:           %[[VAL_4:.*]] = aie.lock(%[[VAL_0]]) {init = 0 : i32, sym_name = "of1_cons_lock_0"}
// CHECK:           %[[VAL_5:.*]] = aie.tile(3, 3)
// CHECK:           %[[VAL_6:.*]] = aie.buffer(%[[VAL_5]]) {sym_name = "of1_cons_buff_0"} : memref<16xi32>
// CHECK:           %[[VAL_7:.*]] = aie.buffer(%[[VAL_5]]) {sym_name = "of1_cons_buff_1"} : memref<16xi32>
// CHECK:           %[[VAL_8:.*]] = aie.lock(%[[VAL_5]]) {init = 2 : i32, sym_name = "of1_cons_prod_lock_0"}
// CHECK:           %[[VAL_9:.*]] = aie.lock(%[[VAL_5]]) {init = 0 : i32, sym_name = "of1_cons_cons_lock_0"}
// CHECK:           aie.packet_flow(0) {
// CHECK:             aie.packet_source<%[[VAL_0]], DMA : 0>
// CHECK:             aie.packet_dest<%[[VAL_5]], DMA : 0>
// CHECK:           }
// CHECK:           %[[VAL_10:.*]] = aie.mem(%[[VAL_0]]) {
// CHECK:             %[[VAL_11:.*]] = arith.constant 1 : i32
// CHECK:             %[[VAL_12:.*]] = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[VAL_4]], AcquireGreaterEqual, %[[VAL_11]])
// CHECK:             aie.dma_bd_packet(0, 0)
// CHECK:             aie.dma_bd(%[[VAL_1]] : memref<16xi32> offset = 0 len = 16)
// CHECK:             aie.use_lock(%[[VAL_3]], Release, %[[VAL_11]])
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[VAL_4]], AcquireGreaterEqual, %[[VAL_11]])
// CHECK:             aie.dma_bd_packet(0, 0)
// CHECK:             aie.dma_bd(%[[VAL_2]] : memref<16xi32> offset = 0 len = 16)
// CHECK:             aie.use_lock(%[[VAL_3]], Release, %[[VAL_11]])
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_13:.*]] = aie.mem(%[[VAL_5]]) {
// CHECK:             %[[VAL_14:.*]] = arith.constant 1 : i32
// CHECK:             %[[VAL_15:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[VAL_8]], AcquireGreaterEqual, %[[VAL_14]])
// CHECK:             aie.dma_bd(%[[VAL_6]] : memref<16xi32> offset = 0 len = 16)
// CHECK:             aie.use_lock(%[[VAL_9]], Release, %[[VAL_14]])
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[VAL_8]], AcquireGreaterEqual, %[[VAL_14]])
// CHECK:             aie.dma_bd(%[[VAL_7]] : memref<16xi32> offset = 0 len = 16)
// CHECK:             aie.use_lock(%[[VAL_9]], Release, %[[VAL_14]])
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:
// CHECK:             aie.end
// CHECK:           }
// CHECK:         }

module @pass_flag {
 aie.device(xcve2302) {
    %tile12 = aie.tile(1, 2)
    %tile33 = aie.tile(3, 3)

    aie.objectfifo @of1 (%tile12, {%tile33}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
 }
}
