//===- packet_id_test.mlir --------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform="packet-sw-objFifos=true" --aie-objectFifo-unroll %s | FileCheck %s

// CHECK-LABEL:   aie.device(xcve2302) {
// CHECK:           %[[VAL_0:.*]] = aie.tile(0, 0)
// CHECK:           %[[VAL_1:.*]] = aie.tile(0, 2)
// CHECK:           %[[VAL_2:.*]] = aie.tile(1, 2)
// CHECK:           %[[VAL_3:.*]] = aie.buffer(%[[VAL_2]]) {sym_name = "of1_buff_0"} : memref<16xi32>
// CHECK:           %[[VAL_4:.*]] = aie.buffer(%[[VAL_2]]) {sym_name = "of1_buff_1"} : memref<16xi32>
// CHECK:           %[[VAL_5:.*]] = aie.lock(%[[VAL_2]]) {init = 2 : i32, sym_name = "of1_prod_lock_0"}
// CHECK:           %[[VAL_6:.*]] = aie.lock(%[[VAL_2]]) {init = 0 : i32, sym_name = "of1_cons_lock_0"}
// CHECK:           %[[VAL_7:.*]] = aie.tile(3, 3)
// CHECK:           %[[VAL_8:.*]] = aie.buffer(%[[VAL_7]]) {sym_name = "of1_cons_buff_0"} : memref<16xi32>
// CHECK:           %[[VAL_9:.*]] = aie.buffer(%[[VAL_7]]) {sym_name = "of1_cons_buff_1"} : memref<16xi32>
// CHECK:           %[[VAL_10:.*]] = aie.lock(%[[VAL_7]]) {init = 2 : i32, sym_name = "of1_cons_prod_lock_0"}
// CHECK:           %[[VAL_11:.*]] = aie.lock(%[[VAL_7]]) {init = 0 : i32, sym_name = "of1_cons_cons_lock_0"}
// CHECK:           aie.packet_flow(0) {
// CHECK:             aie.packet_source<%[[VAL_2]], DMA : 0>
// CHECK:             aie.packet_dest<%[[VAL_7]], DMA : 0>
// CHECK:           }
// CHECK:           aie.packet_flow(1) {
// CHECK:             aie.packet_source<%[[VAL_1]], Trace : 0>
// CHECK:             aie.packet_dest<%[[VAL_0]], DMA : 1>
// CHECK:           } {keep_pkt_header = true}
// CHECK:           %[[VAL_12:.*]] = aie.mem(%[[VAL_2]]) {
// CHECK:             %[[VAL_13:.*]] = arith.constant 1 : i32
// CHECK:             %[[VAL_14:.*]] = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[VAL_6]], AcquireGreaterEqual, %[[VAL_13]])
// CHECK:             aie.dma_bd_packet(0, 0)
// CHECK:             aie.dma_bd(%[[VAL_3]] : memref<16xi32> offset = 0 len = 16)
// CHECK:             aie.use_lock(%[[VAL_5]], Release, %[[VAL_13]])
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[VAL_6]], AcquireGreaterEqual, %[[VAL_13]])
// CHECK:             aie.dma_bd_packet(0, 0)
// CHECK:             aie.dma_bd(%[[VAL_4]] : memref<16xi32> offset = 0 len = 16)
// CHECK:             aie.use_lock(%[[VAL_5]], Release, %[[VAL_13]])
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_15:.*]] = aie.mem(%[[VAL_7]]) {
// CHECK:             %[[VAL_16:.*]] = arith.constant 1 : i32
// CHECK:             %[[VAL_17:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[VAL_10]], AcquireGreaterEqual, %[[VAL_16]])
// CHECK:             aie.dma_bd(%[[VAL_8]] : memref<16xi32> offset = 0 len = 16)
// CHECK:             aie.use_lock(%[[VAL_11]], Release, %[[VAL_16]])
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[VAL_10]], AcquireGreaterEqual, %[[VAL_16]])
// CHECK:             aie.dma_bd(%[[VAL_9]] : memref<16xi32> offset = 0 len = 16)
// CHECK:             aie.use_lock(%[[VAL_11]], Release, %[[VAL_16]])
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:
// CHECK:             aie.end
// CHECK:           }
// CHECK:         }

module @packet_id {
 aie.device(xcve2302) {
    %tile00 = aie.tile(0, 0)
    %tile02 = aie.tile(0, 2)
    %tile12 = aie.tile(1, 2)
    %tile33 = aie.tile(3, 3)

    aie.objectfifo @of1 (%tile12, {%tile33}, 2 : i32) : !aie.objectfifo<memref<16xi32>>

    aie.packet_flow(1) {
      aie.packet_source<%tile02, Trace : 0>
      aie.packet_dest<%tile00, DMA : 1>
    } {keep_pkt_header = true}
 }
}
