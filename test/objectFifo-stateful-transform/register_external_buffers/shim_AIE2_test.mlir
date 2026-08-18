//===- shim_AIE2_test.mlir --------------------------*- MLIR -*-===//
//
// Copyright (C) 2023 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Date: July 3rd 2023
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform="skip-verify=true" --aie-objectFifo-unroll %s | FileCheck %s

// CHECK-LABEL:   aie.device(xcve2302) {
// CHECK:           %[[VAL_0:.*]] = aie.tile(2, 2)
// CHECK:           %[[VAL_1:.*]] = aie.buffer(%[[VAL_0]]) {sym_name = "of_in_cons_buff_0"} : memref<16xi32>
// CHECK:           %[[VAL_2:.*]] = aie.buffer(%[[VAL_0]]) {sym_name = "of_in_cons_buff_1"} : memref<16xi32>
// CHECK:           %[[VAL_3:.*]] = aie.lock(%[[VAL_0]]) {init = 2 : i32, sym_name = "of_in_cons_prod_lock_0"}
// CHECK:           %[[VAL_4:.*]] = aie.lock(%[[VAL_0]]) {init = 0 : i32, sym_name = "of_in_cons_cons_lock_0"}
// CHECK:           %[[VAL_5:.*]] = aie.buffer(%[[VAL_0]]) {sym_name = "of_out_buff_0"} : memref<16xi32>
// CHECK:           %[[VAL_6:.*]] = aie.buffer(%[[VAL_0]]) {sym_name = "of_out_buff_1"} : memref<16xi32>
// CHECK:           %[[VAL_7:.*]] = aie.lock(%[[VAL_0]]) {init = 2 : i32, sym_name = "of_out_prod_lock_0"}
// CHECK:           %[[VAL_8:.*]] = aie.lock(%[[VAL_0]]) {init = 0 : i32, sym_name = "of_out_cons_lock_0"}
// CHECK:           %[[VAL_9:.*]] = aie.tile(2, 0)
// CHECK:           %[[VAL_10:.*]] = aie.lock(%[[VAL_9]]) {init = 1 : i32, sym_name = "of_in_prod_lock_0"}
// CHECK:           %[[VAL_11:.*]] = aie.lock(%[[VAL_9]]) {init = 0 : i32, sym_name = "of_in_cons_lock_0"}
// CHECK:           %[[VAL_12:.*]] = aie.lock(%[[VAL_9]]) {init = 1 : i32, sym_name = "of_out_cons_prod_lock_0"}
// CHECK:           %[[VAL_13:.*]] = aie.lock(%[[VAL_9]]) {init = 0 : i32, sym_name = "of_out_cons_cons_lock_0"}
// CHECK:           aie.flow(%[[VAL_9]], DMA : 0, %[[VAL_0]], DMA : 0)
// CHECK:           aie.flow(%[[VAL_0]], DMA : 0, %[[VAL_9]], DMA : 0)
// CHECK:           %[[VAL_14:.*]] = aie.external_buffer {sym_name = "ext_buffer_in"} : memref<64xi32>
// CHECK:           %[[VAL_15:.*]] = aie.external_buffer {sym_name = "ext_buffer_out"} : memref<64xi32>
// CHECK:           aie.shim_dma_allocation @of_in_shim_alloc(%[[VAL_9]], MM2S, 0)
// CHECK:           aie.shim_dma_allocation @of_out_shim_alloc(%[[VAL_9]], S2MM, 0)
// CHECK:           %[[VAL_16:.*]] = aie.shim_dma(%[[VAL_9]]) {
// CHECK:             %[[VAL_17:.*]] = arith.constant 1 : i32
// CHECK:             %[[VAL_18:.*]] = aie.dma_start(MM2S, 0, ^bb1, ^bb2)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[VAL_11]], AcquireGreaterEqual, %[[VAL_17]])
// CHECK:             aie.dma_bd(%[[VAL_14]] : memref<64xi32> offset = 0 len = 64)
// CHECK:             aie.use_lock(%[[VAL_10]], Release, %[[VAL_17]])
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb2:
// CHECK:             %[[VAL_19:.*]] = aie.dma_start(S2MM, 0, ^bb3, ^bb4)
// CHECK:           ^bb3:
// CHECK:             aie.use_lock(%[[VAL_12]], AcquireGreaterEqual, %[[VAL_17]])
// CHECK:             aie.dma_bd(%[[VAL_15]] : memref<64xi32> offset = 0 len = 64)
// CHECK:             aie.use_lock(%[[VAL_13]], Release, %[[VAL_17]])
// CHECK:             aie.next_bd ^bb3
// CHECK:           ^bb4:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_20:.*]] = aie.mem(%[[VAL_0]]) {
// CHECK:             %[[VAL_21:.*]] = arith.constant 1 : i32
// CHECK:             %[[VAL_22:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[VAL_3]], AcquireGreaterEqual, %[[VAL_21]])
// CHECK:             aie.dma_bd(%[[VAL_1]] : memref<16xi32> offset = 0 len = 16)
// CHECK:             aie.use_lock(%[[VAL_4]], Release, %[[VAL_21]])
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[VAL_3]], AcquireGreaterEqual, %[[VAL_21]])
// CHECK:             aie.dma_bd(%[[VAL_2]] : memref<16xi32> offset = 0 len = 16)
// CHECK:             aie.use_lock(%[[VAL_4]], Release, %[[VAL_21]])
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:
// CHECK:             %[[VAL_23:.*]] = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
// CHECK:           ^bb4:
// CHECK:             aie.use_lock(%[[VAL_8]], AcquireGreaterEqual, %[[VAL_21]])
// CHECK:             aie.dma_bd(%[[VAL_5]] : memref<16xi32> offset = 0 len = 16)
// CHECK:             aie.use_lock(%[[VAL_7]], Release, %[[VAL_21]])
// CHECK:             aie.next_bd ^bb5
// CHECK:           ^bb5:
// CHECK:             aie.use_lock(%[[VAL_8]], AcquireGreaterEqual, %[[VAL_21]])
// CHECK:             aie.dma_bd(%[[VAL_6]] : memref<16xi32> offset = 0 len = 16)
// CHECK:             aie.use_lock(%[[VAL_7]], Release, %[[VAL_21]])
// CHECK:             aie.next_bd ^bb4
// CHECK:           ^bb6:
// CHECK:             aie.end
// CHECK:           }
// CHECK:         }

module @shim_AIE2 {
   aie.device(xcve2302) {
      %tile22 = aie.tile(2, 2)
      %tile20 = aie.tile(2, 0)

      aie.objectfifo @of_in (%tile20, {%tile22}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
      aie.objectfifo @of_out (%tile22, {%tile20}, 2 : i32) : !aie.objectfifo<memref<16xi32>>

      %ext_buffer_in  = aie.external_buffer {sym_name = "ext_buffer_in"}: memref<64xi32>
      %ext_buffer_out  = aie.external_buffer {sym_name = "ext_buffer_out"}: memref<64xi32>
      aie.objectfifo.register_external_buffers @of_in (%tile20, {%ext_buffer_in}) : (memref<64xi32>)
      aie.objectfifo.register_external_buffers @of_out (%tile20, {%ext_buffer_out}) : (memref<64xi32>)
   }
}
