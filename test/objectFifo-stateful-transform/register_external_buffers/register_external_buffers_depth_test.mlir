//===- register_external_buffers_depth_test.mlir ----------------*- MLIR -*-===//
//
// Copyright (C) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform="skip-verify=true" --aie-objectFifo-unroll %s | FileCheck %s

// CHECK-LABEL:   aie.device(xcvc1902) {
// CHECK:           %[[VAL_0:.*]] = aie.tile(7, 1)
// CHECK:           %[[VAL_1:.*]] = aie.buffer(%[[VAL_0]]) {sym_name = "ext_of_cons_buff_0"} : memref<16xi32>
// CHECK:           %[[VAL_2:.*]] = aie.lock(%[[VAL_0]]) {init = 0 : i32, sym_name = "ext_of_cons_lock_0"}
// CHECK:           %[[VAL_3:.*]] = aie.tile(7, 0)
// CHECK:           %[[VAL_4:.*]] = aie.lock(%[[VAL_3]]) {init = 0 : i32, sym_name = "ext_of_lock_0"}
// CHECK:           %[[VAL_5:.*]] = aie.lock(%[[VAL_3]]) {init = 0 : i32, sym_name = "ext_of_lock_1"}
// CHECK:           aie.flow(%[[VAL_3]], DMA : 0, %[[VAL_0]], DMA : 0)
// CHECK:           %[[VAL_6:.*]] = aie.external_buffer {sym_name = "ext_buffer_in0"} : memref<64xi32>
// CHECK:           %[[VAL_7:.*]] = aie.external_buffer {sym_name = "ext_buffer_in1"} : memref<64xi32>
// CHECK:           aie.shim_dma_allocation @ext_of_shim_alloc(%[[VAL_3]], MM2S, 0)
// CHECK:           %[[VAL_8:.*]] = aie.shim_dma(%[[VAL_3]]) {
// CHECK:             %[[VAL_9:.*]] = arith.constant 1 : i32
// CHECK:             %[[VAL_10:.*]] = arith.constant 0 : i32
// CHECK:             %[[VAL_11:.*]] = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[VAL_4]], Acquire, %[[VAL_9]])
// CHECK:             aie.dma_bd(%[[VAL_6]] : memref<64xi32> offset = 0 len = 64)
// CHECK:             aie.use_lock(%[[VAL_4]], Release, %[[VAL_10]])
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[VAL_5]], Acquire, %[[VAL_9]])
// CHECK:             aie.dma_bd(%[[VAL_7]] : memref<64xi32> offset = 0 len = 64)
// CHECK:             aie.use_lock(%[[VAL_5]], Release, %[[VAL_10]])
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_12:.*]] = aie.mem(%[[VAL_0]]) {
// CHECK:             %[[VAL_13:.*]] = arith.constant 0 : i32
// CHECK:             %[[VAL_14:.*]] = arith.constant 1 : i32
// CHECK:             %[[VAL_15:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[VAL_2]], Acquire, %[[VAL_13]])
// CHECK:             aie.dma_bd(%[[VAL_1]] : memref<16xi32> offset = 0 len = 16)
// CHECK:             aie.use_lock(%[[VAL_2]], Release, %[[VAL_14]])
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb2:
// CHECK:             aie.end
// CHECK:           }
// CHECK:         }

module @register_external_buffers_depth {
 aie.device(xcvc1902) {
    %tile71 = aie.tile(7, 1)
    %tile70 = aie.tile(7, 0)

    aie.objectfifo @ext_of (%tile70, {%tile71}, 1 : i32) : !aie.objectfifo<memref<16xi32>>

    %ext_buffer_in0 = aie.external_buffer {sym_name = "ext_buffer_in0"}: memref<64xi32>
    %ext_buffer_in1 = aie.external_buffer {sym_name = "ext_buffer_in1"}: memref<64xi32>
    aie.objectfifo.register_external_buffers @ext_of (%tile70, {%ext_buffer_in0, %ext_buffer_in1}) : (memref<64xi32>, memref<64xi32>)
 }
}
