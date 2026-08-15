//===- shim_to_stream_AIE2.mlir --------------------------------*- MLIR -*-===//
//
// Copyright (C) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform --aie-objectFifo-unroll %s | FileCheck %s

// CHECK-LABEL:   aie.device(xcve2302) {
// CHECK:           %[[VAL_0:.*]] = aie.tile(2, 0)
// CHECK:           %[[VAL_1:.*]] = aie.lock(%[[VAL_0]]) {init = 1 : i32, sym_name = "of_stream_prod_lock_0"}
// CHECK:           %[[VAL_2:.*]] = aie.lock(%[[VAL_0]]) {init = 0 : i32, sym_name = "of_stream_cons_lock_0"}
// CHECK:           %[[VAL_3:.*]] = aie.tile(3, 3)
// CHECK:           aie.flow(%[[VAL_0]], DMA : 0, %[[VAL_3]], Core : 0)
// CHECK:           %[[VAL_4:.*]] = aie.external_buffer {sym_name = "ext_buffer_in"} : memref<16xi32>
// CHECK:           aie.shim_dma_allocation @of_stream_shim_alloc(%[[VAL_0]], MM2S, 0)
// CHECK:           %[[VAL_5:.*]] = aie.shim_dma(%[[VAL_0]]) {
// CHECK:             %[[VAL_6:.*]] = arith.constant 1 : i32
// CHECK:             %[[VAL_7:.*]] = aie.dma_start(MM2S, 0, ^bb1, ^bb2)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[VAL_2]], AcquireGreaterEqual, %[[VAL_6]])
// CHECK:             aie.dma_bd(%[[VAL_4]] : memref<16xi32> offset = 0 len = 16)
// CHECK:             aie.use_lock(%[[VAL_1]], Release, %[[VAL_6]])
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb2:
// CHECK:             aie.end
// CHECK:           }
// CHECK:         }

module @shim_to_stream_AIE2 {
 aie.device(xcve2302) {
    %tile20 = aie.tile(2, 0)
    %tile33 = aie.tile(3, 3)

    aie.objectfifo @of_stream (%tile20, {%tile33}, 2 : i32) {aie_stream = 1 : i32, aie_stream_port = 0 : i32} : !aie.objectfifo<memref<16xi32>>

    %ext_buffer_in = aie.external_buffer {sym_name = "ext_buffer_in"}: memref<16xi32>
    aie.objectfifo.register_external_buffers @of_stream (%tile20, {%ext_buffer_in}) : (memref<16xi32>)
  }
}
