//===- shim_to_stream_AIE2.mlir --------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform %s | FileCheck %s

// CHECK: module @shim_to_stream_AIE2 {
// CHECK:   aie.device(xcve2302) {
// CHECK:     %shim_noc_tile_2_0 = aie.tile(2, 0)
// CHECK:     %tile_3_3 = aie.tile(3, 3)
// CHECK:     %of_stream_prod_lock_0 = aie.lock(%shim_noc_tile_2_0, 0) {init = 1 : i32, sym_name = "of_stream_prod_lock_0"}
// CHECK:     %of_stream_cons_lock_0 = aie.lock(%shim_noc_tile_2_0, 1) {init = 0 : i32, sym_name = "of_stream_cons_lock_0"}
// CHECK:     aie.flow(%shim_noc_tile_2_0, DMA : 0, %tile_3_3, Core : 0)
// CHECK:     %ext_buffer_in = aie.external_buffer {sym_name = "ext_buffer_in"} : memref<16xi32>
// CHECK:     %shim_dma_2_0 = aie.shim_dma(%shim_noc_tile_2_0) {
// CHECK:       %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb2)
// CHECK:     ^bb1:  // 2 preds: ^bb0, ^bb1
// CHECK:       aie.use_lock(%of_stream_cons_lock_0, AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%ext_buffer_in : memref<16xi32>, 0, 16)
// CHECK:       aie.use_lock(%of_stream_prod_lock_0, Release, 1)
// CHECK:       aie.next_bd ^bb1
// CHECK:     ^bb2:  // pred: ^bb0
// CHECK:       aie.end
// CHECK:     }
// CHECK:     aie.shim_dma_allocation @of_stream_shim_alloc(%shim_noc_tile_2_0, MM2S, 0)
// CHECK:   }
// CHECK: }

module @shim_to_stream_AIE2 {
 aie.device(xcve2302) {
    %tile20 = aie.tile(2, 0)
    %tile33 = aie.tile(3, 3)

    aie.objectfifo @of_stream (%tile20, {%tile33}, 2 : i32) {aie_stream = 1 : i32, aie_stream_port = 0 : i32} : !aie.objectfifo<memref<16xi32>>

    %ext_buffer_in = aie.external_buffer {sym_name = "ext_buffer_in"}: memref<16xi32>
    aie.objectfifo.register_external_buffers @of_stream (%tile20, {%ext_buffer_in}) : (memref<16xi32>)
  }
}
