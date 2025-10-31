//===- register_external_buffers_depth_test.mlir ----------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform %s | FileCheck %s

// CHECK-LABEL:   aie.device(xcvc1902) {
// CHECK:               %{{.*}}tile_7_1 = aie.tile(7, 1)
// CHECK:               %{{.*}}tile_7_0 = aie.tile(7, 0)
// CHECK:               %[[VAL_0:.*]] = aie.buffer(%{{.*}}tile_7_1) {sym_name = "ext_of_cons_buff_0"} : memref<16xi32> 
// CHECK:               %[[VAL_1:.*]] = aie.lock(%{{.*}}tile_7_1, 0) {init = 0 : i32, sym_name = "ext_of_cons_lock_0"}
// CHECK:               %[[VAL_2:.*]] = aie.lock(%{{.*}}tile_7_0, 0) {init = 0 : i32, sym_name = "ext_of_lock_0"}
// CHECK:               %[[VAL_3:.*]] = aie.lock(%{{.*}}tile_7_0, 1) {init = 0 : i32, sym_name = "ext_of_lock_1"}
// CHECK:               aie.flow(%{{.*}}tile_7_0, DMA : 0, %{{.*}}tile_7_1, DMA : 0)
// CHECK:               %ext_buffer_in0 = aie.external_buffer {sym_name = "ext_buffer_in0"} : memref<64xi32>
// CHECK:               %ext_buffer_in1 = aie.external_buffer {sym_name = "ext_buffer_in1"} : memref<64xi32>
// CHECK:               %shim_dma_7_0 = aie.shim_dma(%{{.*}}tile_7_0) {
// CHECK:                 %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
// CHECK:               ^bb1:  // 2 preds: ^bb0, ^bb2
// CHECK:                 aie.use_lock(%[[VAL_2]], Acquire, 1)
// CHECK:                 aie.dma_bd(%ext_buffer_in0 : memref<64xi32>, 0, 64)
// CHECK:                 aie.use_lock(%[[VAL_2]], Release, 0)
// CHECK:                 aie.next_bd ^bb2
// CHECK:               ^bb2:  // pred: ^bb1
// CHECK:                 aie.use_lock(%[[VAL_3]], Acquire, 1)
// CHECK:                 aie.dma_bd(%ext_buffer_in1 : memref<64xi32>, 0, 64)
// CHECK:                 aie.use_lock(%[[VAL_3]], Release, 0)
// CHECK:                 aie.next_bd ^bb1
// CHECK:               ^bb3:  // pred: ^bb0
// CHECK:                 aie.end
// CHECK:               }
// CHECK:               aie.shim_dma_allocation @ext_of_shim_alloc(MM2S, 0, 7)
// CHECK:               %mem_7_1 = aie.mem(%{{.*}}tile_7_1) {
// CHECK:                 %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
// CHECK:               ^bb1:  // 2 preds: ^bb0, ^bb1
// CHECK:                 aie.use_lock(%[[VAL_1]], Acquire, 0)
// CHECK:                 aie.dma_bd(%[[VAL_0]] : memref<16xi32>, 0, 16)
// CHECK:                 aie.use_lock(%[[VAL_1]], Release, 1)
// CHECK:                 aie.next_bd ^bb1
// CHECK:               ^bb2:  // pred: ^bb0
// CHECK:                 aie.end
// CHECK:               }
// CHECK:             }

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
