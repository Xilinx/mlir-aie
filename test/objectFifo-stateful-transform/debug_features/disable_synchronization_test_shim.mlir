//===- disable_synchronization_test_shim.mlir ------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform %s | FileCheck %s

// CHECK: module @disable_sync {
// CHECK:   aie.device(xcve2302) {
// CHECK:     %{{.*}}tile_2_0 = aie.tile(2, 0)
// CHECK:     %{{.*}}tile_1_3 = aie.tile(1, 3)
// CHECK:     %[[VAL_0:.*]] = aie.buffer(%{{.*}}tile_1_3) {sym_name = "of0_cons_buff_0"} : memref<16xi32> 
// CHECK:     %[[VAL_1:.*]] = aie.buffer(%{{.*}}tile_1_3) {sym_name = "of0_cons_buff_1"} : memref<16xi32> 
// CHECK:     aie.flow(%{{.*}}tile_2_0, DMA : 0, %{{.*}}tile_1_3, DMA : 0)
// CHECK:     %ext_buffer_in = aie.external_buffer {sym_name = "ext_buffer_in"} : memref<64xi32>
// CHECK:     %shim_dma_2_0 = aie.shim_dma(%{{.*}}tile_2_0) {
// CHECK:       %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb2)
// CHECK:     ^bb1:  // 2 preds: ^bb0, ^bb1
// CHECK:       aie.dma_bd(%ext_buffer_in : memref<64xi32>, 0, 64)
// CHECK:       aie.next_bd ^bb1
// CHECK:     ^bb2:  // pred: ^bb0
// CHECK:       aie.end
// CHECK:     }
// CHECK:     aie.shim_dma_allocation @of0_shim_alloc(%shim_noc_tile_2_0, MM2S, 0)
// CHECK:     %mem_1_3 = aie.mem(%{{.*}}tile_1_3) {
// CHECK:       %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:     ^bb1:  // 2 preds: ^bb0, ^bb2
// CHECK:       aie.dma_bd(%[[VAL_0]] : memref<16xi32>, 0, 16)
// CHECK:       aie.next_bd ^bb2
// CHECK:     ^bb2:  // pred: ^bb1
// CHECK:       aie.dma_bd(%[[VAL_1]] : memref<16xi32>, 0, 16)
// CHECK:       aie.next_bd ^bb1
// CHECK:     ^bb3:  // pred: ^bb0
// CHECK:       aie.end
// CHECK:     }
// CHECK:   }
// CHECK: }

module @disable_sync {
 aie.device(xcve2302) {
    %tile20 = aie.tile(2, 0)
    %tile13 = aie.tile(1, 3)
   
    aie.objectfifo @of0 (%tile20, {%tile13}, 2 : i32) { disable_synchronization = true } : !aie.objectfifo<memref<16xi32>>
    
    %ext_buffer_in = aie.external_buffer {sym_name = "ext_buffer_in"}: memref<64xi32>
    aie.objectfifo.register_external_buffers @of0 (%tile20, {%ext_buffer_in}) : (memref<64xi32>)
 }
}
