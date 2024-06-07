//===- link_test_AIE1.mlir --------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2023, Advanced Micro Devices, Inc.
//
// Date: May 9th 2023
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform -verify-diagnostics %s

//CHECK:     error: unexpected warning: Data type mismatch in objectFifoLink

//CHECK-LABEL: aie.device(xcvc1902) {
//CHECK:     memref.global "public" @of2_cons : memref<16xi32>
//CHECK:     memref.global "public" @of2 : memref<16xi32>
//CHECK:     memref.global "public" @of1_cons : memref<16xbf16>
//CHECK:     memref.global "public" @of1 : memref<16xbf16>
//CHECK:     %tile_2_0 = aie.tile(2, 0)
//CHECK:     %tile_1_2 = aie.tile(1, 2)
//CHECK:     %tile_2_2 = aie.tile(2, 2)
//CHECK:     %of2_cons_buff_0 = aie.buffer(%tile_2_2) {sym_name = "of2_cons_buff_0"} : memref<16xi32> 
//CHECK:     %of2_cons_buff_1 = aie.buffer(%tile_2_2) {sym_name = "of2_cons_buff_1"} : memref<16xi32> 
//CHECK:     %of2_cons_lock_0 = aie.lock(%tile_2_2, 0) {init = 0 : i32, sym_name = "of2_cons_lock_0"}
//CHECK:     %of2_cons_lock_1 = aie.lock(%tile_2_2, 1) {init = 0 : i32, sym_name = "of2_cons_lock_1"}
//CHECK:     %of1_cons_buff_0 = aie.buffer(%tile_1_2) {sym_name = "of1_cons_buff_0"} : memref<16xbf16> 
//CHECK:     %of1_cons_buff_1 = aie.buffer(%tile_1_2) {sym_name = "of1_cons_buff_1"} : memref<16xbf16> 
//CHECK:     %of1_cons_lock_0 = aie.lock(%tile_1_2, 0) {init = 0 : i32, sym_name = "of1_cons_lock_0"}
//CHECK:     %of1_cons_lock_1 = aie.lock(%tile_1_2, 1) {init = 0 : i32, sym_name = "of1_cons_lock_1"}
//CHECK:     %of1_lock_0 = aie.lock(%tile_2_0, 0) {init = 0 : i32, sym_name = "of1_lock_0"}
//CHECK:     aie.flow(%tile_2_0, DMA : 0, %tile_1_2, DMA : 0)
//CHECK:     aie.flow(%tile_1_2, DMA : 0, %tile_2_2, DMA : 0)
//CHECK:     %ext_buff_in = aie.external_buffer {sym_name = "ext_buff_in"} : memref<16xi32>
//CHECK:     aie.shim_dma_allocation @of1(MM2S, 0, 2)
//CHECK:     %shim_dma_2_0 = aie.shim_dma(%tile_2_0) {
//CHECK:       %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb2)
//CHECK:     ^bb1:  // 2 preds: ^bb0, ^bb1
//CHECK:       aie.use_lock(%of1_lock_0, Acquire, 1)
//CHECK:       aie.dma_bd(%ext_buff_in : memref<16xi32>, 0, 16)
//CHECK:       aie.use_lock(%of1_lock_0, Release, 0)
//CHECK:       aie.next_bd ^bb1
//CHECK:     ^bb2:  // pred: ^bb0
//CHECK:       aie.end
//CHECK:    }
//CHECK:     %mem_1_2 = aie.mem(%tile_1_2) {
//CHECK:       %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
//CHECK:     ^bb1:  // 2 preds: ^bb0, ^bb2
//CHECK:       aie.use_lock(%of1_cons_lock_0, Acquire, 0)
//CHECK:       aie.dma_bd(%of1_cons_buff_0 : memref<16xbf16>, 0, 16)
//CHECK:       aie.use_lock(%of1_cons_lock_0, Release, 1)
//CHECK:       aie.next_bd ^bb2
//CHECK:     ^bb2:  // pred: ^bb1
//CHECK:       aie.use_lock(%of1_cons_lock_1, Acquire, 0)
//CHECK:      aie.dma_bd(%of1_cons_buff_1 : memref<16xbf16>, 0, 16)
//CHECK:       aie.use_lock(%of1_cons_lock_1, Release, 1)
//CHECK:       aie.next_bd ^bb1
//CHECK:     ^bb3:  // pred: ^bb0
//CHECK:       %1 = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
//CHECK:     ^bb4:  // 2 preds: ^bb3, ^bb5
//CHECK:       aie.use_lock(%of1_cons_lock_0, Acquire, 1)
//CHECK:       aie.dma_bd(%of1_cons_buff_0 : memref<16xbf16>, 0, 16)
//CHECK:       aie.use_lock(%of1_cons_lock_0, Release, 0)
//CHECK:       aie.next_bd ^bb5
//CHECK:     ^bb5:  // pred: ^bb4
//CHECK:       aie.use_lock(%of1_cons_lock_1, Acquire, 1)
//CHECK:       aie.dma_bd(%of1_cons_buff_1 : memref<16xbf16>, 0, 16)
//CHECK:       aie.use_lock(%of1_cons_lock_1, Release, 0)
//CHECK:       aie.next_bd ^bb4
//CHECK:     ^bb6:  // pred: ^bb3
//CHECK:       aie.end
//CHECK:     }
//CHECK:     %mem_2_2 = aie.mem(%tile_2_2) {
//CHECK:       %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
//CHECK:     ^bb1:  // 2 preds: ^bb0, ^bb2
//CHECK:       aie.use_lock(%of2_cons_lock_0, Acquire, 0)
//CHECK:       aie.dma_bd(%of2_cons_buff_0 : memref<16xi32>, 0, 16)
//CHECK:       aie.use_lock(%of2_cons_lock_0, Release, 1)
//CHECK:       aie.next_bd ^bb2
//CHECK:     ^bb2:  // pred: ^bb1
//CHECK:       aie.use_lock(%of2_cons_lock_1, Acquire, 0)
//CHECK:       aie.dma_bd(%of2_cons_buff_1 : memref<16xi32>, 0, 16)
//CHECK:       aie.use_lock(%of2_cons_lock_1, Release, 1)
//CHECK:       aie.next_bd ^bb1
//CHECK:     ^bb3:  // pred: ^bb0
//CHECK:       aie.end
//CHECK:     }
//CHECK:   }


module @link_AIE1 {
    aie.device(xcvc1902) {
        %tile20 = aie.tile(2, 0)
        %tile12 = aie.tile(1, 2)
        %tile22 = aie.tile(2, 2)

        aie.objectfifo @of1 (%tile20, {%tile12}, 2 : i32) : !aie.objectfifo<memref<16xbf16>>
        aie.objectfifo @of2 (%tile12, {%tile22}, 2 : i32) : !aie.objectfifo<memref<16xi32>>

        aie.objectfifo.link [@of1] -> [@of2] ()

        %ext_buff_in = aie.external_buffer {sym_name = "ext_buff_in"} : memref<16xi32>
        aie.objectfifo.register_external_buffers @of1 (%tile20, {%ext_buff_in}) : (memref<16xi32>)
    }
}
