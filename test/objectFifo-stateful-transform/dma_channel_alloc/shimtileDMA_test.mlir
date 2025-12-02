//===- shimtileDMA_test.mlir -----------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform %s | FileCheck %s

// CHECK-LABEL:   aie.device(xcve2302) {
// CHECK:           %{{.*}}tile_2_0 = aie.tile(2, 0)
// CHECK:           %{{.*}}tile_3_3 = aie.tile(3, 3)
// CHECK:           %[[VAL_0:.*]] = aie.buffer(%{{.*}}tile_3_3) {sym_name = "objfifo_cons_buff_0"} : memref<16xi32> 
// CHECK:           %[[VAL_1:.*]] = aie.buffer(%{{.*}}tile_3_3) {sym_name = "objfifo_cons_buff_1"} : memref<16xi32> 
// CHECK:           %[[VAL_2:.*]] = aie.lock(%{{.*}}tile_3_3, 0) {init = 2 : i32, sym_name = "objfifo_cons_prod_lock_0"}
// CHECK:           %[[VAL_3:.*]] = aie.lock(%{{.*}}tile_3_3, 1) {init = 0 : i32, sym_name = "objfifo_cons_cons_lock_0"}
// CHECK:           %[[VAL_4:.*]] = aie.lock(%{{.*}}tile_2_0, 3) {init = 1 : i32, sym_name = "objfifo_prod_lock_0"}
// CHECK:           %[[VAL_5:.*]] = aie.lock(%{{.*}}tile_2_0, 4) {init = 0 : i32, sym_name = "objfifo_cons_lock_0"}
// CHECK:           %0 = aie.external_buffer : memref<16xi32>
// CHECK:           %lock_2_0 = aie.lock(%{{.*}}tile_2_0, 0)
// CHECK:           %1 = aie.external_buffer : memref<16xi32>
// CHECK:           %lock_2_0_0 = aie.lock(%{{.*}}tile_2_0, 1)
// CHECK:           %2 = aie.external_buffer : memref<16xi32>
// CHECK:           %lock_2_0_1 = aie.lock(%{{.*}}tile_2_0, 2)
// CHECK:           aie.flow(%{{.*}}tile_2_0, DMA : 0, %{{.*}}tile_3_3, DMA : 0)
// CHECK:           %ext_buffer_in = aie.external_buffer {sym_name = "ext_buffer_in"} : memref<16xi32>
// CHECK:           %shim_dma_2_0 = aie.shim_dma(%{{.*}}tile_2_0) {
// CHECK:             %3 = aie.dma_start(MM2S, 1, ^bb1, ^bb3)
// CHECK:           ^bb1:  // 2 preds: ^bb0, ^bb2
// CHECK:             aie.use_lock(%lock_2_0, Acquire, 1)
// CHECK:             aie.dma_bd(%0 : memref<16xi32>, 0, 16)
// CHECK:             aie.use_lock(%lock_2_0, Release, 0)
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:  // pred: ^bb1
// CHECK:             aie.use_lock(%lock_2_0_0, Acquire, 1)
// CHECK:             aie.dma_bd(%1 : memref<16xi32>, 0, 16)
// CHECK:             aie.use_lock(%lock_2_0_0, Release, 0)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:  // pred: ^bb0
// CHECK:             %4 = aie.dma_start(S2MM, 0, ^bb4, ^bb5)
// CHECK:           ^bb4:  // 2 preds: ^bb3, ^bb4
// CHECK:             aie.use_lock(%lock_2_0_1, Acquire, 0)
// CHECK:             aie.dma_bd(%2 : memref<16xi32>, 0, 16)
// CHECK:             aie.use_lock(%lock_2_0_1, Release, 1)
// CHECK:             aie.next_bd ^bb4
// CHECK:           ^bb5:  // pred: ^bb3
// CHECK:             %5 = aie.dma_start(MM2S, 0, ^bb6, ^bb7)
// CHECK:           ^bb6:  // 2 preds: ^bb5, ^bb6
// CHECK:             aie.use_lock(%[[VAL_5]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%ext_buffer_in : memref<16xi32>, 0, 16)
// CHECK:             aie.use_lock(%[[VAL_4]], Release, 1)
// CHECK:             aie.next_bd ^bb6
// CHECK:           ^bb7:  // pred: ^bb5
// CHECK:             aie.end
// CHECK:           }
// CHECK:           aie.shim_dma_allocation @objfifo_shim_alloc (%tile_2_0, MM2S, 0)
// CHECK:           %mem_3_3 = aie.mem(%{{.*}}tile_3_3) {
// CHECK:             %3 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:  // 2 preds: ^bb0, ^bb2
// CHECK:             aie.use_lock(%[[VAL_2]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[VAL_0]] : memref<16xi32>, 0, 16)
// CHECK:             aie.use_lock(%[[VAL_3]], Release, 1)
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:  // pred: ^bb1
// CHECK:             aie.use_lock(%[[VAL_2]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[VAL_1]] : memref<16xi32>, 0, 16)
// CHECK:             aie.use_lock(%[[VAL_3]], Release, 1)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:  // pred: ^bb0
// CHECK:             aie.end
// CHECK:           }
// CHECK:         }

module @shimtileDMA_channels {
    aie.device(xcve2302) {
        %tile20 = aie.tile(2, 0)
        %tile33 = aie.tile(3, 3)

        %buff0 = aie.external_buffer : memref<16xi32>
        %lock0 = aie.lock(%tile20, 0)
        %buff1 = aie.external_buffer : memref<16xi32>
        %lock1 = aie.lock(%tile20, 1)
        %buff2 = aie.external_buffer : memref<16xi32>
        %lock2 = aie.lock(%tile20, 2)

        aie.objectfifo @objfifo (%tile20, {%tile33}, 2 : i32) : !aie.objectfifo<memref<16xi32>>

        %ext_buffer_in = aie.external_buffer {sym_name = "ext_buffer_in"}: memref<16xi32>
        aie.objectfifo.register_external_buffers @objfifo (%tile20, {%ext_buffer_in}) : (memref<16xi32>)

        %mem12 = aie.shim_dma(%tile20) {
            %dma1 = aie.dma_start(MM2S, 1, ^bb1, ^bb3)
        ^bb1:
            aie.use_lock(%lock0, Acquire, 1)
            aie.dma_bd(%buff0 : memref<16xi32>, 0, 16)
            aie.use_lock(%lock0, Release, 0)
            aie.next_bd ^bb2
        ^bb2:
            aie.use_lock(%lock1, Acquire, 1)
            aie.dma_bd(%buff1 : memref<16xi32>, 0, 16)
            aie.use_lock(%lock1, Release, 0)
            aie.next_bd ^bb1
        ^bb3:
            %dma2 = aie.dma_start(S2MM, 0, ^bb4, ^bb5)
        ^bb4:
            aie.use_lock(%lock2, Acquire, 0)
            aie.dma_bd(%buff2 : memref<16xi32>, 0, 16)
            aie.use_lock(%lock2, Release, 1)
            aie.next_bd ^bb4
        ^bb5:
            aie.end
        }
    }
}
