//===- shimtileDMA_test.mlir -----------------------------------*- MLIR -*-===//
//
// Copyright (C) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform --aie-objectFifo-unroll %s | FileCheck %s

// CHECK-LABEL:   aie.device(xcve2302) {
// CHECK:           %[[VAL_0:.*]] = aie.tile(2, 0)
// CHECK:           %[[VAL_1:.*]] = aie.lock(%[[VAL_0]]) {init = 1 : i32, sym_name = "objfifo_prod_lock_0"}
// CHECK:           %[[VAL_2:.*]] = aie.lock(%[[VAL_0]]) {init = 0 : i32, sym_name = "objfifo_cons_lock_0"}
// CHECK:           %[[VAL_3:.*]] = aie.tile(3, 3)
// CHECK:           %[[VAL_4:.*]] = aie.buffer(%[[VAL_3]]) {sym_name = "objfifo_cons_buff_0"} : memref<16xi32>
// CHECK:           %[[VAL_5:.*]] = aie.buffer(%[[VAL_3]]) {sym_name = "objfifo_cons_buff_1"} : memref<16xi32>
// CHECK:           %[[VAL_6:.*]] = aie.lock(%[[VAL_3]]) {init = 2 : i32, sym_name = "objfifo_cons_prod_lock_0"}
// CHECK:           %[[VAL_7:.*]] = aie.lock(%[[VAL_3]]) {init = 0 : i32, sym_name = "objfifo_cons_cons_lock_0"}
// CHECK:           %[[VAL_8:.*]] = aie.external_buffer : memref<16xi32>
// CHECK:           %[[VAL_9:.*]] = aie.lock(%[[VAL_0]], 0)
// CHECK:           %[[VAL_10:.*]] = aie.external_buffer : memref<16xi32>
// CHECK:           %[[VAL_11:.*]] = aie.lock(%[[VAL_0]], 1)
// CHECK:           %[[VAL_12:.*]] = aie.external_buffer : memref<16xi32>
// CHECK:           %[[VAL_13:.*]] = aie.lock(%[[VAL_0]], 2)
// CHECK:           aie.flow(%[[VAL_0]], DMA : 0, %[[VAL_3]], DMA : 0)
// CHECK:           %[[VAL_14:.*]] = aie.external_buffer {sym_name = "ext_buffer_in"} : memref<16xi32>
// CHECK:           %[[VAL_15:.*]] = aie.shim_dma(%[[VAL_0]]) {
// CHECK:             %[[VAL_16:.*]] = arith.constant 1 : i32
// CHECK:             %[[VAL_17:.*]] = arith.constant 0 : i32
// CHECK:             %[[VAL_18:.*]] = aie.dma_start(MM2S, 1, ^bb1, ^bb3)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[VAL_9]], Acquire, %[[VAL_16]])
// CHECK:             aie.dma_bd(%[[VAL_8]] : memref<16xi32> offset = 0 len = 16)
// CHECK:             aie.use_lock(%[[VAL_9]], Release, %[[VAL_17]])
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[VAL_11]], Acquire, %[[VAL_16]])
// CHECK:             aie.dma_bd(%[[VAL_10]] : memref<16xi32> offset = 0 len = 16)
// CHECK:             aie.use_lock(%[[VAL_11]], Release, %[[VAL_17]])
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:
// CHECK:             %[[VAL_19:.*]] = aie.dma_start(S2MM, 0, ^bb4, ^bb5)
// CHECK:           ^bb4:
// CHECK:             aie.use_lock(%[[VAL_13]], Acquire, %[[VAL_17]])
// CHECK:             aie.dma_bd(%[[VAL_12]] : memref<16xi32> offset = 0 len = 16)
// CHECK:             aie.use_lock(%[[VAL_13]], Release, %[[VAL_16]])
// CHECK:             aie.next_bd ^bb4
// CHECK:           ^bb5:
// CHECK:             %[[VAL_20:.*]] = aie.dma_start(MM2S, 0, ^bb6, ^bb7)
// CHECK:           ^bb6:
// CHECK:             aie.use_lock(%[[VAL_2]], AcquireGreaterEqual, %[[VAL_16]])
// CHECK:             aie.dma_bd(%[[VAL_14]] : memref<16xi32> offset = 0 len = 16)
// CHECK:             aie.use_lock(%[[VAL_1]], Release, %[[VAL_16]])
// CHECK:             aie.next_bd ^bb6
// CHECK:           ^bb7:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           aie.shim_dma_allocation @objfifo_shim_alloc(%[[VAL_0]], MM2S, 0)
// CHECK:           %[[VAL_21:.*]] = aie.mem(%[[VAL_3]]) {
// CHECK:             %[[VAL_22:.*]] = arith.constant 1 : i32
// CHECK:             %[[VAL_23:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[VAL_6]], AcquireGreaterEqual, %[[VAL_22]])
// CHECK:             aie.dma_bd(%[[VAL_4]] : memref<16xi32> offset = 0 len = 16)
// CHECK:             aie.use_lock(%[[VAL_7]], Release, %[[VAL_22]])
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[VAL_6]], AcquireGreaterEqual, %[[VAL_22]])
// CHECK:             aie.dma_bd(%[[VAL_5]] : memref<16xi32> offset = 0 len = 16)
// CHECK:             aie.use_lock(%[[VAL_7]], Release, %[[VAL_22]])
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:
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
            %c1_ul1 = arith.constant 1 : i32
            aie.use_lock(%lock0, Acquire, %c1_ul1)
            aie.dma_bd(%buff0 : memref<16xi32> offset = 0 len = 16)
            %c0_ul2 = arith.constant 0 : i32
            aie.use_lock(%lock0, Release, %c0_ul2)
            aie.next_bd ^bb2
        ^bb2:
            %c1_ul3 = arith.constant 1 : i32
            aie.use_lock(%lock1, Acquire, %c1_ul3)
            aie.dma_bd(%buff1 : memref<16xi32> offset = 0 len = 16)
            %c0_ul4 = arith.constant 0 : i32
            aie.use_lock(%lock1, Release, %c0_ul4)
            aie.next_bd ^bb1
        ^bb3:
            %dma2 = aie.dma_start(S2MM, 0, ^bb4, ^bb5)
        ^bb4:
            %c0_ul5 = arith.constant 0 : i32
            aie.use_lock(%lock2, Acquire, %c0_ul5)
            aie.dma_bd(%buff2 : memref<16xi32> offset = 0 len = 16)
            %c1_ul6 = arith.constant 1 : i32
            aie.use_lock(%lock2, Release, %c1_ul6)
            aie.next_bd ^bb4
        ^bb5:
            aie.end
        }
    }
}
