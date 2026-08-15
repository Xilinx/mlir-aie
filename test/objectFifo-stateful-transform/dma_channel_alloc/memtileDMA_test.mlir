//===- memtileDMA_test.mlir ------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform --aie-objectFifo-unroll %s | FileCheck %s

// CHECK-LABEL:   aie.device(xcve2302) {
// CHECK:           %[[VAL_0:.*]] = aie.tile(1, 1)
// CHECK:           %[[VAL_1:.*]] = aie.buffer(%[[VAL_0]]) {sym_name = "objfifo_buff_0"} : memref<16xi32>
// CHECK:           %[[VAL_2:.*]] = aie.buffer(%[[VAL_0]]) {sym_name = "objfifo_buff_1"} : memref<16xi32>
// CHECK:           %[[VAL_3:.*]] = aie.lock(%[[VAL_0]]) {init = 2 : i32, sym_name = "objfifo_prod_lock_0"}
// CHECK:           %[[VAL_4:.*]] = aie.lock(%[[VAL_0]]) {init = 0 : i32, sym_name = "objfifo_cons_lock_0"}
// CHECK:           %[[VAL_5:.*]] = aie.tile(3, 3)
// CHECK:           %[[VAL_6:.*]] = aie.buffer(%[[VAL_5]]) {sym_name = "objfifo_cons_buff_0"} : memref<16xi32>
// CHECK:           %[[VAL_7:.*]] = aie.buffer(%[[VAL_5]]) {sym_name = "objfifo_cons_buff_1"} : memref<16xi32>
// CHECK:           %[[VAL_8:.*]] = aie.lock(%[[VAL_5]]) {init = 2 : i32, sym_name = "objfifo_cons_prod_lock_0"}
// CHECK:           %[[VAL_9:.*]] = aie.lock(%[[VAL_5]]) {init = 0 : i32, sym_name = "objfifo_cons_cons_lock_0"}
// CHECK:           %[[VAL_10:.*]] = aie.buffer(%[[VAL_0]]) : memref<16xi32>
// CHECK:           %[[VAL_11:.*]] = aie.lock(%[[VAL_0]], 0)
// CHECK:           %[[VAL_12:.*]] = aie.buffer(%[[VAL_0]]) : memref<16xi32>
// CHECK:           %[[VAL_13:.*]] = aie.lock(%[[VAL_0]], 1)
// CHECK:           %[[VAL_14:.*]] = aie.buffer(%[[VAL_0]]) : memref<16xi32>
// CHECK:           %[[VAL_15:.*]] = aie.lock(%[[VAL_0]], 2)
// CHECK:           aie.flow(%[[VAL_0]], DMA : 0, %[[VAL_5]], DMA : 0)
// CHECK:           %[[VAL_16:.*]] = aie.memtile_dma(%[[VAL_0]]) {
// CHECK:             %[[VAL_17:.*]] = arith.constant 1 : i32
// CHECK:             %[[VAL_18:.*]] = arith.constant 0 : i32
// CHECK:             %[[VAL_19:.*]] = aie.dma_start(MM2S, 1, ^bb1, ^bb3)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[VAL_11]], Acquire, %[[VAL_17]])
// CHECK:             aie.dma_bd(%[[VAL_10]] : memref<16xi32> offset = 0 len = 16)
// CHECK:             aie.use_lock(%[[VAL_11]], Release, %[[VAL_18]])
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[VAL_13]], Acquire, %[[VAL_17]])
// CHECK:             aie.dma_bd(%[[VAL_12]] : memref<16xi32> offset = 0 len = 16)
// CHECK:             aie.use_lock(%[[VAL_13]], Release, %[[VAL_18]])
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:
// CHECK:             %[[VAL_20:.*]] = aie.dma_start(S2MM, 0, ^bb4, ^bb5)
// CHECK:           ^bb4:
// CHECK:             aie.use_lock(%[[VAL_15]], Acquire, %[[VAL_18]])
// CHECK:             aie.dma_bd(%[[VAL_14]] : memref<16xi32> offset = 0 len = 16)
// CHECK:             aie.use_lock(%[[VAL_15]], Release, %[[VAL_17]])
// CHECK:             aie.next_bd ^bb4
// CHECK:           ^bb5:
// CHECK:             %[[VAL_21:.*]] = aie.dma_start(MM2S, 0, ^bb6, ^bb8)
// CHECK:           ^bb6:
// CHECK:             aie.use_lock(%[[VAL_4]], AcquireGreaterEqual, %[[VAL_17]])
// CHECK:             aie.dma_bd(%[[VAL_1]] : memref<16xi32> offset = 0 len = 16)
// CHECK:             aie.use_lock(%[[VAL_3]], Release, %[[VAL_17]])
// CHECK:             aie.next_bd ^bb7
// CHECK:           ^bb7:
// CHECK:             aie.use_lock(%[[VAL_4]], AcquireGreaterEqual, %[[VAL_17]])
// CHECK:             aie.dma_bd(%[[VAL_2]] : memref<16xi32> offset = 0 len = 16)
// CHECK:             aie.use_lock(%[[VAL_3]], Release, %[[VAL_17]])
// CHECK:             aie.next_bd ^bb6
// CHECK:           ^bb8:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_22:.*]] = aie.mem(%[[VAL_5]]) {
// CHECK:             %[[VAL_23:.*]] = arith.constant 1 : i32
// CHECK:             %[[VAL_24:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[VAL_8]], AcquireGreaterEqual, %[[VAL_23]])
// CHECK:             aie.dma_bd(%[[VAL_6]] : memref<16xi32> offset = 0 len = 16)
// CHECK:             aie.use_lock(%[[VAL_9]], Release, %[[VAL_23]])
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[VAL_8]], AcquireGreaterEqual, %[[VAL_23]])
// CHECK:             aie.dma_bd(%[[VAL_7]] : memref<16xi32> offset = 0 len = 16)
// CHECK:             aie.use_lock(%[[VAL_9]], Release, %[[VAL_23]])
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:
// CHECK:             aie.end
// CHECK:           }
// CHECK:         }

module @memtileDMA_channels {
    aie.device(xcve2302) {
        %tile11 = aie.tile(1, 1)
        %tile33 = aie.tile(3, 3)

        %buff0 = aie.buffer(%tile11) : memref<16xi32>
        %lock0 = aie.lock(%tile11, 0)
        %buff1 = aie.buffer(%tile11) : memref<16xi32>
        %lock1 = aie.lock(%tile11, 1)
        %buff2 = aie.buffer(%tile11) : memref<16xi32>
        %lock2 = aie.lock(%tile11, 2)

        aie.objectfifo @objfifo (%tile11, {%tile33}, 2 : i32) : !aie.objectfifo<memref<16xi32>>

        %mem12 = aie.memtile_dma(%tile11) {
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
