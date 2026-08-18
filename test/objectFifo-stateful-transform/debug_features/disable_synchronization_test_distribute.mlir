//===- disable_synchronization_test_distribute.mlir -------------*- MLIR -*-===//
//
// Copyright (C) 2024 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform="skip-verify=true" --aie-objectFifo-unroll %s | FileCheck %s

// CHECK-LABEL:   aie.device(xcve2302) {
// CHECK:           %[[VAL_0:.*]] = aie.tile(2, 0)
// CHECK:           %[[VAL_1:.*]] = aie.tile(2, 1)
// CHECK:           %[[VAL_2:.*]] = aie.buffer(%[[VAL_1]]) {sym_name = "link3_buff_0"} : memref<36xi32>
// CHECK:           %[[VAL_3:.*]] = aie.tile(2, 2)
// CHECK:           %[[VAL_4:.*]] = aie.buffer(%[[VAL_3]]) {sym_name = "link1_buff_0"} : memref<4x4xi32>
// CHECK:           %[[VAL_5:.*]] = aie.lock(%[[VAL_3]]) {init = 1 : i32, sym_name = "link1_prod_lock_0"}
// CHECK:           %[[VAL_6:.*]] = aie.lock(%[[VAL_3]]) {init = 0 : i32, sym_name = "link1_cons_lock_0"}
// CHECK:           %[[VAL_7:.*]] = aie.tile(2, 3)
// CHECK:           %[[VAL_8:.*]] = aie.buffer(%[[VAL_7]]) {sym_name = "link2_buff_0"} : memref<20xi32>
// CHECK:           %[[VAL_9:.*]] = aie.lock(%[[VAL_7]]) {init = 1 : i32, sym_name = "link2_prod_lock_0"}
// CHECK:           %[[VAL_10:.*]] = aie.lock(%[[VAL_7]]) {init = 0 : i32, sym_name = "link2_cons_lock_0"}
// CHECK:           aie.flow(%[[VAL_3]], DMA : 0, %[[VAL_1]], DMA : 0)
// CHECK:           aie.flow(%[[VAL_7]], DMA : 0, %[[VAL_1]], DMA : 1)
// CHECK:           aie.flow(%[[VAL_1]], DMA : 0, %[[VAL_0]], DMA : 0)
// CHECK:           aie.shim_dma_allocation @link3_shim_alloc(%[[VAL_0]], S2MM, 0)
// CHECK:           %[[VAL_11:.*]] = aie.mem(%[[VAL_3]]) {
// CHECK:             %[[VAL_12:.*]] = arith.constant 1 : i32
// CHECK:             %[[VAL_13:.*]] = aie.dma_start(MM2S, 0, ^bb1, ^bb2)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[VAL_6]], AcquireGreaterEqual, %[[VAL_12]])
// CHECK:             aie.dma_bd(%[[VAL_4]] : memref<4x4xi32> offset = 0 len = 16)
// CHECK:             aie.use_lock(%[[VAL_5]], Release, %[[VAL_12]])
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb2:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_14:.*]] = aie.memtile_dma(%[[VAL_1]]) {
// CHECK:             %[[VAL_15:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
// CHECK:           ^bb1:
// CHECK:             aie.dma_bd(%[[VAL_2]] : memref<36xi32> offset = 0 len = 16)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb2:
// CHECK:             %[[VAL_16:.*]] = aie.dma_start(S2MM, 1, ^bb3, ^bb4)
// CHECK:           ^bb3:
// CHECK:             aie.dma_bd(%[[VAL_2]] : memref<36xi32> offset = 16 len = 20)
// CHECK:             aie.next_bd ^bb3
// CHECK:           ^bb4:
// CHECK:             %[[VAL_17:.*]] = aie.dma_start(MM2S, 0, ^bb5, ^bb7)
// CHECK:           ^bb5:
// CHECK:             aie.dma_bd(%[[VAL_2]] : memref<36xi32> offset = 0 len = 16)
// CHECK:             aie.next_bd ^bb6
// CHECK:           ^bb6:
// CHECK:             aie.dma_bd(%[[VAL_2]] : memref<36xi32> offset = 16 len = 20)
// CHECK:             aie.next_bd ^bb5
// CHECK:           ^bb7:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_18:.*]] = aie.mem(%[[VAL_7]]) {
// CHECK:             %[[VAL_19:.*]] = arith.constant 1 : i32
// CHECK:             %[[VAL_20:.*]] = aie.dma_start(MM2S, 0, ^bb1, ^bb2)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[VAL_10]], AcquireGreaterEqual, %[[VAL_19]])
// CHECK:             aie.dma_bd(%[[VAL_8]] : memref<20xi32> offset = 0 len = 20)
// CHECK:             aie.use_lock(%[[VAL_9]], Release, %[[VAL_19]])
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb2:
// CHECK:             aie.end
// CHECK:           }
// CHECK:         }

module @disable_sync {
 aie.device(xcve2302) {
    %tile20 = aie.tile(2, 0)
    %tile21 = aie.tile(2, 1)
    %tile22 = aie.tile(2, 2)
    %tile23 = aie.tile(2, 3)

    aie.objectfifo @link1 (%tile22, {%tile21}, 1 : i32) : !aie.objectfifo<memref<4x4xi32>>
    aie.objectfifo @link2 (%tile23, {%tile21}, 1 : i32) : !aie.objectfifo<memref<20xi32>>
    aie.objectfifo @link3 (%tile21, {%tile20}, 1 : i32) { disable_synchronization = true } : !aie.objectfifo<memref<36xi32>>

    aie.objectfifo.link [@link1, @link2] -> [@link3] ([0, 16][])
 }
}
