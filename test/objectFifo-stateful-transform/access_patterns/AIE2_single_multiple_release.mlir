//===- AIE2_single_multiple_release.mlir ------------------------*- MLIR -*-===//
//
// Copyright (C) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform --aie-objectFifo-unroll %s | FileCheck %s

// CHECK-LABEL:   aie.device(npu1_1col) {
// CHECK:           %[[VAL_0:.*]] = aie.tile(0, 0)
// CHECK:           %[[VAL_1:.*]] = aie.tile(0, 2)
// CHECK:           %[[VAL_2:.*]] = aie.buffer(%[[VAL_1]]) {sym_name = "of_cons_buff_0"} : memref<16xi32>
// CHECK:           %[[VAL_3:.*]] = aie.buffer(%[[VAL_1]]) {sym_name = "of_cons_buff_1"} : memref<16xi32>
// CHECK:           %[[VAL_4:.*]] = aie.buffer(%[[VAL_1]]) {sym_name = "of_cons_buff_2"} : memref<16xi32>
// CHECK:           %[[VAL_5:.*]] = aie.lock(%[[VAL_1]]) {init = 3 : i32, sym_name = "of_cons_prod_lock_0"}
// CHECK:           %[[VAL_6:.*]] = aie.lock(%[[VAL_1]]) {init = 0 : i32, sym_name = "of_cons_cons_lock_0"}
// CHECK:           %[[VAL_7:.*]] = aie.tile(0, 3)
// CHECK:           %[[VAL_8:.*]] = aie.buffer(%[[VAL_7]]) {sym_name = "of2_cons_buff_0"} : memref<16xi32>
// CHECK:           %[[VAL_9:.*]] = aie.buffer(%[[VAL_7]]) {sym_name = "of2_cons_buff_1"} : memref<16xi32>
// CHECK:           %[[VAL_10:.*]] = aie.buffer(%[[VAL_7]]) {sym_name = "of2_cons_buff_2"} : memref<16xi32>
// CHECK:           %[[VAL_11:.*]] = aie.lock(%[[VAL_7]]) {init = 3 : i32, sym_name = "of2_cons_prod_lock_0"}
// CHECK:           %[[VAL_12:.*]] = aie.lock(%[[VAL_7]]) {init = 0 : i32, sym_name = "of2_cons_cons_lock_0"}
// CHECK:           aie.flow(%[[VAL_0]], DMA : 0, %[[VAL_1]], DMA : 0)
// CHECK:           aie.flow(%[[VAL_0]], DMA : 1, %[[VAL_7]], DMA : 0)
// CHECK:           func.func @some_work(%[[VAL_13:.*]]: memref<16xi32>) {
// CHECK:             return
// CHECK:           }
// CHECK:           %[[VAL_14:.*]] = aie.core(%[[VAL_1]]) {
// CHECK:             %[[VAL_15:.*]] = arith.constant 2 : i32
// CHECK:             %[[VAL_16:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_6]], AcquireGreaterEqual, %[[VAL_15]])
// CHECK:             func.call @some_work(%[[VAL_2]]) : (memref<16xi32>) -> ()
// CHECK:             func.call @some_work(%[[VAL_3]]) : (memref<16xi32>) -> ()
// CHECK:             aie.use_lock(%[[VAL_5]], Release, %[[VAL_15]])
// CHECK:             aie.use_lock(%[[VAL_6]], AcquireGreaterEqual, %[[VAL_16]])
// CHECK:             func.call @some_work(%[[VAL_4]]) : (memref<16xi32>) -> ()
// CHECK:             aie.use_lock(%[[VAL_5]], Release, %[[VAL_16]])
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_17:.*]] = aie.core(%[[VAL_7]]) {
// CHECK:             %[[VAL_18:.*]] = arith.constant 2 : i32
// CHECK:             %[[VAL_19:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_12]], AcquireGreaterEqual, %[[VAL_18]])
// CHECK:             func.call @some_work(%[[VAL_8]]) : (memref<16xi32>) -> ()
// CHECK:             func.call @some_work(%[[VAL_9]]) : (memref<16xi32>) -> ()
// CHECK:             aie.use_lock(%[[VAL_11]], Release, %[[VAL_19]])
// CHECK:             aie.use_lock(%[[VAL_11]], Release, %[[VAL_19]])
// CHECK:             aie.use_lock(%[[VAL_12]], AcquireGreaterEqual, %[[VAL_19]])
// CHECK:             func.call @some_work(%[[VAL_10]]) : (memref<16xi32>) -> ()
// CHECK:             aie.use_lock(%[[VAL_11]], Release, %[[VAL_19]])
// CHECK:             aie.end
// CHECK:           }
// CHECK:           aie.shim_dma_allocation @of_shim_alloc(%[[VAL_0]], MM2S, 0)
// CHECK:           aie.shim_dma_allocation @of2_shim_alloc(%[[VAL_0]], MM2S, 1)
// CHECK:           %[[VAL_20:.*]] = aie.mem(%[[VAL_1]]) {
// CHECK:             %[[VAL_21:.*]] = arith.constant 1 : i32
// CHECK:             %[[VAL_22:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb4)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[VAL_5]], AcquireGreaterEqual, %[[VAL_21]])
// CHECK:             aie.dma_bd(%[[VAL_2]] : memref<16xi32> offset = 0 len = 16)
// CHECK:             aie.use_lock(%[[VAL_6]], Release, %[[VAL_21]])
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[VAL_5]], AcquireGreaterEqual, %[[VAL_21]])
// CHECK:             aie.dma_bd(%[[VAL_3]] : memref<16xi32> offset = 0 len = 16)
// CHECK:             aie.use_lock(%[[VAL_6]], Release, %[[VAL_21]])
// CHECK:             aie.next_bd ^bb3
// CHECK:           ^bb3:
// CHECK:             aie.use_lock(%[[VAL_5]], AcquireGreaterEqual, %[[VAL_21]])
// CHECK:             aie.dma_bd(%[[VAL_4]] : memref<16xi32> offset = 0 len = 16)
// CHECK:             aie.use_lock(%[[VAL_6]], Release, %[[VAL_21]])
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb4:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_23:.*]] = aie.mem(%[[VAL_7]]) {
// CHECK:             %[[VAL_24:.*]] = arith.constant 1 : i32
// CHECK:             %[[VAL_25:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb4)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[VAL_11]], AcquireGreaterEqual, %[[VAL_24]])
// CHECK:             aie.dma_bd(%[[VAL_8]] : memref<16xi32> offset = 0 len = 16)
// CHECK:             aie.use_lock(%[[VAL_12]], Release, %[[VAL_24]])
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[VAL_11]], AcquireGreaterEqual, %[[VAL_24]])
// CHECK:             aie.dma_bd(%[[VAL_9]] : memref<16xi32> offset = 0 len = 16)
// CHECK:             aie.use_lock(%[[VAL_12]], Release, %[[VAL_24]])
// CHECK:             aie.next_bd ^bb3
// CHECK:           ^bb3:
// CHECK:             aie.use_lock(%[[VAL_11]], AcquireGreaterEqual, %[[VAL_24]])
// CHECK:             aie.dma_bd(%[[VAL_10]] : memref<16xi32> offset = 0 len = 16)
// CHECK:             aie.use_lock(%[[VAL_12]], Release, %[[VAL_24]])
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb4:
// CHECK:             aie.end
// CHECK:           }
// CHECK:         }

module @single_multiple_release {
    aie.device(npu1_1col) {
        %tile00 = aie.tile(0, 0)
        %tile02 = aie.tile(0, 2)
        %tile03 = aie.tile(0, 3)

        aie.objectfifo @of (%tile00, {%tile02}, 3 : i32) : !aie.objectfifo<memref<16xi32>>
        aie.objectfifo @of2 (%tile00, {%tile03}, 3 : i32) : !aie.objectfifo<memref<16xi32>>

        func.func @some_work(%line_in:memref<16xi32>) -> () {
            return
        }

        %core12 = aie.core(%tile02) {
            %1, %2 = aie.objectfifo.acquire @of(Consume) : memref<16xi32>, memref<16xi32>
            func.call @some_work(%1) : (memref<16xi32>) -> ()
            func.call @some_work(%2) : (memref<16xi32>) -> ()
            aie.objectfifo.release @of(Consume) [2]
            %4 = aie.objectfifo.acquire @of(Consume) : memref<16xi32>
            func.call @some_work(%4) : (memref<16xi32>) -> ()
            aie.objectfifo.release @of(Consume) [1]
            aie.end
        }

        %core13 = aie.core(%tile03) {
            %1, %2 = aie.objectfifo.acquire @of2(Consume) : memref<16xi32>, memref<16xi32>
            func.call @some_work(%1) : (memref<16xi32>) -> ()
            func.call @some_work(%2) : (memref<16xi32>) -> ()
            aie.objectfifo.release @of2(Consume) [1]
            aie.objectfifo.release @of2(Consume) [1]
            %4 = aie.objectfifo.acquire @of2(Consume) : memref<16xi32>
            func.call @some_work(%4) : (memref<16xi32>) -> ()
            aie.objectfifo.release @of2(Consume) [1]
            aie.end
        }
    }
}
