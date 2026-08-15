//===- matmul_test.mlir -----------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2023 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Date: September 5th 2023
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform --aie-objectFifo-unroll %s | FileCheck %s

// CHECK-LABEL:   aie.device(xcve2302) {
// CHECK:           %[[VAL_0:.*]] = aie.tile(0, 0)
// CHECK:           %[[VAL_1:.*]] = aie.tile(0, 2)
// CHECK:           %[[VAL_2:.*]] = aie.buffer(%[[VAL_1]]) {sym_name = "inA_cons_buff_0"} : memref<16x8xi16>
// CHECK:           %[[VAL_3:.*]] = aie.buffer(%[[VAL_1]]) {sym_name = "inA_cons_buff_1"} : memref<16x8xi16>
// CHECK:           %[[VAL_4:.*]] = aie.lock(%[[VAL_1]]) {init = 2 : i32, sym_name = "inA_cons_prod_lock_0"}
// CHECK:           %[[VAL_5:.*]] = aie.lock(%[[VAL_1]]) {init = 0 : i32, sym_name = "inA_cons_cons_lock_0"}
// CHECK:           %[[VAL_6:.*]] = aie.buffer(%[[VAL_1]]) {sym_name = "inB_cons_buff_0"} : memref<8x16xi16>
// CHECK:           %[[VAL_7:.*]] = aie.buffer(%[[VAL_1]]) {sym_name = "inB_cons_buff_1"} : memref<8x16xi16>
// CHECK:           %[[VAL_8:.*]] = aie.lock(%[[VAL_1]]) {init = 2 : i32, sym_name = "inB_cons_prod_lock_0"}
// CHECK:           %[[VAL_9:.*]] = aie.lock(%[[VAL_1]]) {init = 0 : i32, sym_name = "inB_cons_cons_lock_0"}
// CHECK:           %[[VAL_10:.*]] = aie.buffer(%[[VAL_1]]) {sym_name = "outC_buff_0"} : memref<16x16xi16>
// CHECK:           %[[VAL_11:.*]] = aie.buffer(%[[VAL_1]]) {sym_name = "outC_buff_1"} : memref<16x16xi16>
// CHECK:           %[[VAL_12:.*]] = aie.lock(%[[VAL_1]]) {init = 2 : i32, sym_name = "outC_prod_lock_0"}
// CHECK:           %[[VAL_13:.*]] = aie.lock(%[[VAL_1]]) {init = 0 : i32, sym_name = "outC_cons_lock_0"}
// CHECK:           aie.flow(%[[VAL_0]], DMA : 0, %[[VAL_1]], DMA : 0)
// CHECK:           aie.flow(%[[VAL_0]], DMA : 1, %[[VAL_1]], DMA : 1)
// CHECK:           aie.flow(%[[VAL_1]], DMA : 0, %[[VAL_0]], DMA : 0)
// CHECK:           func.func @zero_scalar_i16(%[[VAL_14:.*]]: memref<16x16xi16>) {
// CHECK:             return
// CHECK:           }
// CHECK:           func.func @matmul_scalar_i16_i16(%[[VAL_15:.*]]: memref<16x8xi16>, %[[VAL_16:.*]]: memref<8x16xi16>, %[[VAL_17:.*]]: memref<16x16xi16>) {
// CHECK:             return
// CHECK:           }
// CHECK:           %[[VAL_18:.*]] = aie.core(%[[VAL_1]]) {
// CHECK:             %[[VAL_19:.*]] = arith.constant 4294967295 : index
// CHECK:             %[[VAL_20:.*]] = arith.constant 4 : index
// CHECK:             %[[VAL_21:.*]] = arith.constant 1 : index
// CHECK:             %[[VAL_22:.*]] = arith.constant 0 : index
// CHECK:             %[[VAL_23:.*]] = arith.constant 2 : index
// CHECK:             %[[VAL_24:.*]] = arith.constant 1 : i32
// CHECK:             scf.for %[[VAL_25:.*]] = %[[VAL_22]] to %[[VAL_19]] step %[[VAL_21]] {
// CHECK:               scf.for %[[VAL_26:.*]] = %[[VAL_22]] to %[[VAL_20]] step %[[VAL_23]] {
// CHECK:                 aie.use_lock(%[[VAL_12]], AcquireGreaterEqual, %[[VAL_24]])
// CHECK:                 func.call @zero_scalar_i16(%[[VAL_10]]) : (memref<16x16xi16>) -> ()
// CHECK:                 scf.for %[[VAL_27:.*]] = %[[VAL_22]] to %[[VAL_20]] step %[[VAL_23]] {
// CHECK:                   aie.use_lock(%[[VAL_5]], AcquireGreaterEqual, %[[VAL_24]])
// CHECK:                   aie.use_lock(%[[VAL_9]], AcquireGreaterEqual, %[[VAL_24]])
// CHECK:                   func.call @matmul_scalar_i16_i16(%[[VAL_2]], %[[VAL_6]], %[[VAL_10]]) : (memref<16x8xi16>, memref<8x16xi16>, memref<16x16xi16>) -> ()
// CHECK:                   aie.use_lock(%[[VAL_4]], Release, %[[VAL_24]])
// CHECK:                   aie.use_lock(%[[VAL_8]], Release, %[[VAL_24]])
// CHECK:                   aie.use_lock(%[[VAL_5]], AcquireGreaterEqual, %[[VAL_24]])
// CHECK:                   aie.use_lock(%[[VAL_9]], AcquireGreaterEqual, %[[VAL_24]])
// CHECK:                   func.call @matmul_scalar_i16_i16(%[[VAL_3]], %[[VAL_7]], %[[VAL_10]]) : (memref<16x8xi16>, memref<8x16xi16>, memref<16x16xi16>) -> ()
// CHECK:                   aie.use_lock(%[[VAL_4]], Release, %[[VAL_24]])
// CHECK:                   aie.use_lock(%[[VAL_8]], Release, %[[VAL_24]])
// CHECK:                 }
// CHECK:                 aie.use_lock(%[[VAL_13]], Release, %[[VAL_24]])
// CHECK:                 aie.use_lock(%[[VAL_12]], AcquireGreaterEqual, %[[VAL_24]])
// CHECK:                 func.call @zero_scalar_i16(%[[VAL_11]]) : (memref<16x16xi16>) -> ()
// CHECK:                 scf.for %[[VAL_28:.*]] = %[[VAL_22]] to %[[VAL_20]] step %[[VAL_23]] {
// CHECK:                   aie.use_lock(%[[VAL_5]], AcquireGreaterEqual, %[[VAL_24]])
// CHECK:                   aie.use_lock(%[[VAL_9]], AcquireGreaterEqual, %[[VAL_24]])
// CHECK:                   func.call @matmul_scalar_i16_i16(%[[VAL_2]], %[[VAL_6]], %[[VAL_11]]) : (memref<16x8xi16>, memref<8x16xi16>, memref<16x16xi16>) -> ()
// CHECK:                   aie.use_lock(%[[VAL_4]], Release, %[[VAL_24]])
// CHECK:                   aie.use_lock(%[[VAL_8]], Release, %[[VAL_24]])
// CHECK:                   aie.use_lock(%[[VAL_5]], AcquireGreaterEqual, %[[VAL_24]])
// CHECK:                   aie.use_lock(%[[VAL_9]], AcquireGreaterEqual, %[[VAL_24]])
// CHECK:                   func.call @matmul_scalar_i16_i16(%[[VAL_3]], %[[VAL_7]], %[[VAL_11]]) : (memref<16x8xi16>, memref<8x16xi16>, memref<16x16xi16>) -> ()
// CHECK:                   aie.use_lock(%[[VAL_4]], Release, %[[VAL_24]])
// CHECK:                   aie.use_lock(%[[VAL_8]], Release, %[[VAL_24]])
// CHECK:                 }
// CHECK:                 aie.use_lock(%[[VAL_13]], Release, %[[VAL_24]])
// CHECK:               }
// CHECK:             }
// CHECK:             aie.end
// CHECK:           }
// CHECK:           aie.shim_dma_allocation @inA_shim_alloc(%[[VAL_0]], MM2S, 0)
// CHECK:           aie.shim_dma_allocation @inB_shim_alloc(%[[VAL_0]], MM2S, 1)
// CHECK:           aie.shim_dma_allocation @outC_shim_alloc(%[[VAL_0]], S2MM, 0)
// CHECK:           %[[VAL_29:.*]] = aie.mem(%[[VAL_1]]) {
// CHECK:             %[[VAL_30:.*]] = arith.constant 1 : i32
// CHECK:             %[[VAL_31:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[VAL_4]], AcquireGreaterEqual, %[[VAL_30]])
// CHECK:             aie.dma_bd(%[[VAL_2]] : memref<16x8xi16> offset = 0 len = 128)
// CHECK:             aie.use_lock(%[[VAL_5]], Release, %[[VAL_30]])
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[VAL_4]], AcquireGreaterEqual, %[[VAL_30]])
// CHECK:             aie.dma_bd(%[[VAL_3]] : memref<16x8xi16> offset = 0 len = 128)
// CHECK:             aie.use_lock(%[[VAL_5]], Release, %[[VAL_30]])
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:
// CHECK:             %[[VAL_32:.*]] = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
// CHECK:           ^bb4:
// CHECK:             aie.use_lock(%[[VAL_8]], AcquireGreaterEqual, %[[VAL_30]])
// CHECK:             aie.dma_bd(%[[VAL_6]] : memref<8x16xi16> offset = 0 len = 128)
// CHECK:             aie.use_lock(%[[VAL_9]], Release, %[[VAL_30]])
// CHECK:             aie.next_bd ^bb5
// CHECK:           ^bb5:
// CHECK:             aie.use_lock(%[[VAL_8]], AcquireGreaterEqual, %[[VAL_30]])
// CHECK:             aie.dma_bd(%[[VAL_7]] : memref<8x16xi16> offset = 0 len = 128)
// CHECK:             aie.use_lock(%[[VAL_9]], Release, %[[VAL_30]])
// CHECK:             aie.next_bd ^bb4
// CHECK:           ^bb6:
// CHECK:             %[[VAL_33:.*]] = aie.dma_start(MM2S, 0, ^bb7, ^bb9)
// CHECK:           ^bb7:
// CHECK:             aie.use_lock(%[[VAL_13]], AcquireGreaterEqual, %[[VAL_30]])
// CHECK:             aie.dma_bd(%[[VAL_10]] : memref<16x16xi16> offset = 0 len = 256)
// CHECK:             aie.use_lock(%[[VAL_12]], Release, %[[VAL_30]])
// CHECK:             aie.next_bd ^bb8
// CHECK:           ^bb8:
// CHECK:             aie.use_lock(%[[VAL_13]], AcquireGreaterEqual, %[[VAL_30]])
// CHECK:             aie.dma_bd(%[[VAL_11]] : memref<16x16xi16> offset = 0 len = 256)
// CHECK:             aie.use_lock(%[[VAL_12]], Release, %[[VAL_30]])
// CHECK:             aie.next_bd ^bb7
// CHECK:           ^bb9:
// CHECK:             aie.end
// CHECK:           }
// CHECK:         }

module @matmul {
  aie.device(xcve2302) {

    %t00 = aie.tile(0, 0)
    %t02 = aie.tile(0, 2)

    aie.objectfifo @inA  (%t00, { %t02 }, 2 : i32) : !aie.objectfifo<memref<16x8xi16>>
    aie.objectfifo @inB  (%t00, { %t02 }, 2 : i32) : !aie.objectfifo<memref<8x16xi16>>
    aie.objectfifo @outC (%t02, { %t00 }, 2 : i32) : !aie.objectfifo<memref<16x16xi16>>

    func.func @zero_scalar_i16(%elem0 : memref<16x16xi16>) -> () { return }
    func.func @matmul_scalar_i16_i16(%elem0 : memref<16x8xi16>, %elem1 : memref<8x16xi16>, %elem2 : memref<16x16xi16>) -> () { return }

    aie.core(%t02) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      %intmax = arith.constant 0xFFFFFFFF : index

      scf.for %reps = %c0 to %intmax step %c1 {

        scf.for %arg2 = %c0 to %c4 step %c1 {
          %elem2 = aie.objectfifo.acquire @outC(Produce) : memref<16x16xi16>
          func.call @zero_scalar_i16(%elem2) : (memref<16x16xi16>) -> ()

          scf.for %arg3 = %c0 to %c4 step %c1 {
            %elem0 = aie.objectfifo.acquire @inA(Consume) : memref<16x8xi16>
            %elem1 = aie.objectfifo.acquire @inB(Consume) : memref<8x16xi16>

            func.call @matmul_scalar_i16_i16(%elem0, %elem1, %elem2) : (memref<16x8xi16>, memref<8x16xi16>, memref<16x16xi16>) -> ()

            aie.objectfifo.release @inA(Consume) [1]
            aie.objectfifo.release @inB(Consume) [1]
          }
          aie.objectfifo.release @outC(Produce) [1]
        }
      }
      aie.end
    }
  }
}
