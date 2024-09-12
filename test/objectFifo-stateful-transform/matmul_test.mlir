//===- matmul_test.mlir -----------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2023, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Date: September 5th 2023
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform %s | FileCheck %s

// CHECK:           %[[VAL_0:.*]] = aie.tile(0, 0)
// CHECK:           %[[VAL_1:.*]] = aie.tile(0, 2)
// CHECK:           %[[VAL_4:.*]] = aie.buffer(%[[VAL_1]]) {sym_name = "outC_buff_0"} : memref<16x16xi16>
// CHECK:           %[[VAL_5:.*]] = aie.buffer(%[[VAL_1]]) {sym_name = "outC_buff_1"} : memref<16x16xi16>
// CHECK:           %[[VAL_6:.*]] = aie.lock(%[[VAL_1]], 4) {init = 2 : i32, sym_name = "outC_prod_lock"}
// CHECK:           %[[VAL_7:.*]] = aie.lock(%[[VAL_1]], 5) {init = 0 : i32, sym_name = "outC_cons_lock"}
// CHECK:           %[[VAL_8:.*]] = aie.buffer(%[[VAL_1]]) {sym_name = "inB_cons_buff_0"} : memref<8x16xi16>
// CHECK:           %[[VAL_9:.*]] = aie.buffer(%[[VAL_1]]) {sym_name = "inB_cons_buff_1"} : memref<8x16xi16>
// CHECK:           %[[VAL_10:.*]] = aie.lock(%[[VAL_1]], 2) {init = 2 : i32, sym_name = "inB_cons_prod_lock"}
// CHECK:           %[[VAL_11:.*]] = aie.lock(%[[VAL_1]], 3) {init = 0 : i32, sym_name = "inB_cons_cons_lock"}
// CHECK:           %[[VAL_14:.*]] = aie.buffer(%[[VAL_1]]) {sym_name = "inA_cons_buff_0"} : memref<16x8xi16>
// CHECK:           %[[VAL_15:.*]] = aie.buffer(%[[VAL_1]]) {sym_name = "inA_cons_buff_1"} : memref<16x8xi16>
// CHECK:           %[[VAL_16:.*]] = aie.lock(%[[VAL_1]], 0) {init = 2 : i32, sym_name = "inA_cons_prod_lock"}
// CHECK:           %[[VAL_17:.*]] = aie.lock(%[[VAL_1]], 1) {init = 0 : i32, sym_name = "inA_cons_cons_lock"}
// CHECK:           aie.flow(%[[VAL_0]], DMA : 0, %[[VAL_1]], DMA : 0)
// CHECK:           aie.flow(%[[VAL_0]], DMA : 1, %[[VAL_1]], DMA : 1)
// CHECK:           aie.flow(%[[VAL_1]], DMA : 0, %[[VAL_0]], DMA : 0)
// CHECK:           func.func @zero_scalar_i16(%[[VAL_20:.*]]: memref<16x16xi16>) {
// CHECK:             return
// CHECK:           }
// CHECK:           func.func @matmul_scalar_i16_i16(%[[VAL_21:.*]]: memref<16x8xi16>, %[[VAL_22:.*]]: memref<8x16xi16>, %[[VAL_23:.*]]: memref<16x16xi16>) {
// CHECK:             return
// CHECK:           }
// CHECK:           aie.shim_dma_allocation @inA(MM2S, 0, 0)
// CHECK:           %[[VAL_24:.*]] = aie.core(%[[VAL_1]]) {
// CHECK:             %[[VAL_25:.*]] = arith.constant 0 : index
// CHECK:             %[[VAL_26:.*]] = arith.constant 1 : index
// CHECK:             %[[VAL_27:.*]] = arith.constant 4 : index
// CHECK:             %[[VAL_28:.*]] = arith.constant 4294967295 : index
// CHECK:             scf.for %[[VAL_29:.*]] = %[[VAL_25]] to %[[VAL_28]] step %[[VAL_26]] {
// CHECK:               %[[VAL_30:.*]] = arith.constant 2 : index
// CHECK:               scf.for %[[VAL_31:.*]] = %[[VAL_25]] to %[[VAL_27]] step %[[VAL_30]] {
// CHECK:                 aie.use_lock(%[[VAL_6]], AcquireGreaterEqual, 1)
// CHECK:                 func.call @zero_scalar_i16(%[[VAL_4]]) : (memref<16x16xi16>) -> ()
// CHECK:                 %[[VAL_32:.*]] = arith.constant 2 : index
// CHECK:                 scf.for %[[VAL_33:.*]] = %[[VAL_25]] to %[[VAL_27]] step %[[VAL_32]] {
// CHECK:                   aie.use_lock(%[[VAL_17]], AcquireGreaterEqual, 1)
// CHECK:                   aie.use_lock(%[[VAL_11]], AcquireGreaterEqual, 1)
// CHECK:                   func.call @matmul_scalar_i16_i16(%[[VAL_14]], %[[VAL_8]], %[[VAL_4]]) : (memref<16x8xi16>, memref<8x16xi16>, memref<16x16xi16>) -> ()
// CHECK:                   aie.use_lock(%[[VAL_16]], Release, 1)
// CHECK:                   aie.use_lock(%[[VAL_10]], Release, 1)
// CHECK:                   aie.use_lock(%[[VAL_17]], AcquireGreaterEqual, 1)
// CHECK:                   aie.use_lock(%[[VAL_11]], AcquireGreaterEqual, 1)
// CHECK:                   func.call @matmul_scalar_i16_i16(%[[VAL_15]], %[[VAL_9]], %[[VAL_4]]) : (memref<16x8xi16>, memref<8x16xi16>, memref<16x16xi16>) -> ()
// CHECK:                   aie.use_lock(%[[VAL_16]], Release, 1)
// CHECK:                   aie.use_lock(%[[VAL_10]], Release, 1)
// CHECK:                 }
// CHECK:                 aie.use_lock(%[[VAL_7]], Release, 1)
// CHECK:                 aie.use_lock(%[[VAL_6]], AcquireGreaterEqual, 1)
// CHECK:                 func.call @zero_scalar_i16(%[[VAL_5]]) : (memref<16x16xi16>) -> ()
// CHECK:                 %[[VAL_34:.*]] = arith.constant 2 : index
// CHECK:                 scf.for %[[VAL_35:.*]] = %[[VAL_25]] to %[[VAL_27]] step %[[VAL_34]] {
// CHECK:                   aie.use_lock(%[[VAL_17]], AcquireGreaterEqual, 1)
// CHECK:                   aie.use_lock(%[[VAL_11]], AcquireGreaterEqual, 1)
// CHECK:                   func.call @matmul_scalar_i16_i16(%[[VAL_14]], %[[VAL_8]], %[[VAL_5]]) : (memref<16x8xi16>, memref<8x16xi16>, memref<16x16xi16>) -> ()
// CHECK:                   aie.use_lock(%[[VAL_16]], Release, 1)
// CHECK:                   aie.use_lock(%[[VAL_10]], Release, 1)
// CHECK:                   aie.use_lock(%[[VAL_17]], AcquireGreaterEqual, 1)
// CHECK:                   aie.use_lock(%[[VAL_11]], AcquireGreaterEqual, 1)
// CHECK:                   func.call @matmul_scalar_i16_i16(%[[VAL_15]], %[[VAL_9]], %[[VAL_5]]) : (memref<16x8xi16>, memref<8x16xi16>, memref<16x16xi16>) -> ()
// CHECK:                   aie.use_lock(%[[VAL_16]], Release, 1)
// CHECK:                   aie.use_lock(%[[VAL_10]], Release, 1)
// CHECK:                 }
// CHECK:                 aie.use_lock(%[[VAL_7]], Release, 1)
// CHECK:               }
// CHECK:             }
// CHECK:             aie.end
// CHECK:           }
// CHECK:           aie.shim_dma_allocation @inB(MM2S, 1, 0)
// CHECK:           aie.shim_dma_allocation @outC(S2MM, 0, 0)
// CHECK:           %[[VAL_36:.*]] = aie.mem(%[[VAL_1]]) {
// CHECK:             %[[VAL_37:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:  // 2 preds: ^bb0, ^bb2
// CHECK:             aie.use_lock(%[[VAL_16]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[VAL_14]] : memref<16x8xi16>, 0, 128)
// CHECK:             aie.use_lock(%[[VAL_17]], Release, 1)
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:  // pred: ^bb1
// CHECK:             aie.use_lock(%[[VAL_16]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[VAL_15]] : memref<16x8xi16>, 0, 128)
// CHECK:             aie.use_lock(%[[VAL_17]], Release, 1)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:  // pred: ^bb0
// CHECK:             %[[VAL_38:.*]] = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
// CHECK:           ^bb4:  // 2 preds: ^bb3, ^bb5
// CHECK:             aie.use_lock(%[[VAL_10]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[VAL_8]] : memref<8x16xi16>, 0, 128)
// CHECK:             aie.use_lock(%[[VAL_11]], Release, 1)
// CHECK:             aie.next_bd ^bb5
// CHECK:           ^bb5:  // pred: ^bb4
// CHECK:             aie.use_lock(%[[VAL_10]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[VAL_9]] : memref<8x16xi16>, 0, 128)
// CHECK:             aie.use_lock(%[[VAL_11]], Release, 1)
// CHECK:             aie.next_bd ^bb4
// CHECK:           ^bb6:  // pred: ^bb3
// CHECK:             %[[VAL_39:.*]] = aie.dma_start(MM2S, 0, ^bb7, ^bb9)
// CHECK:           ^bb7:  // 2 preds: ^bb6, ^bb8
// CHECK:             aie.use_lock(%[[VAL_7]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[VAL_4]] : memref<16x16xi16>, 0, 256)
// CHECK:             aie.use_lock(%[[VAL_6]], Release, 1)
// CHECK:             aie.next_bd ^bb8
// CHECK:           ^bb8:  // pred: ^bb7
// CHECK:             aie.use_lock(%[[VAL_7]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[VAL_5]] : memref<16x16xi16>, 0, 256)
// CHECK:             aie.use_lock(%[[VAL_6]], Release, 1)
// CHECK:             aie.next_bd ^bb7
// CHECK:           ^bb9:  // pred: ^bb6
// CHECK:             aie.end
// CHECK:           }

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
          %subview2 = aie.objectfifo.acquire @outC (Produce, 1) : !aie.objectfifosubview<memref<16x16xi16>>
          %elem2 = aie.objectfifo.subview.access %subview2[0] : !aie.objectfifosubview<memref<16x16xi16>> -> memref<16x16xi16>
          func.call @zero_scalar_i16(%elem2) : (memref<16x16xi16>) -> ()

          scf.for %arg3 = %c0 to %c4 step %c1 {
            %subview0 = aie.objectfifo.acquire @inA (Consume, 1) : !aie.objectfifosubview<memref<16x8xi16>>
            %elem0 = aie.objectfifo.subview.access %subview0[0] : !aie.objectfifosubview<memref<16x8xi16>> -> memref<16x8xi16>
            %subview1 = aie.objectfifo.acquire @inB (Consume, 1) : !aie.objectfifosubview<memref<8x16xi16>>
            %elem1 = aie.objectfifo.subview.access %subview1[0] : !aie.objectfifosubview<memref<8x16xi16>> -> memref<8x16xi16>

            func.call @matmul_scalar_i16_i16(%elem0, %elem1, %elem2) : (memref<16x8xi16>, memref<8x16xi16>, memref<16x16xi16>) -> ()

            aie.objectfifo.release @inA (Consume, 1)
            aie.objectfifo.release @inB (Consume, 1)
          }
          aie.objectfifo.release @outC (Produce, 1)
        }
      }
      aie.end
    }
  }
}
