//===- matmul_test.mlir -----------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2023, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Date: September 5th 2023
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform %s | FileCheck %s

// CHECK:           %[[VAL_0:.*]] = AIE.tile(0, 0)
// CHECK:           %[[VAL_1:.*]] = AIE.tile(0, 2)
// CHECK:           %[[VAL_2:.*]] = AIE.lock(%[[VAL_0]], 4) {init = 0 : i32, sym_name = "outC_cons_prod_lock"}
// CHECK:           %[[VAL_3:.*]] = AIE.lock(%[[VAL_0]], 5) {init = 0 : i32, sym_name = "outC_cons_cons_lock"}
// CHECK:           %[[VAL_4:.*]] = AIE.buffer(%[[VAL_1]]) {sym_name = "outC_buff_0"} : memref<16x16xi16>
// CHECK:           %[[VAL_5:.*]] = AIE.buffer(%[[VAL_1]]) {sym_name = "outC_buff_1"} : memref<16x16xi16>
// CHECK:           %[[VAL_6:.*]] = AIE.lock(%[[VAL_1]], 4) {init = 2 : i32, sym_name = "outC_prod_lock"}
// CHECK:           %[[VAL_7:.*]] = AIE.lock(%[[VAL_1]], 5) {init = 0 : i32, sym_name = "outC_cons_lock"}
// CHECK:           %[[VAL_8:.*]] = AIE.buffer(%[[VAL_1]]) {sym_name = "inB_cons_buff_0"} : memref<8x16xi16>
// CHECK:           %[[VAL_9:.*]] = AIE.buffer(%[[VAL_1]]) {sym_name = "inB_cons_buff_1"} : memref<8x16xi16>
// CHECK:           %[[VAL_10:.*]] = AIE.lock(%[[VAL_1]], 2) {init = 2 : i32, sym_name = "inB_cons_prod_lock"}
// CHECK:           %[[VAL_11:.*]] = AIE.lock(%[[VAL_1]], 3) {init = 0 : i32, sym_name = "inB_cons_cons_lock"}
// CHECK:           %[[VAL_12:.*]] = AIE.lock(%[[VAL_0]], 2) {init = 0 : i32, sym_name = "inB_prod_lock"}
// CHECK:           %[[VAL_13:.*]] = AIE.lock(%[[VAL_0]], 3) {init = 0 : i32, sym_name = "inB_cons_lock"}
// CHECK:           %[[VAL_14:.*]] = AIE.buffer(%[[VAL_1]]) {sym_name = "inA_cons_buff_0"} : memref<16x8xi16>
// CHECK:           %[[VAL_15:.*]] = AIE.buffer(%[[VAL_1]]) {sym_name = "inA_cons_buff_1"} : memref<16x8xi16>
// CHECK:           %[[VAL_16:.*]] = AIE.lock(%[[VAL_1]], 0) {init = 2 : i32, sym_name = "inA_cons_prod_lock"}
// CHECK:           %[[VAL_17:.*]] = AIE.lock(%[[VAL_1]], 1) {init = 0 : i32, sym_name = "inA_cons_cons_lock"}
// CHECK:           %[[VAL_18:.*]] = AIE.lock(%[[VAL_0]], 0) {init = 0 : i32, sym_name = "inA_prod_lock"}
// CHECK:           %[[VAL_19:.*]] = AIE.lock(%[[VAL_0]], 1) {init = 0 : i32, sym_name = "inA_cons_lock"}
// CHECK:           AIE.flow(%[[VAL_0]], DMA : 0, %[[VAL_1]], DMA : 0)
// CHECK:           AIE.flow(%[[VAL_0]], DMA : 1, %[[VAL_1]], DMA : 1)
// CHECK:           AIE.flow(%[[VAL_1]], DMA : 0, %[[VAL_0]], DMA : 0)
// CHECK:           func.func @zero_scalar_i16(%[[VAL_20:.*]]: memref<16x16xi16>) {
// CHECK:             return
// CHECK:           }
// CHECK:           func.func @matmul_scalar_i16_i16(%[[VAL_21:.*]]: memref<16x8xi16>, %[[VAL_22:.*]]: memref<8x16xi16>, %[[VAL_23:.*]]: memref<16x16xi16>) {
// CHECK:             return
// CHECK:           }
// CHECK:           AIE.shimDMAAllocation @inA(MM2S, 0, 0)
// CHECK:           %[[VAL_24:.*]] = AIE.core(%[[VAL_1]]) {
// CHECK:             %[[VAL_25:.*]] = arith.constant 0 : index
// CHECK:             %[[VAL_26:.*]] = arith.constant 1 : index
// CHECK:             %[[VAL_27:.*]] = arith.constant 4 : index
// CHECK:             %[[VAL_28:.*]] = arith.constant 4294967295 : index
// CHECK:             scf.for %[[VAL_29:.*]] = %[[VAL_25]] to %[[VAL_28]] step %[[VAL_26]] {
// CHECK:               %[[VAL_30:.*]] = arith.constant 2 : index
// CHECK:               scf.for %[[VAL_31:.*]] = %[[VAL_25]] to %[[VAL_27]] step %[[VAL_30]] {
// CHECK:                 AIE.useLock(%[[VAL_6]], AcquireGreaterEqual, 1)
// CHECK:                 func.call @zero_scalar_i16(%[[VAL_4]]) : (memref<16x16xi16>) -> ()
// CHECK:                 %[[VAL_32:.*]] = arith.constant 2 : index
// CHECK:                 scf.for %[[VAL_33:.*]] = %[[VAL_25]] to %[[VAL_27]] step %[[VAL_32]] {
// CHECK:                   AIE.useLock(%[[VAL_17]], AcquireGreaterEqual, 1)
// CHECK:                   AIE.useLock(%[[VAL_11]], AcquireGreaterEqual, 1)
// CHECK:                   func.call @matmul_scalar_i16_i16(%[[VAL_14]], %[[VAL_8]], %[[VAL_4]]) : (memref<16x8xi16>, memref<8x16xi16>, memref<16x16xi16>) -> ()
// CHECK:                   AIE.useLock(%[[VAL_16]], Release, 1)
// CHECK:                   AIE.useLock(%[[VAL_10]], Release, 1)
// CHECK:                   AIE.useLock(%[[VAL_17]], AcquireGreaterEqual, 1)
// CHECK:                   AIE.useLock(%[[VAL_11]], AcquireGreaterEqual, 1)
// CHECK:                   func.call @matmul_scalar_i16_i16(%[[VAL_15]], %[[VAL_9]], %[[VAL_4]]) : (memref<16x8xi16>, memref<8x16xi16>, memref<16x16xi16>) -> ()
// CHECK:                   AIE.useLock(%[[VAL_16]], Release, 1)
// CHECK:                   AIE.useLock(%[[VAL_10]], Release, 1)
// CHECK:                 }
// CHECK:                 AIE.useLock(%[[VAL_7]], Release, 1)
// CHECK:                 AIE.useLock(%[[VAL_6]], AcquireGreaterEqual, 1)
// CHECK:                 func.call @zero_scalar_i16(%[[VAL_5]]) : (memref<16x16xi16>) -> ()
// CHECK:                 %[[VAL_34:.*]] = arith.constant 2 : index
// CHECK:                 scf.for %[[VAL_35:.*]] = %[[VAL_25]] to %[[VAL_27]] step %[[VAL_34]] {
// CHECK:                   AIE.useLock(%[[VAL_17]], AcquireGreaterEqual, 1)
// CHECK:                   AIE.useLock(%[[VAL_11]], AcquireGreaterEqual, 1)
// CHECK:                   func.call @matmul_scalar_i16_i16(%[[VAL_14]], %[[VAL_8]], %[[VAL_5]]) : (memref<16x8xi16>, memref<8x16xi16>, memref<16x16xi16>) -> ()
// CHECK:                   AIE.useLock(%[[VAL_16]], Release, 1)
// CHECK:                   AIE.useLock(%[[VAL_10]], Release, 1)
// CHECK:                   AIE.useLock(%[[VAL_17]], AcquireGreaterEqual, 1)
// CHECK:                   AIE.useLock(%[[VAL_11]], AcquireGreaterEqual, 1)
// CHECK:                   func.call @matmul_scalar_i16_i16(%[[VAL_15]], %[[VAL_9]], %[[VAL_5]]) : (memref<16x8xi16>, memref<8x16xi16>, memref<16x16xi16>) -> ()
// CHECK:                   AIE.useLock(%[[VAL_16]], Release, 1)
// CHECK:                   AIE.useLock(%[[VAL_10]], Release, 1)
// CHECK:                 }
// CHECK:                 AIE.useLock(%[[VAL_7]], Release, 1)
// CHECK:               }
// CHECK:             }
// CHECK:             AIE.end
// CHECK:           }
// CHECK:           AIE.shimDMAAllocation @inB(MM2S, 1, 0)
// CHECK:           AIE.shimDMAAllocation @outC(S2MM, 0, 0)
// CHECK:           %[[VAL_36:.*]] = AIE.mem(%[[VAL_1]]) {
// CHECK:             %[[VAL_37:.*]] = AIE.dmaStart(S2MM, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:  // 2 preds: ^bb0, ^bb2
// CHECK:             AIE.useLock(%[[VAL_16]], AcquireGreaterEqual, 1)
// CHECK:             AIE.dmaBd(<%[[VAL_14]] : memref<16x8xi16>, 0, 128>, 0)
// CHECK:             AIE.useLock(%[[VAL_17]], Release, 1)
// CHECK:             AIE.nextBd ^bb2
// CHECK:           ^bb2:  // pred: ^bb1
// CHECK:             AIE.useLock(%[[VAL_16]], AcquireGreaterEqual, 1)
// CHECK:             AIE.dmaBd(<%[[VAL_15]] : memref<16x8xi16>, 0, 128>, 0)
// CHECK:             AIE.useLock(%[[VAL_17]], Release, 1)
// CHECK:             AIE.nextBd ^bb1
// CHECK:           ^bb3:  // pred: ^bb0
// CHECK:             %[[VAL_38:.*]] = AIE.dmaStart(S2MM, 1, ^bb4, ^bb6)
// CHECK:           ^bb4:  // 2 preds: ^bb3, ^bb5
// CHECK:             AIE.useLock(%[[VAL_10]], AcquireGreaterEqual, 1)
// CHECK:             AIE.dmaBd(<%[[VAL_8]] : memref<8x16xi16>, 0, 128>, 0)
// CHECK:             AIE.useLock(%[[VAL_11]], Release, 1)
// CHECK:             AIE.nextBd ^bb5
// CHECK:           ^bb5:  // pred: ^bb4
// CHECK:             AIE.useLock(%[[VAL_10]], AcquireGreaterEqual, 1)
// CHECK:             AIE.dmaBd(<%[[VAL_9]] : memref<8x16xi16>, 0, 128>, 0)
// CHECK:             AIE.useLock(%[[VAL_11]], Release, 1)
// CHECK:             AIE.nextBd ^bb4
// CHECK:           ^bb6:  // pred: ^bb3
// CHECK:             %[[VAL_39:.*]] = AIE.dmaStart(MM2S, 0, ^bb7, ^bb9)
// CHECK:           ^bb7:  // 2 preds: ^bb6, ^bb8
// CHECK:             AIE.useLock(%[[VAL_7]], AcquireGreaterEqual, 1)
// CHECK:             AIE.dmaBd(<%[[VAL_4]] : memref<16x16xi16>, 0, 256>, 0)
// CHECK:             AIE.useLock(%[[VAL_6]], Release, 1)
// CHECK:             AIE.nextBd ^bb8
// CHECK:           ^bb8:  // pred: ^bb7
// CHECK:             AIE.useLock(%[[VAL_7]], AcquireGreaterEqual, 1)
// CHECK:             AIE.dmaBd(<%[[VAL_5]] : memref<16x16xi16>, 0, 256>, 0)
// CHECK:             AIE.useLock(%[[VAL_6]], Release, 1)
// CHECK:             AIE.nextBd ^bb7
// CHECK:           ^bb9:  // pred: ^bb6
// CHECK:             AIE.end
// CHECK:           }

module @matmul {
  AIE.device(xcve2302) {

    %t00 = AIE.tile(0, 0)
    %t02 = AIE.tile(0, 2)

    AIE.objectfifo @inA  (%t00, { %t02 }, 2 : i32) : !AIE.objectfifo<memref<16x8xi16>>
    AIE.objectfifo @inB  (%t00, { %t02 }, 2 : i32) : !AIE.objectfifo<memref<8x16xi16>>
    AIE.objectfifo @outC (%t02, { %t00 }, 2 : i32) : !AIE.objectfifo<memref<16x16xi16>>

    func.func @zero_scalar_i16(%elem0 : memref<16x16xi16>) -> () { return }
    func.func @matmul_scalar_i16_i16(%elem0 : memref<16x8xi16>, %elem1 : memref<8x16xi16>, %elem2 : memref<16x16xi16>) -> () { return }

    AIE.core(%t02) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      %intmax = arith.constant 0xFFFFFFFF : index

      scf.for %reps = %c0 to %intmax step %c1 {

        scf.for %arg2 = %c0 to %c4 step %c1 {
          %subview2 = AIE.objectfifo.acquire @outC (Produce, 1) : !AIE.objectfifosubview<memref<16x16xi16>>
          %elem2 = AIE.objectfifo.subview.access %subview2[0] : !AIE.objectfifosubview<memref<16x16xi16>> -> memref<16x16xi16>
          func.call @zero_scalar_i16(%elem2) : (memref<16x16xi16>) -> ()

          scf.for %arg3 = %c0 to %c4 step %c1 {
            %subview0 = AIE.objectfifo.acquire @inA (Consume, 1) : !AIE.objectfifosubview<memref<16x8xi16>>
            %elem0 = AIE.objectfifo.subview.access %subview0[0] : !AIE.objectfifosubview<memref<16x8xi16>> -> memref<16x8xi16>
            %subview1 = AIE.objectfifo.acquire @inB (Consume, 1) : !AIE.objectfifosubview<memref<8x16xi16>>
            %elem1 = AIE.objectfifo.subview.access %subview1[0] : !AIE.objectfifosubview<memref<8x16xi16>> -> memref<8x16xi16>

            func.call @matmul_scalar_i16_i16(%elem0, %elem1, %elem2) : (memref<16x8xi16>, memref<8x16xi16>, memref<16x16xi16>) -> ()

            AIE.objectfifo.release @inA (Consume, 1)
            AIE.objectfifo.release @inB (Consume, 1)
          }
          AIE.objectfifo.release @outC (Produce, 1)
        }
      }
      AIE.end
    }
  }
}
