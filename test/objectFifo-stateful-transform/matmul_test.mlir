//===- matmul_test.mlir -----------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2023, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Date: September 5th 2023
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform %s | FileCheck %s

//CHECK: %0 = AIE.tile(0, 0)
//CHECK: %1 = AIE.tile(0, 2)
//CHECK: %2 = AIE.lock(%0, 4) {init = 0 : i32, sym_name = "outC_cons_prod_lock"}
//CHECK: %3 = AIE.lock(%0, 5) {init = 0 : i32, sym_name = "outC_cons_cons_lock"}
//CHECK: %4 = AIE.buffer(%1) {sym_name = "outC_buff_0"} : memref<16x16xi16>
//CHECK: %5 = AIE.buffer(%1) {sym_name = "outC_buff_1"} : memref<16x16xi16>
//CHECK: %6 = AIE.lock(%1, 4) {init = 2 : i32, sym_name = "outC_prod_lock"}
//CHECK: %7 = AIE.lock(%1, 5) {init = 0 : i32, sym_name = "outC_cons_lock"}
//CHECK: %8 = AIE.buffer(%1) {sym_name = "inB_cons_buff_0"} : memref<8x16xi16>
//CHECK: %9 = AIE.buffer(%1) {sym_name = "inB_cons_buff_1"} : memref<8x16xi16>
//CHECK: %10 = AIE.lock(%1, 2) {init = 2 : i32, sym_name = "inB_cons_prod_lock"}
//CHECK: %11 = AIE.lock(%1, 3) {init = 0 : i32, sym_name = "inB_cons_cons_lock"}
//CHECK: %12 = AIE.lock(%0, 2) {init = 0 : i32, sym_name = "inB_prod_lock"}
//CHECK: %13 = AIE.lock(%0, 3) {init = 0 : i32, sym_name = "inB_cons_lock"}
//CHECK: %14 = AIE.buffer(%1) {sym_name = "inA_cons_buff_0"} : memref<16x8xi16>
//CHECK: %15 = AIE.buffer(%1) {sym_name = "inA_cons_buff_1"} : memref<16x8xi16>
//CHECK: %16 = AIE.lock(%1, 0) {init = 2 : i32, sym_name = "inA_cons_prod_lock"}
//CHECK: %17 = AIE.lock(%1, 1) {init = 0 : i32, sym_name = "inA_cons_cons_lock"}
//CHECK: %18 = AIE.lock(%0, 0) {init = 0 : i32, sym_name = "inA_prod_lock"}
//CHECK: %19 = AIE.lock(%0, 1) {init = 0 : i32, sym_name = "inA_cons_lock"}
//CHECK: AIE.flow(%0, DMA : 0, %1, DMA : 0)
//CHECK: AIE.flow(%0, DMA : 1, %1, DMA : 1)
//CHECK: AIE.flow(%1, DMA : 0, %0, DMA : 0)
//CHECK: func.func @zero_scalar_i16(%arg0: memref<16x16xi16>) {
//CHECK:   return
//CHECK: }
//CHECK: func.func @matmul_scalar_i16_i16(%arg0: memref<16x8xi16>, %arg1: memref<8x16xi16>, %arg2: memref<16x16xi16>) {
//CHECK:   return
//CHECK: }
//CHECK: AIE.shimDMAAllocation @inA(MM2S, 0, 0)
//CHECK: %20 = AIE.core(%1) {
//CHECK:   %c0 = arith.constant 0 : index
//CHECK:   %c1 = arith.constant 1 : index
//CHECK:   %c4 = arith.constant 4 : index
//CHECK:   %c4294967295 = arith.constant 4294967295 : index
//CHECK:   scf.for %arg0 = %c0 to %c4294967295 step %c1 {
//CHECK:     %c2 = arith.constant 2 : index
//CHECK:     scf.for %arg1 = %c0 to %c4 step %c2 {
//CHECK:       AIE.useLock(%6, AcquireGreaterEqual, 1)
//CHECK:       func.call @zero_scalar_i16(%4) : (memref<16x16xi16>) -> ()
//CHECK:       %c2_0 = arith.constant 2 : index
//CHECK:       scf.for %arg2 = %c0 to %c4 step %c2_0 {
//CHECK:         AIE.useLock(%17, AcquireGreaterEqual, 1)
//CHECK:         AIE.useLock(%11, AcquireGreaterEqual, 1)
//CHECK:         func.call @matmul_scalar_i16_i16(%14, %8, %4) : (memref<16x8xi16>, memref<8x16xi16>, memref<16x16xi16>) -> ()
//CHECK:         AIE.useLock(%16, Release, 1)
//CHECK:         AIE.useLock(%10, Release, 1)
//CHECK:         AIE.useLock(%17, AcquireGreaterEqual, 1)
//CHECK:         AIE.useLock(%11, AcquireGreaterEqual, 1)
//CHECK:         func.call @matmul_scalar_i16_i16(%15, %9, %4) : (memref<16x8xi16>, memref<8x16xi16>, memref<16x16xi16>) -> ()
//CHECK:         AIE.useLock(%16, Release, 1)
//CHECK:         AIE.useLock(%10, Release, 1)
//CHECK:       }
//CHECK:       AIE.useLock(%7, Release, 1)
//CHECK:       AIE.useLock(%6, AcquireGreaterEqual, 1)
//CHECK:       func.call @zero_scalar_i16(%5) : (memref<16x16xi16>) -> ()
//CHECK:       %c2_1 = arith.constant 2 : index
//CHECK:       scf.for %arg2 = %c0 to %c4 step %c2_1 {
//CHECK:         AIE.useLock(%17, AcquireGreaterEqual, 1)
//CHECK:         AIE.useLock(%11, AcquireGreaterEqual, 1)
//CHECK:         func.call @matmul_scalar_i16_i16(%14, %8, %5) : (memref<16x8xi16>, memref<8x16xi16>, memref<16x16xi16>) -> ()
//CHECK:         AIE.useLock(%16, Release, 1)
//CHECK:         AIE.useLock(%10, Release, 1)
//CHECK:         AIE.useLock(%17, AcquireGreaterEqual, 1)
//CHECK:         AIE.useLock(%11, AcquireGreaterEqual, 1)
//CHECK:         func.call @matmul_scalar_i16_i16(%15, %9, %5) : (memref<16x8xi16>, memref<8x16xi16>, memref<16x16xi16>) -> ()
//CHECK:         AIE.useLock(%16, Release, 1)
//CHECK:         AIE.useLock(%10, Release, 1)
//CHECK:       }
//CHECK:       AIE.useLock(%7, Release, 1)
//CHECK:     }
//CHECK:   }
//CHECK:   AIE.end
//CHECK: }
//CHECK: AIE.shimDMAAllocation @inB(MM2S, 1, 0)
//CHECK: AIE.shimDMAAllocation @outC(S2MM, 0, 0)
//CHECK: %21 = AIE.mem(%1) {
//CHECK:   %22 = AIE.dmaStart(S2MM, 0, ^bb1, ^bb3)
//CHECK: ^bb1:  // 2 preds: ^bb0, ^bb2
//CHECK:   AIE.useLock(%16, AcquireGreaterEqual, 1)
//CHECK:   AIE.dmaBd(<%14 : memref<16x8xi16>, 0, 128>, 0)
//CHECK:   AIE.useLock(%17, Release, 1)
//CHECK:   AIE.nextBd ^bb2
//CHECK: ^bb2:  // pred: ^bb1
//CHECK:   AIE.useLock(%16, AcquireGreaterEqual, 1)
//CHECK:   AIE.dmaBd(<%15 : memref<16x8xi16>, 0, 128>, 0)
//CHECK:   AIE.useLock(%17, Release, 1)
//CHECK:   AIE.nextBd ^bb1
//CHECK: ^bb3:  // pred: ^bb0
//CHECK:   %23 = AIE.dmaStart(S2MM, 1, ^bb4, ^bb6)
//CHECK: ^bb4:  // 2 preds: ^bb3, ^bb5
//CHECK:   AIE.useLock(%10, AcquireGreaterEqual, 1)
//CHECK:   AIE.dmaBd(<%8 : memref<8x16xi16>, 0, 128>, 0)
//CHECK:   AIE.useLock(%11, Release, 1)
//CHECK:   AIE.nextBd ^bb5
//CHECK: ^bb5:  // pred: ^bb4
//CHECK:   AIE.useLock(%10, AcquireGreaterEqual, 1)
//CHECK:   AIE.dmaBd(<%9 : memref<8x16xi16>, 0, 128>, 0)
//CHECK:   AIE.useLock(%11, Release, 1)
//CHECK:   AIE.nextBd ^bb4
//CHECK: ^bb6:  // pred: ^bb3
//CHECK:   %24 = AIE.dmaStart(MM2S, 0, ^bb7, ^bb9)
//CHECK: ^bb7:  // 2 preds: ^bb6, ^bb8
//CHECK:   AIE.useLock(%7, AcquireGreaterEqual, 1)
//CHECK:   AIE.dmaBd(<%4 : memref<16x16xi16>, 0, 256>, 0)
//CHECK:   AIE.useLock(%6, Release, 1)
//CHECK:   AIE.nextBd ^bb8
//CHECK: ^bb8:  // pred: ^bb7
//CHECK:   AIE.useLock(%7, AcquireGreaterEqual, 1)
//CHECK:   AIE.dmaBd(<%5 : memref<16x16xi16>, 0, 256>, 0)
//CHECK:   AIE.useLock(%6, Release, 1)
//CHECK:   AIE.nextBd ^bb7
//CHECK: ^bb9:  // pred: ^bb6
//CHECK:   AIE.end
//CHECK: }

module @matmul {
  AIE.device(xcve2302) {
  
    %t00 = AIE.tile(0, 0)
    %t02 = AIE.tile(0, 2)
  
    AIE.objectFifo @inA  (%t00, { %t02 }, 2 : i32) : !AIE.objectFifo<memref<16x8xi16>>
    AIE.objectFifo @inB  (%t00, { %t02 }, 2 : i32) : !AIE.objectFifo<memref<8x16xi16>>
    AIE.objectFifo @outC (%t02, { %t00 }, 2 : i32) : !AIE.objectFifo<memref<16x16xi16>>
  
    func.func @zero_scalar_i16(%elem0 : memref<16x16xi16>) -> () { return }
    func.func @matmul_scalar_i16_i16(%elem0 : memref<16x8xi16>, %elem1 : memref<8x16xi16>, %elem2 : memref<16x16xi16>) -> () { return }
  
    AIE.core(%t02) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      %intmax = arith.constant 0xFFFFFFFF : index

      scf.for %reps = %c0 to %intmax step %c1 {
  
        scf.for %arg2 = %c0 to %c4 step %c1 {
          %subview2 = AIE.objectFifo.acquire @outC (Produce, 1) : !AIE.objectFifoSubview<memref<16x16xi16>>
          %elem2 = AIE.objectFifo.subview.access %subview2[0] : !AIE.objectFifoSubview<memref<16x16xi16>> -> memref<16x16xi16>
          func.call @zero_scalar_i16(%elem2) : (memref<16x16xi16>) -> ()

          scf.for %arg3 = %c0 to %c4 step %c1 {
            %subview0 = AIE.objectFifo.acquire @inA (Consume, 1) : !AIE.objectFifoSubview<memref<16x8xi16>>
            %elem0 = AIE.objectFifo.subview.access %subview0[0] : !AIE.objectFifoSubview<memref<16x8xi16>> -> memref<16x8xi16>
            %subview1 = AIE.objectFifo.acquire @inB (Consume, 1) : !AIE.objectFifoSubview<memref<8x16xi16>>
            %elem1 = AIE.objectFifo.subview.access %subview1[0] : !AIE.objectFifoSubview<memref<8x16xi16>> -> memref<8x16xi16>

            func.call @matmul_scalar_i16_i16(%elem0, %elem1, %elem2) : (memref<16x8xi16>, memref<8x16xi16>, memref<16x16xi16>) -> ()

            AIE.objectFifo.release @inA (Consume, 1)
            AIE.objectFifo.release @inB (Consume, 1)
          }
          AIE.objectFifo.release @outC (Produce, 1)
        }

      }
 
      AIE.end
  
    }
  }
}
