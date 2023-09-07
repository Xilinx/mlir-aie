//===- matmul_test.mlir -----------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2023, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Date: September 5th 2023
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform %s | FileCheck %s

// CHECK:  %0 = AIE.tile(0, 0)
// CHECK:  %1 = AIE.tile(0, 2)
// CHECK:  AIE.flow(%0, DMA : 0, %1, DMA : 0)
// CHECK:  %2 = AIE.lock(%0, 0) {init = 0 : i32, sym_name = "inA_prod_lock"}
// CHECK:  %3 = AIE.lock(%0, 1) {init = 0 : i32, sym_name = "inA_cons_lock"}
// CHECK:  %4 = AIE.buffer(%1) {sym_name = "inA_cons_buff_0"} : memref<16x8xi16>
// CHECK:  %5 = AIE.buffer(%1) {sym_name = "inA_cons_buff_1"} : memref<16x8xi16>
// CHECK:  %6 = AIE.lock(%1, 0) {init = 2 : i32, sym_name = "inA_cons_prod_lock"}
// CHECK:  %7 = AIE.lock(%1, 1) {init = 0 : i32, sym_name = "inA_cons_cons_lock"}
// CHECK:  AIE.flow(%0, DMA : 1, %1, DMA : 1)
// CHECK:  %8 = AIE.lock(%0, 2) {init = 0 : i32, sym_name = "inB_prod_lock"}
// CHECK:  %9 = AIE.lock(%0, 3) {init = 0 : i32, sym_name = "inB_cons_lock"}
// CHECK:  %10 = AIE.buffer(%1) {sym_name = "inB_cons_buff_0"} : memref<8x16xi16>
// CHECK:  %11 = AIE.buffer(%1) {sym_name = "inB_cons_buff_1"} : memref<8x16xi16>
// CHECK:  %12 = AIE.lock(%1, 2) {init = 2 : i32, sym_name = "inB_cons_prod_lock"}
// CHECK:  %13 = AIE.lock(%1, 3) {init = 0 : i32, sym_name = "inB_cons_cons_lock"}
// CHECK:  AIE.flow(%1, DMA : 0, %0, DMA : 0)
// CHECK:  %14 = AIE.buffer(%1) {sym_name = "outC_buff_0"} : memref<16x16xi16>
// CHECK:  %15 = AIE.buffer(%1) {sym_name = "outC_buff_1"} : memref<16x16xi16>
// CHECK:  %16 = AIE.lock(%1, 4) {init = 2 : i32, sym_name = "outC_prod_lock"}
// CHECK:  %17 = AIE.lock(%1, 5) {init = 0 : i32, sym_name = "outC_cons_lock"}
// CHECK:  %18 = AIE.lock(%0, 4) {init = 0 : i32, sym_name = "outC_cons_prod_lock"}
// CHECK:  %19 = AIE.lock(%0, 5) {init = 0 : i32, sym_name = "outC_cons_cons_lock"}
// CHECK:  func.func @zero_scalar_i16(%arg0: memref<16x16xi16>) {
// CHECK:    return
// CHECK:  }
// CHECK:  func.func @matmul_scalar_i16_i16(%arg0: memref<16x8xi16>, %arg1: memref<8x16xi16>, %arg2: memref<16x16xi16>) {
// CHECK:    return
// CHECK:  }
// CHECK:  AIE.shimDMAAllocation @inA(MM2S, 0, 0)
// CHECK:  %20 = AIE.core(%1) {
// CHECK:    %c0 = arith.constant 0 : index
// CHECK:    %c1 = arith.constant 1 : index
// CHECK:    %c4 = arith.constant 4 : index
// CHECK:    %c4294967295 = arith.constant 4294967295 : index
// CHECK:    scf.for %arg0 = %c0 to %c4294967295 step %c1 {
// CHECK:      %c2 = arith.constant 2 : index
// CHECK:      scf.for %arg1 = %c0 to %c4 step %c2 {
// CHECK:        AIE.useLock(%16, AcquireGreaterEqual, 1)
// CHECK:        func.call @zero_scalar_i16(%14) : (memref<16x16xi16>) -> ()
// CHECK:        %c2_0 = arith.constant 2 : index
// CHECK:          scf.for %arg2 = %c0 to %c4 step %c2_0 {
// CHECK:          AIE.useLock(%7, AcquireGreaterEqual, 1)
// CHECK:          AIE.useLock(%13, AcquireGreaterEqual, 1)
// CHECK:          func.call @matmul_scalar_i16_i16(%4, %10, %14) : (memref<16x8xi16>, memref<8x16xi16>, memref<16x16xi16>) -> ()
// CHECK:          AIE.useLock(%6, Release, 1)
// CHECK:          AIE.useLock(%12, Release, 1)
// CHECK:          AIE.useLock(%7, AcquireGreaterEqual, 1)
// CHECK:          AIE.useLock(%13, AcquireGreaterEqual, 1)
// CHECK:          func.call @matmul_scalar_i16_i16(%5, %11, %14) : (memref<16x8xi16>, memref<8x16xi16>, memref<16x16xi16>) -> ()
// CHECK:          AIE.useLock(%6, Release, 1)
// CHECK:          AIE.useLock(%12, Release, 1)
// CHECK:        }
// CHECK:        AIE.useLock(%17, Release, 1)
// CHECK:        AIE.useLock(%16, AcquireGreaterEqual, 1)
// CHECK:        func.call @zero_scalar_i16(%15) : (memref<16x16xi16>) -> ()
// CHECK:        %c2_1 = arith.constant 2 : index
// CHECK:        scf.for %arg2 = %c0 to %c4 step %c2_1 {
// CHECK:          AIE.useLock(%7, AcquireGreaterEqual, 1)
// CHECK:          AIE.useLock(%13, AcquireGreaterEqual, 1)
// CHECK:          func.call @matmul_scalar_i16_i16(%4, %10, %15) : (memref<16x8xi16>, memref<8x16xi16>, memref<16x16xi16>) -> ()
// CHECK:          AIE.useLock(%6, Release, 1)
// CHECK:          AIE.useLock(%12, Release, 1)
// CHECK:          AIE.useLock(%7, AcquireGreaterEqual, 1)
// CHECK:          AIE.useLock(%13, AcquireGreaterEqual, 1)
// CHECK:          func.call @matmul_scalar_i16_i16(%5, %11, %15) : (memref<16x8xi16>, memref<8x16xi16>, memref<16x16xi16>) -> ()
// CHECK:          AIE.useLock(%6, Release, 1)
// CHECK:          AIE.useLock(%12, Release, 1)
// CHECK:        }
// CHECK:        AIE.useLock(%17, Release, 1)
// CHECK:      }
// CHECK:    }
// CHECK:    AIE.end
// CHECK:  }

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
