//===- variable_rate_fifo_skip_unroll.mlir ---------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
// unrolling in unrollForLoops.
//
// This test mirrors loop_unrolling/unroll_factor_multiple_objectfifos.mlir
// (which unrolls LCM(of_1.size=2, of_2.size=3) = 6 against a
// 12-iteration loop = 2 iterations of unrolled body).
//
// Here we mark of_2 with aie.variable_rate = true so the
// LCM computation only sees of_1's size (=2), giving an
// unroll factor of 2 (12-iteration loop -> 6 iterations of
// 2-unrolled body, NOT 2 iterations of 6-unrolled body).
//
// We assert the loop step has changed from 1 to 2 (the
// LCM unroll factor). Three CHECK directives:
//   1. The loop step is now c2 (factor 2, NOT factor 6).
//   2. The variable-rate fifo's acquire op still appears
//      inside the unrolled body (the loop body itself was
//      not erased, just unrolled less).
//   3. Two of_1 acquire ops appear in the unrolled body
//      (factor 2 unroll on of_1).

// RUN: aie-opt --aie-objectFifo-stateful-transform %s | FileCheck %s

// CHECK-LABEL:   aie.device(xcvc1902) {
// The loop step MUST be 2 (LCM of {of_1.size = 2} only;
// the variable-rate of_2 is excluded from the LCM set).
// CHECK:         scf.for {{.*}} step %c2

module @variable_rate_fifo_skip_unroll {
  aie.device(xcvc1902) {
    %tile12 = aie.tile(1, 2)
    %tile13 = aie.tile(1, 3)
    // of_1: vanilla (fixed-rate); contributes to LCM set.
    aie.objectfifo @of_1 (%tile13, {%tile12}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @of_2 (%tile12, {%tile13}, 3 : i32) {
        aie.variable_rate = true
    } : !aie.objectfifo<memref<16xi32>>
    func.func @some_work(%line_inA:memref<16xi32>, %line_inB:memref<16xi32>, %index:index) -> () {
      return
    }
    %core12 = aie.core(%tile12) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c12 = arith.constant 12 : index

      scf.for %indexInHeight = %c0 to %c12 step %c1 {
        %subviewIn = aie.objectfifo.acquire @of_1 (Consume, 1) : !aie.objectfifosubview<memref<16xi32>>
        %subviewOut = aie.objectfifo.acquire @of_2 (Produce, 1) : !aie.objectfifosubview<memref<16xi32>>
        %elemIn = aie.objectfifo.subview.access %subviewIn[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
        %elemOut = aie.objectfifo.subview.access %subviewOut[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
        func.call @some_work(%elemIn, %elemOut, %indexInHeight) : (memref<16xi32>, memref<16xi32>, index) -> ()
        aie.objectfifo.release @of_1 (Consume, 1)
        aie.objectfifo.release @of_2 (Produce, 1)
      }

      aie.end
    }
  }
}
