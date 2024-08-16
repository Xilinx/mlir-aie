//===- loop_test_inner_to_outer.mlir ---------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform %s | FileCheck %s

// CHECK-LABEL:   aie.device(xcvc1902) {
// TODO

module {
  aie.device(xcvc1902) {
    %tile12 = aie.tile(1, 2)
    %tile13 = aie.tile(1, 3)
    aie.objectfifo @of_1 (%tile13, {%tile12}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @of_2 (%tile12, {%tile13}, 3 : i32) : !aie.objectfifo<memref<16xi32>>
    func.func @some_work(%line_inA:memref<16xi32>, %line_inB:memref<16xi32>, %indexH:index, %indexW:index) -> () {
      return
    }
    %core12 = aie.core(%tile12) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c12 = arith.constant 12 : index
      %c13 = arith.constant 13 : index

      // because of the iteration* introduced in the next nested loop, this loop needs
      // to be unrolled as well
      scf.for %indexInHeight = %c0 to %c12 step %c1 {
        // this loop unrolls as for (0 to 12, step = 2) + 1 more iteration*
        scf.for %indexInWidth = %c0 to %c13 step %c1 {
          %subviewIn = aie.objectfifo.acquire @of_1 (Consume, 1) : !aie.objectfifosubview<memref<16xi32>>
          scf.for %i = %c0 to %c13 step %c1 {
            %subviewOut = aie.objectfifo.acquire @of_2 (Produce, 1) : !aie.objectfifosubview<memref<16xi32>>
            %elemIn = aie.objectfifo.subview.access %subviewIn[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
            %elemOut = aie.objectfifo.subview.access %subviewOut[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
            func.call @some_work(%elemIn, %elemOut, %indexInHeight, %indexInWidth) : (memref<16xi32>, memref<16xi32>, index, index) -> ()
            aie.objectfifo.release @of_2 (Produce, 1)
          }
          aie.objectfifo.release @of_1 (Consume, 1)
        }
      }

      aie.end
    }
  }
}
