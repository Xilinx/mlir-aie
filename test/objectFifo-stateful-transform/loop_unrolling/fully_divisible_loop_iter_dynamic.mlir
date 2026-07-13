//===- fully_divisible_loop_iter_dynamic.mlir -------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Dynamic counterpart of fully_divisible_loop_iter.mlir. Under dynamic lowering
// (the aiecc driver default) the loop is preserved (step 1, single body) with a
// runtime buffer-index switch, instead of being unrolled by the buffer depth.

// RUN: aie-opt --aie-objectFifo-stateful-transform="dynamic-objFifos=true" %s | FileCheck %s

// The loop is NOT unrolled: original step of 1 is kept (static lowering would
// rewrite this to step 2 and emit two acquire/release bodies).
// CHECK:      %[[C1:.*]] = arith.constant 1 : index
// CHECK:      %[[C4:.*]] = arith.constant 4 : index
// CHECK:      scf.for %{{.*}} = %{{.*}} to %[[C4]] step %[[C1]] {
// CHECK:        %{{.*}} = arith.constant 0 : i32
// CHECK:        aie.use_lock(%{{.*}}, Acquire, %{{.*}})
// Runtime buffer selection via index_switch is the hallmark of dynamic lowering.
// CHECK:        scf.index_switch
// CHECK:        func.call @some_work
// CHECK:        %{{.*}} = arith.constant 1 : i32
// CHECK:        aie.use_lock(%{{.*}}, Release, %{{.*}})
// CHECK:      }
// CHECK:      aie.end

module {
  aie.device(xcvc1902) {
    %tile12 = aie.tile(1, 2)
    %tile13 = aie.tile(1, 3)
    aie.objectfifo @loop_of (%tile12, {%tile13}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    func.func @some_work(%line_in:memref<16xi32>, %index:index) -> () {
      return
    }
    %core12 = aie.core(%tile12) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      scf.for %indexInHeight = %c0 to %c4 step %c1 {
        %subview = aie.objectfifo.acquire @loop_of (Produce, 1) : !aie.objectfifosubview<memref<16xi32>>
        %elem0 = aie.objectfifo.subview.access %subview[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
        func.call @some_work(%elem0,%indexInHeight) : (memref<16xi32>,index) -> ()
        aie.objectfifo.release @loop_of (Produce, 1)
      }
      aie.end
    }
  }
}
