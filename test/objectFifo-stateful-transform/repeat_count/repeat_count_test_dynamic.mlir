//===- repeat_count_test_dynamic.mlir ---------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Dynamic counterpart of repeat_count_test.mlir. Confirms repeat_count is still
// honored under dynamic lowering (the aiecc driver default): the producer
// acquire/release reflect the repeat count, and the consumer loop is preserved.

// RUN: aie-opt --aie-objectFifo-stateful-transform="dynamic-objFifos=true" --aie-assign-lock-ids %s | FileCheck %s

// CHECK:      %[[C12:.*]] = arith.constant 12 : index
// CHECK:      scf.for %{{.*}} = %{{.*}} to %[[C12]] step %{{.*}} {
// repeat_count = 3 surfaces as a count-3 acquire/release, not a multiplied loop.
// CHECK:        %{{.*}} = arith.constant 3 : i32
// CHECK:        aie.use_lock(%{{.*}}, AcquireGreaterEqual, %{{.*}})
// CHECK:        func.call @some_work
// CHECK:        %{{.*}} = arith.constant 3 : i32
// CHECK:        aie.use_lock(%{{.*}}, Release, %{{.*}})
// CHECK:      }
// CHECK:      aie.end

module @repeatCount {
 aie.device(npu1) {
    %tile12 = aie.tile(1, 2)
    %tile13 = aie.tile(1, 3)

    aie.objectfifo @of1 (%tile12, {%tile13}, 1 : i32) {repeat_count = 3 : i32} : !aie.objectfifo<memref<16xi32>>

    func.func @some_work(%lineOut : memref<16xi32>) -> () {
       return
    }

    %core12 = aie.core(%tile12) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %height = arith.constant 12 : index

      scf.for %indexInHeight = %c0 to %height step %c1 {
         %subview = aie.objectfifo.acquire @of1 (Produce, 1) : !aie.objectfifosubview<memref<16xi32>>
         %elem0 = aie.objectfifo.subview.access %subview[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
         func.call @some_work(%elem0) : (memref<16xi32>) -> ()
         aie.objectfifo.release @of1 (Produce, 1)
      }

      aie.end
   }
 }
}
