//===- AIE2_delayed_release_inside_funcs.mlir ------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2023, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// This tests ensures that locks are acquired correctly to preserve the same
// semantics as in the AIE2_delayed_release.mlir test, even if the acquires
// and releases happen across function calls.

// What this boils down to is simply verifying that an error is generated
// when acquire/release is called from within a function. If this behavior
// is ever to be changed, this test can easily be adapted to make sure 
// semantics are preserved.

// RUN: aie-opt --verify-diagnostics --aie-objectFifo-stateful-transform %s 

module @AIE2_delayed_release {
    AIE.device(xcve2302) {
        %tile22 = AIE.tile(2, 2)
        %tile23 = AIE.tile(2, 3)
        %buf23 = AIE.buffer(%tile23) {sym_name = "buf23"} : memref<4xi32>

        AIE.objectfifo @fifo (%tile22, {%tile23}, 4 : i32) : !AIE.objectfifo<memref<i32>>

        // Producer -- produces one element at a time
        %core22 = AIE.core(%tile22) {
            %c99 = arith.constant 99 : i32
            %i0 = arith.constant 0 : index
            %i1 = arith.constant 1 : index
            %i4 = arith.constant 4 : index
            scf.for %it = %i0 to %i4 step %i1 {
                // Produce one 1 element (acquire producer lock) ...
                %subview = AIE.objectfifo.acquire @fifo (Produce, 1) : !AIE.objectfifosubview<memref<i32>>
                %subview_obj = AIE.objectfifo.subview.access %subview[0] : !AIE.objectfifosubview<memref<i32>> -> memref<i32>
                memref.store %c99, %subview_obj[] : memref<i32>
                AIE.objectfifo.release @fifo (Produce, 1)
                // ... done producing (release consumer lock)
            }
            AIE.end
        }

        // The following three functions encapsulate the consumers functionality

        func.func @step1(%buf : memref<4xi32>) -> () {
            %i0 = arith.constant 0 : index
            // Begin consuming 2 elements (acquire consumer lock with value 2)
            // expected-error@+1 {{op must be called from inside a CoreOp}}
            %subview0 = AIE.objectfifo.acquire @fifo (Consume, 2) : !AIE.objectfifosubview<memref<i32>>
            %subview0_obj = AIE.objectfifo.subview.access %subview0[0] : !AIE.objectfifosubview<memref<i32>> -> memref<i32>
            %v0 = memref.load %subview0_obj[] : memref<i32>
            memref.store %v0, %buf[%i0] : memref<4xi32>
            return
        }

        func.func @step2(%buf : memref<4xi32>) -> () {
            %i1 = arith.constant 1 : index
            // For the next step, we only need one element (this could be a subroutine that acquires 1, not knowing that we already acquired 2)
            %subview1 = AIE.objectfifo.acquire @fifo (Consume, 1) : !AIE.objectfifosubview<memref<i32>>
            %subview1_obj = AIE.objectfifo.subview.access %subview1[0] : !AIE.objectfifosubview<memref<i32>> -> memref<i32>
            %v1 = memref.load %subview1_obj[] : memref<i32>
            memref.store %v1, %buf[%i1] : memref<4xi32>
            return
        }

        func.func @step3(%buf : memref<4xi32>) -> () {
            %i2 = arith.constant 2 : index
            // Actually, give us the two from before and one more for three objects total (consumer lock should increase by one)
            %subview2 = AIE.objectfifo.acquire @fifo (Consume, 3) : !AIE.objectfifosubview<memref<i32>>
            %subview2_obj = AIE.objectfifo.subview.access %subview2[0] : !AIE.objectfifosubview<memref<i32>> -> memref<i32>
            %v2 = memref.load %subview2_obj[] : memref<i32>
            memref.store %v2, %buf[%i2] : memref<4xi32>
            return
        }

        func.func @step4(%buf : memref<4xi32>) -> () {
            %i3 = arith.constant 3 : index
            // Now let's just work on one element (consumer lock should not change value)
            %subview3 = AIE.objectfifo.acquire @fifo (Consume, 1) : !AIE.objectfifosubview<memref<i32>>
            %subview3_obj = AIE.objectfifo.subview.access %subview3[0] : !AIE.objectfifosubview<memref<i32>> -> memref<i32>
            %v3 = memref.load %subview3_obj[] : memref<i32>
            memref.store %v3, %buf[%i3] : memref<4xi32>
            return
        }

        // Consumer -- consumes {2, 1, 3, 1}; releases {0, 0, 0, 2}
        %core23 = AIE.core(%tile23) {
            func.call @step1(%buf23) : (memref<4xi32>) -> ()
            func.call @step2(%buf23) : (memref<4xi32>) -> ()
            func.call @step3(%buf23) : (memref<4xi32>) -> ()
            func.call @step4(%buf23) : (memref<4xi32>) -> ()

            // Done, let's release everything we hold (we hold 3 objects from our max acquire)
            AIE.objectfifo.release @fifo (Consume, 3)

            AIE.end
        }
    }
}
