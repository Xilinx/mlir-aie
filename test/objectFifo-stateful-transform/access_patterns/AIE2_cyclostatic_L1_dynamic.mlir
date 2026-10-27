//===- AIE2_cyclostatic_L1_dynamic.mlir -------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Dynamic counterpart of AIE2_cyclostatic_L1.mlir. Under dynamic lowering (the
// aiecc driver default) a cyclostatic acquire/release pattern selects the active
// buffer at run time via scf.index_switch rather than statically binding each
// access to a fixed buffer. However, after canonicalization, CSE and constant
// folding, this becomes a static access pattern again even without loop
// unrolling (dynamic mode), since there are no loops in this test.

// RUN: aie-opt --aie-objectFifo-stateful-transform --aie-objectFifo-unroll="default-dynamic=true" %s | FileCheck %s

// Consumer pattern {1,2,1} over a depth-4 fifo: each acquire resolves its buffer
// through a runtime index_switch (the hallmark of dynamic lowering).
// CHECK-LABEL:   aie.device(xcve2302) {
// CHECK-DAG:           %[[T22:.*]] = aie.tile(2, 2)
// CHECK-DAG:           %[[T23:.*]] = aie.tile(2, 3)
// CHECK-DAG:           aie.buffer @[[B0:fifo_buff_0]](%[[T22]]) : memref<i32>
// CHECK-DAG:           aie.buffer @[[B1:fifo_buff_1]](%[[T22]]) : memref<i32>
// CHECK-DAG:           aie.buffer @[[B2:fifo_buff_2]](%[[T22]]) : memref<i32>
// CHECK-DAG:           aie.buffer @[[B3:fifo_buff_3]](%[[T22]]) : memref<i32>
// CHECK-DAG:           aie.lock @[[PROD:fifo_prod_lock_0]](%[[T22]]) {init = 4 : i32}
// CHECK-DAG:           aie.lock @[[CONS:fifo_cons_lock_0]](%[[T22]]) {init = 0 : i32}
// CHECK-DAG:           aie.buffer @[[BUF23:buf23]](%[[T23]]) : memref<4xi32>
// CHECK:           %{{.*}} = aie.core(%[[T22]]) {
// CHECK:             %[[C1I:.*]] = arith.constant 1 : i32
// CHECK:             %[[C88:.*]] = arith.constant 88 : i32
// CHECK:             %[[C77:.*]] = arith.constant 77 : i32
// CHECK:             %[[C66:.*]] = arith.constant 66 : i32
// CHECK:             %[[C55:.*]] = arith.constant 55 : i32
// CHECK:             aie.use_lock(@[[PROD]], AcquireGreaterEqual, %[[C1I]])
// CHECK:             memref.store %[[C55]], %{{[^ ,)]+}}[] : memref<i32>
// CHECK:             aie.use_lock(@[[CONS]], Release, %[[C1I]])
// CHECK:             aie.use_lock(@[[PROD]], AcquireGreaterEqual, %[[C1I]])
// CHECK:             memref.store %[[C66]], %{{[^ ,)]+}}[] : memref<i32>
// CHECK:             aie.use_lock(@[[CONS]], Release, %[[C1I]])
// CHECK:             aie.use_lock(@[[PROD]], AcquireGreaterEqual, %[[C1I]])
// CHECK:             memref.store %[[C77]], %{{[^ ,)]+}}[] : memref<i32>
// CHECK:             aie.use_lock(@[[CONS]], Release, %[[C1I]])
// CHECK:             aie.use_lock(@[[PROD]], AcquireGreaterEqual, %[[C1I]])
// CHECK:             memref.store %[[C88]], %{{[^ ,)]+}}[] : memref<i32>
// CHECK:             aie.use_lock(@[[CONS]], Release, %[[C1I]])
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %{{.*}} = aie.core(%[[T23]]) {
// CHECK:             %[[D1I:.*]] = arith.constant 1 : i32
// CHECK:             %[[C3:.*]] = arith.constant 3 : index
// CHECK:             %[[C2:.*]] = arith.constant 2 : index
// CHECK:             %[[C1:.*]] = arith.constant 1 : index
// CHECK:             %[[C0:.*]] = arith.constant 0 : index
// CHECK:             %[[C2I:.*]] = arith.constant 2 : i32
// CHECK:             aie.use_lock(@[[CONS]], AcquireGreaterEqual, %[[D1I]])
// CHECK:             %[[V0:.*]] = memref.load %{{[^ ,)]+}}[] : memref<i32>
// CHECK:             memref.store %[[V0]], %{{[^ ,)]+}}{{\[}}%[[C0]]] : memref<4xi32>
// CHECK:             aie.use_lock(@[[PROD]], Release, %[[D1I]])
// CHECK:             aie.use_lock(@[[CONS]], AcquireGreaterEqual, %[[C2I]])
// CHECK:             %[[V1:.*]] = memref.load %{{[^ ,)]+}}[] : memref<i32>
// CHECK:             %[[V2:.*]] = memref.load %{{[^ ,)]+}}[] : memref<i32>
// CHECK:             memref.store %[[V1]], %{{[^ ,)]+}}{{\[}}%[[C1]]] : memref<4xi32>
// CHECK:             memref.store %[[V2]], %{{[^ ,)]+}}{{\[}}%[[C2]]] : memref<4xi32>
// CHECK:             aie.use_lock(@[[PROD]], Release, %[[C2I]])
// CHECK:             aie.use_lock(@[[CONS]], AcquireGreaterEqual, %[[D1I]])
// CHECK:             %[[V3:.*]] = memref.load %{{[^ ,)]+}}[] : memref<i32>
// CHECK:             memref.store %[[V3]], %{{[^ ,)]+}}{{\[}}%[[C3]]] : memref<4xi32>
// CHECK:             aie.use_lock(@[[PROD]], Release, %[[D1I]])
// CHECK:             aie.end
// CHECK:           }
// CHECK:         }

module @aie2_cyclostatic_L1 {
    aie.device(xcve2302) {

        %tile22 = aie.tile(2, 2)  // producer tile
        %tile23 = aie.tile(2, 3)  // consumer tile
        aie.buffer @buf23(%tile23) : memref<4xi32>

        aie.objectfifo @fifo (%tile22, {%tile23}, 4 : i32) : !aie.objectfifo<memref<i32>>

        %core22 = aie.core(%tile22) {
            %c55 = arith.constant 55 : i32
            %c66 = arith.constant 66 : i32
            %c77 = arith.constant 77 : i32
            %c88 = arith.constant 88 : i32

            %subview0_obj = aie.objectfifo.acquire @fifo (Produce, 1) : memref<i32>
            memref.store %c55, %subview0_obj[] : memref<i32>
            aie.objectfifo.release @fifo (Produce, 1)

            %subview1_obj = aie.objectfifo.acquire @fifo (Produce, 1) : memref<i32>
            memref.store %c66, %subview1_obj[] : memref<i32>
            aie.objectfifo.release @fifo (Produce, 1)

            %subview2_obj = aie.objectfifo.acquire @fifo (Produce, 1) : memref<i32>
            memref.store %c77, %subview2_obj[] : memref<i32>
            aie.objectfifo.release @fifo (Produce, 1)

            %subview3_obj = aie.objectfifo.acquire @fifo (Produce, 1) : memref<i32>
            memref.store %c88, %subview3_obj[] : memref<i32>
            aie.objectfifo.release @fifo (Produce, 1)

            aie.end
        }

        %core23 = aie.core(%tile23) {
            %i0 = arith.constant 0 : index
            %i1 = arith.constant 1 : index
            %i2 = arith.constant 2 : index
            %i3 = arith.constant 3 : index

            %subview0_obj = aie.objectfifo.acquire @fifo (Consume, 1) : memref<i32>
            %v55 = memref.load %subview0_obj[] : memref<i32>
            %buf23 = aie.get_buffer @buf23 : memref<4xi32>
            memref.store %v55, %buf23[%i0] : memref<4xi32>
            aie.objectfifo.release @fifo (Consume, 1)

            %subview1_obj0, %subview1_obj1 = aie.objectfifo.acquire @fifo (Consume, 2) : memref<i32>, memref<i32>
            %v66 = memref.load %subview1_obj0[] : memref<i32>
            %v77 = memref.load %subview1_obj1[] : memref<i32>
            memref.store %v66, %buf23[%i1] : memref<4xi32>
            memref.store %v77, %buf23[%i2] : memref<4xi32>
            aie.objectfifo.release @fifo (Consume, 2)

            %subview2_obj = aie.objectfifo.acquire @fifo (Consume, 1) : memref<i32>
            %v88 = memref.load %subview2_obj[] : memref<i32>
            memref.store %v88, %buf23[%i3] : memref<4xi32>
            aie.objectfifo.release @fifo (Consume, 1)

            aie.end
        }
    }
}
