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
// CHECK:           %[[VAL_0:.*]] = aie.tile(2, 2)
// CHECK:           %[[VAL_1:.*]] = aie.buffer(%[[VAL_0]]) {sym_name = "fifo_buff_0"} : memref<i32>
// CHECK:           %[[VAL_2:.*]] = aie.buffer(%[[VAL_0]]) {sym_name = "fifo_buff_1"} : memref<i32>
// CHECK:           %[[VAL_3:.*]] = aie.buffer(%[[VAL_0]]) {sym_name = "fifo_buff_2"} : memref<i32>
// CHECK:           %[[VAL_4:.*]] = aie.buffer(%[[VAL_0]]) {sym_name = "fifo_buff_3"} : memref<i32>
// CHECK:           %[[VAL_5:.*]] = aie.lock(%[[VAL_0]]) {init = 4 : i32, sym_name = "fifo_prod_lock_0"}
// CHECK:           %[[VAL_6:.*]] = aie.lock(%[[VAL_0]]) {init = 0 : i32, sym_name = "fifo_cons_lock_0"}
// CHECK:           %[[VAL_7:.*]] = aie.tile(2, 3)
// CHECK:           %[[VAL_8:.*]] = aie.buffer(%[[VAL_7]]) {sym_name = "buf23"} : memref<4xi32>
// CHECK:           %[[VAL_9:.*]] = aie.core(%[[VAL_0]]) {
// CHECK:             %[[VAL_10:.*]] = arith.constant 1 : i32
// CHECK:             %[[VAL_11:.*]] = arith.constant 88 : i32
// CHECK:             %[[VAL_12:.*]] = arith.constant 77 : i32
// CHECK:             %[[VAL_13:.*]] = arith.constant 66 : i32
// CHECK:             %[[VAL_14:.*]] = arith.constant 55 : i32
// CHECK:             aie.use_lock(%[[VAL_5]], AcquireGreaterEqual, %[[VAL_10]])
// CHECK:             memref.store %[[VAL_14]], %[[VAL_1]][] : memref<i32>
// CHECK:             aie.use_lock(%[[VAL_6]], Release, %[[VAL_10]])
// CHECK:             aie.use_lock(%[[VAL_5]], AcquireGreaterEqual, %[[VAL_10]])
// CHECK:             memref.store %[[VAL_13]], %[[VAL_2]][] : memref<i32>
// CHECK:             aie.use_lock(%[[VAL_6]], Release, %[[VAL_10]])
// CHECK:             aie.use_lock(%[[VAL_5]], AcquireGreaterEqual, %[[VAL_10]])
// CHECK:             memref.store %[[VAL_12]], %[[VAL_3]][] : memref<i32>
// CHECK:             aie.use_lock(%[[VAL_6]], Release, %[[VAL_10]])
// CHECK:             aie.use_lock(%[[VAL_5]], AcquireGreaterEqual, %[[VAL_10]])
// CHECK:             memref.store %[[VAL_11]], %[[VAL_4]][] : memref<i32>
// CHECK:             aie.use_lock(%[[VAL_6]], Release, %[[VAL_10]])
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_15:.*]] = aie.core(%[[VAL_7]]) {
// CHECK:             %[[VAL_16:.*]] = arith.constant 1 : i32
// CHECK:             %[[VAL_17:.*]] = arith.constant 3 : index
// CHECK:             %[[VAL_18:.*]] = arith.constant 2 : index
// CHECK:             %[[VAL_19:.*]] = arith.constant 1 : index
// CHECK:             %[[VAL_20:.*]] = arith.constant 0 : index
// CHECK:             %[[VAL_21:.*]] = arith.constant 2 : i32
// CHECK:             aie.use_lock(%[[VAL_6]], AcquireGreaterEqual, %[[VAL_16]])
// CHECK:             %[[VAL_22:.*]] = memref.load %[[VAL_1]][] : memref<i32>
// CHECK:             memref.store %[[VAL_22]], %[[VAL_8]]{{\[}}%[[VAL_20]]] : memref<4xi32>
// CHECK:             aie.use_lock(%[[VAL_5]], Release, %[[VAL_16]])
// CHECK:             aie.use_lock(%[[VAL_6]], AcquireGreaterEqual, %[[VAL_21]])
// CHECK:             %[[VAL_23:.*]] = memref.load %[[VAL_2]][] : memref<i32>
// CHECK:             %[[VAL_24:.*]] = memref.load %[[VAL_3]][] : memref<i32>
// CHECK:             memref.store %[[VAL_23]], %[[VAL_8]]{{\[}}%[[VAL_19]]] : memref<4xi32>
// CHECK:             memref.store %[[VAL_24]], %[[VAL_8]]{{\[}}%[[VAL_18]]] : memref<4xi32>
// CHECK:             aie.use_lock(%[[VAL_5]], Release, %[[VAL_21]])
// CHECK:             aie.use_lock(%[[VAL_6]], AcquireGreaterEqual, %[[VAL_16]])
// CHECK:             %[[VAL_25:.*]] = memref.load %[[VAL_4]][] : memref<i32>
// CHECK:             memref.store %[[VAL_25]], %[[VAL_8]]{{\[}}%[[VAL_17]]] : memref<4xi32>
// CHECK:             aie.use_lock(%[[VAL_5]], Release, %[[VAL_16]])
// CHECK:             aie.end
// CHECK:           }
// CHECK:         }

module @aie2_cyclostatic_L1 {
    aie.device(xcve2302) {

        %tile22 = aie.tile(2, 2)  // producer tile
        %tile23 = aie.tile(2, 3)  // consumer tile
        %buf23  = aie.buffer(%tile23) {sym_name = "buf23"} : memref<4xi32>

        aie.objectfifo @fifo (%tile22, {%tile23}, 4 : i32) : !aie.objectfifo<memref<i32>>

        %core22 = aie.core(%tile22) {
            %c55 = arith.constant 55 : i32
            %c66 = arith.constant 66 : i32
            %c77 = arith.constant 77 : i32
            %c88 = arith.constant 88 : i32

            %subview0_obj = aie.objectfifo.acquire @fifo(Produce) : memref<i32>
            memref.store %c55, %subview0_obj[] : memref<i32>
            aie.objectfifo.release @fifo(Produce) [1]

            %subview1_obj = aie.objectfifo.acquire @fifo(Produce) : memref<i32>
            memref.store %c66, %subview1_obj[] : memref<i32>
            aie.objectfifo.release @fifo(Produce) [1]

            %subview2_obj = aie.objectfifo.acquire @fifo(Produce) : memref<i32>
            memref.store %c77, %subview2_obj[] : memref<i32>
            aie.objectfifo.release @fifo(Produce) [1]

            %subview3_obj = aie.objectfifo.acquire @fifo(Produce) : memref<i32>
            memref.store %c88, %subview3_obj[] : memref<i32>
            aie.objectfifo.release @fifo(Produce) [1]

            aie.end
        }

        %core23 = aie.core(%tile23) {
            %i0 = arith.constant 0 : index
            %i1 = arith.constant 1 : index
            %i2 = arith.constant 2 : index
            %i3 = arith.constant 3 : index

            %subview0_obj = aie.objectfifo.acquire @fifo(Consume) : memref<i32>
            %v55 = memref.load %subview0_obj[] : memref<i32>
            memref.store %v55, %buf23[%i0] : memref<4xi32>
            aie.objectfifo.release @fifo(Consume) [1]

            %subview1_obj0, %subview1_obj1 = aie.objectfifo.acquire @fifo(Consume) : memref<i32>, memref<i32>
            %v66 = memref.load %subview1_obj0[] : memref<i32>
            %v77 = memref.load %subview1_obj1[] : memref<i32>
            memref.store %v66, %buf23[%i1] : memref<4xi32>
            memref.store %v77, %buf23[%i2] : memref<4xi32>
            aie.objectfifo.release @fifo(Consume) [2]

            %subview2_obj = aie.objectfifo.acquire @fifo(Consume) : memref<i32>
            %v88 = memref.load %subview2_obj[] : memref<i32>
            memref.store %v88, %buf23[%i3] : memref<4xi32>
            aie.objectfifo.release @fifo(Consume) [1]

            aie.end
        }
    }
}
