//===- repeat_count_test_dynamic.mlir ---------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Dynamic counterpart of repeat_count_test.mlir. Confirms repeat_count is still
// honored under dynamic lowering (the aiecc driver default): the producer
// acquire/release reflect the repeat count, and the consumer loop is preserved.

// RUN: aie-opt --aie-objectFifo-stateful-transform --aie-objectFifo-unroll="default-dynamic=true" %s | FileCheck %s

// CHECK-LABEL:   aie.device(npu1) {
// CHECK:           %[[T12:.*]] = aie.tile(1, 2)
// CHECK:           %[[T13:.*]] = aie.tile(1, 3)
// CHECK:           %[[CB0:.*]] = aie.buffer(%[[T13]]) {sym_name = "of1_cons_buff_0"} : memref<16xi32>
// CHECK:           %[[CPROD:.*]] = aie.lock(%[[T13]], 0) {init = 1 : i32, sym_name = "of1_cons_prod_lock_0"}
// CHECK:           %[[CCONS:.*]] = aie.lock(%[[T13]], 1) {init = 0 : i32, sym_name = "of1_cons_cons_lock_0"}
// CHECK:           %[[B0:.*]] = aie.buffer(%[[T12]]) {sym_name = "of1_buff_0"} : memref<16xi32>
// CHECK:           %[[PROD:.*]] = aie.lock(%[[T12]], 0) {init = 3 : i32, sym_name = "of1_prod_lock_0"}
// CHECK:           %[[CONS:.*]] = aie.lock(%[[T12]], 1) {init = 0 : i32, sym_name = "of1_cons_lock_0"}
// CHECK:           aie.flow(%[[T12]], DMA : 0, %[[T13]], DMA : 0)
// CHECK:           func.func @some_work(%{{.*}}: memref<16xi32>) {
// CHECK:             return
// CHECK:           }
// CHECK:           %{{.*}} = aie.core(%[[T12]]) {
// CHECK:             %[[C12:.*]] = arith.constant 12 : index
// CHECK:             %[[C1:.*]] = arith.constant 1 : index
// CHECK:             %[[C0:.*]] = arith.constant 0 : index
// CHECK:             %[[C3I:.*]] = arith.constant 3 : i32
// CHECK:             scf.for %{{.*}} = %[[C0]] to %[[C12]] step %[[C1]] {
// CHECK:               aie.use_lock(%[[PROD]], AcquireGreaterEqual, %[[C3I]])
// CHECK:               func.call @some_work(%[[B0]]) : (memref<16xi32>) -> ()
// CHECK:               aie.use_lock(%[[CONS]], Release, %[[C3I]])
// CHECK:             }
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %{{.*}} = aie.mem(%[[T12]]) {
// CHECK:             %[[M1:.*]] = arith.constant 1 : i32
// CHECK:             %{{.*}} = aie.dma_start(MM2S, 0, ^bb1, ^bb2, repeat_count = 2)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[CONS]], AcquireGreaterEqual, %[[M1]])
// CHECK:             aie.dma_bd(%[[B0]] : memref<16xi32> offset = {{.*}} len = {{.*}})
// CHECK:             aie.use_lock(%[[PROD]], Release, %[[M1]])
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb2:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %{{.*}} = aie.mem(%[[T13]]) {
// CHECK:             %[[N1:.*]] = arith.constant 1 : i32
// CHECK:             %{{.*}} = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[CPROD]], AcquireGreaterEqual, %[[N1]])
// CHECK:             aie.dma_bd(%[[CB0]] : memref<16xi32> offset = {{.*}} len = {{.*}})
// CHECK:             aie.use_lock(%[[CCONS]], Release, %[[N1]])
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb2:
// CHECK:             aie.end
// CHECK:           }
// CHECK:         }

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
