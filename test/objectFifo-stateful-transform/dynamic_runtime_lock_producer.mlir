//===- dynamic_runtime_lock_producer.mlir ---------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Producer-side acquire/release also uses the runtime held counter. The
// producer acquires the producer lock with a value-carrying
// `AcquireGreaterEqual` (the sign of the value is applied later during the
// core-to-standard lock lowering) and releases the consumer lock.

// RUN: aie-opt --aie-objectFifo-stateful-transform --aie-objectFifo-unroll="default-dynamic=true" %s | FileCheck %s

// CHECK-LABEL:   aie.device(npu2) {
// CHECK:           %[[T2:.*]] = aie.tile(0, 2)
// CHECK:           %[[B0:.*]] = aie.buffer(%[[T2]]) {sym_name = "fifo_buff_0"} : memref<8xi8>
// CHECK:           %[[B1:.*]] = aie.buffer(%[[T2]]) {sym_name = "fifo_buff_1"} : memref<8xi8>
// CHECK:           %[[B2:.*]] = aie.buffer(%[[T2]]) {sym_name = "fifo_buff_2"} : memref<8xi8>
// CHECK:           %[[B3:.*]] = aie.buffer(%[[T2]]) {sym_name = "fifo_buff_3"} : memref<8xi8>
// CHECK:           %[[PROD:.*]] = aie.lock(%[[T2]]) {init = 4 : i32, sym_name = "fifo_prod_lock_0"}
// CHECK:           %[[CONS:.*]] = aie.lock(%[[T2]]) {init = 0 : i32, sym_name = "fifo_cons_lock_0"}
// CHECK:           %{{.*}} = aie.core(%[[T2]]) {
// CHECK:             %[[C14:.*]] = arith.constant 14 : index
// CHECK:             %[[C1:.*]] = arith.constant 1 : index
// CHECK:             %[[C0:.*]] = arith.constant 0 : index
// CHECK:             %[[INIT:.*]] = arith.constant 0 : i32
// CHECK:             %[[ONE:.*]] = arith.constant 1 : i32
// CHECK:             %[[FOUR:.*]] = arith.constant 4 : i32
// CHECK:             %{{.*}} = scf.for %{{.*}} = %[[C0]] to %[[C14]] step %[[C1]] iter_args(%[[IDX:.*]] = %[[INIT]]) -> (i32) {
// CHECK:               aie.use_lock(%[[PROD]], AcquireGreaterEqual, %[[ONE]])
// CHECK:               aie.use_lock(%[[CONS]], Release, %[[ONE]])
// CHECK:               %[[NX:.*]] = arith.addi %[[IDX]], %[[ONE]] : i32
// CHECK:               %[[CMP:.*]] = arith.cmpi sge, %[[NX]], %[[FOUR]] : i32
// CHECK:               %[[SEL:.*]] = arith.select %[[CMP]], %[[INIT]], %[[NX]] : i32
// CHECK:               scf.yield %[[SEL]] : i32
// CHECK:             }
// CHECK:             aie.end
// CHECK:           }
// CHECK:         }

module {
  aie.device(npu2) {
    %tile_0_2 = aie.tile(0, 2)
    %tile_0_3 = aie.tile(0, 3)

    aie.objectfifo @fifo(%tile_0_2, {%tile_0_3}, 4 : i32) : !aie.objectfifo<memref<8xi8>>

    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c14 = arith.constant 14 : index
      scf.for %arg0 = %c0 to %c14 step %c1 {
        %a = aie.objectfifo.acquire @fifo(Produce, 1) : !aie.objectfifosubview<memref<8xi8>>
        %e = aie.objectfifo.subview.access %a[0] : !aie.objectfifosubview<memref<8xi8>> -> memref<8xi8>
        aie.objectfifo.release @fifo(Produce, 1)
      }
      aie.end
    }
  }
}
