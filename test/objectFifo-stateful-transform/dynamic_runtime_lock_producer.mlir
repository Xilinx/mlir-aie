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

// RUN: aie-opt --aie-objectFifo-stateful-transform --aie-objectFifo-unroll="dynamic-objFifos=true" %s | FileCheck %s

// CHECK-LABEL:   aie.device(npu2) {
// CHECK:           %[[VAL_0:.*]] = aie.tile(0, 2)
// CHECK:           %[[VAL_1:.*]] = aie.tile(0, 3)
// CHECK:           %[[VAL_2:.*]] = aie.buffer(%[[VAL_0]]) {sym_name = "fifo_buff_0"} : memref<8xi8>
// CHECK:           %[[VAL_3:.*]] = aie.buffer(%[[VAL_0]]) {sym_name = "fifo_buff_1"} : memref<8xi8>
// CHECK:           %[[VAL_4:.*]] = aie.buffer(%[[VAL_0]]) {sym_name = "fifo_buff_2"} : memref<8xi8>
// CHECK:           %[[VAL_5:.*]] = aie.buffer(%[[VAL_0]]) {sym_name = "fifo_buff_3"} : memref<8xi8>
// CHECK:           %[[VAL_6:.*]] = aie.lock(%[[VAL_0]], 0) {init = 4 : i32, sym_name = "fifo_prod_lock_0"}
// CHECK:           %[[VAL_7:.*]] = aie.lock(%[[VAL_0]], 1) {init = 0 : i32, sym_name = "fifo_cons_lock_0"}
// CHECK:           %[[VAL_9:.*]] = aie.core(%[[VAL_0]]) {
// CHECK:             %[[INIT:.*]] = arith.constant 0 : i32
// CHECK:             %[[LB:.*]] = arith.constant 0 : index
// CHECK:             %[[STEP:.*]] = arith.constant 1 : index
// CHECK:             %[[UB:.*]] = arith.constant 14 : index
// The rotating buffer index and the held-object count are carried as loop
// iter_args (mem2reg-promoted from bookkeeping allocas), not a memref.
// CHECK:             %{{.*}}:2 = scf.for %{{.*}} = %[[LB]] to %[[UB]] step %[[STEP]] iter_args(%[[IDX:.*]] = %[[INIT]], %[[HELD:.*]] = %[[INIT]]) -> (i32, i32) {
// CHECK:               %[[ONE:.*]] = arith.constant 1 : i32
// CHECK:               %[[ZERO:.*]] = arith.constant 0 : i32
// CHECK:               %[[SUB:.*]] = arith.subi %[[ONE]], %[[HELD]] : i32
// CHECK:               %[[DELTA:.*]] = arith.maxsi %[[SUB]], %[[ZERO]] : i32
// CHECK:               aie.use_lock(%[[VAL_6]], AcquireGreaterEqual, %[[DELTA]])
// CHECK:               %[[NEWHELD:.*]] = arith.addi %[[HELD]], %[[DELTA]] : i32
// CHECK:               %[[IDXCAST:.*]] = arith.index_cast %[[IDX]] : i32 to index
// CHECK:               %{{.*}} = scf.index_switch %[[IDXCAST]] -> memref<8xi8>
// CHECK:               case 0 {
// CHECK:                 scf.yield %[[VAL_2]] : memref<8xi8>
// CHECK:               }
// CHECK:               case 1 {
// CHECK:                 scf.yield %[[VAL_3]] : memref<8xi8>
// CHECK:               }
// CHECK:               case 2 {
// CHECK:                 scf.yield %[[VAL_4]] : memref<8xi8>
// CHECK:               }
// CHECK:               case 3 {
// CHECK:                 scf.yield %[[VAL_5]] : memref<8xi8>
// CHECK:               }
// CHECK:               default {
// CHECK:                 scf.yield %[[VAL_2]] : memref<8xi8>
// CHECK:               }
// CHECK:               %[[ONE2:.*]] = arith.constant 1 : i32
// CHECK:               aie.use_lock(%[[VAL_7]], Release, %[[ONE2]])
// CHECK:               %[[RELHELD:.*]] = arith.subi %[[NEWHELD]], %[[ONE2]] : i32
// CHECK:               %[[FOUR:.*]] = arith.constant 4 : i32
// CHECK:               %[[ONE3:.*]] = arith.constant 1 : i32
// CHECK:               %[[NEXTIDX:.*]] = arith.addi %[[IDX]], %[[ONE3]] : i32
// CHECK:               %[[WRAP:.*]] = arith.cmpi sge, %[[NEXTIDX]], %[[FOUR]] : i32
// CHECK:               %[[WRAPPED:.*]] = arith.subi %[[NEXTIDX]], %[[FOUR]] : i32
// CHECK:               %[[SEL:.*]] = arith.select %[[WRAP]], %[[WRAPPED]], %[[NEXTIDX]] : i32
// CHECK:               scf.yield %[[SEL]], %[[RELHELD]] : i32, i32
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
