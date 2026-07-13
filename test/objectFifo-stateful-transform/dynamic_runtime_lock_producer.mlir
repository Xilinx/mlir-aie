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

// RUN: aie-opt --aie-objectFifo-stateful-transform="dynamic-objFifos=true" %s | FileCheck %s

// CHECK-LABEL:   aie.device(npu2) {
// CHECK:           %[[VAL_0:.*]] = aie.tile(0, 2)
// CHECK:           %[[VAL_1:.*]] = aie.tile(0, 3)
// CHECK:           %[[VAL_2:.*]] = aie.buffer(%[[VAL_0]]) {sym_name = "fifo_buff_0"} : memref<8xi8>
// CHECK:           %[[VAL_3:.*]] = aie.buffer(%[[VAL_0]]) {sym_name = "fifo_buff_1"} : memref<8xi8>
// CHECK:           %[[VAL_4:.*]] = aie.buffer(%[[VAL_0]]) {sym_name = "fifo_buff_2"} : memref<8xi8>
// CHECK:           %[[VAL_5:.*]] = aie.buffer(%[[VAL_0]]) {sym_name = "fifo_buff_3"} : memref<8xi8>
// CHECK:           %[[VAL_6:.*]] = aie.lock(%[[VAL_0]], 0) {init = 4 : i32, sym_name = "fifo_prod_lock_0"}
// CHECK:           %[[VAL_7:.*]] = aie.lock(%[[VAL_0]], 1) {init = 0 : i32, sym_name = "fifo_cons_lock_0"}
// CHECK:           %[[VAL_8:.*]] = aie.buffer(%[[VAL_0]]) : memref<1xi32>
// CHECK:           %[[VAL_9:.*]] = aie.core(%[[VAL_0]]) {
// CHECK:             %[[VAL_10:.*]] = arith.constant 0 : i32
// CHECK:             %[[VAL_11:.*]] = arith.constant 0 : i32
// CHECK:             %[[VAL_12:.*]] = arith.constant 0 : index
// CHECK:             %[[VAL_13:.*]] = arith.constant 4 : i32
// CHECK:             memref.store %[[VAL_11]], %[[VAL_8]]{{\[}}%[[VAL_12]]] : memref<1xi32>
// CHECK:             %[[VAL_14:.*]] = arith.constant 0 : index
// CHECK:             %[[VAL_15:.*]] = arith.constant 1 : index
// CHECK:             %[[VAL_16:.*]] = arith.constant 14 : index
// CHECK:             %[[VAL_17:.*]] = scf.for %[[VAL_18:.*]] = %[[VAL_14]] to %[[VAL_16]] step %[[VAL_15]] iter_args(%[[VAL_19:.*]] = %[[VAL_10]]) -> (i32) {
// CHECK:               %[[VAL_20:.*]] = arith.constant 1 : i32
// CHECK:               %[[VAL_21:.*]] = arith.constant 0 : i32
// CHECK:               %[[VAL_22:.*]] = arith.subi %[[VAL_20]], %[[VAL_19]] : i32
// CHECK:               %[[VAL_23:.*]] = arith.maxsi %[[VAL_22]], %[[VAL_21]] : i32
// CHECK:               aie.use_lock(%[[VAL_6]], AcquireGreaterEqual, %[[VAL_23]])
// CHECK:               %[[VAL_24:.*]] = arith.addi %[[VAL_19]], %[[VAL_23]] : i32
// CHECK:               %[[VAL_25:.*]] = memref.load %[[VAL_8]]{{\[}}%[[VAL_12]]] : memref<1xi32>
// CHECK:               %[[VAL_26:.*]] = arith.index_cast %[[VAL_25]] : i32 to index
// CHECK:               %[[VAL_27:.*]] = scf.index_switch %[[VAL_26]] -> memref<8xi8>
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
// CHECK:               %[[VAL_28:.*]] = arith.constant 1 : i32
// CHECK:               aie.use_lock(%[[VAL_7]], Release, %[[VAL_28]])
// CHECK:               %[[VAL_29:.*]] = arith.constant 1 : i32
// CHECK:               %[[VAL_30:.*]] = arith.subi %[[VAL_24]], %[[VAL_29]] : i32
// CHECK:               %[[VAL_31:.*]] = memref.load %[[VAL_8]]{{\[}}%[[VAL_12]]] : memref<1xi32>
// CHECK:               %[[VAL_32:.*]] = arith.constant 1 : i32
// CHECK:               %[[VAL_33:.*]] = arith.addi %[[VAL_31]], %[[VAL_32]] : i32
// CHECK:               %[[VAL_34:.*]] = arith.cmpi sge, %[[VAL_33]], %[[VAL_13]] : i32
// CHECK:               %[[VAL_35:.*]] = arith.subi %[[VAL_33]], %[[VAL_13]] : i32
// CHECK:               %[[VAL_36:.*]] = arith.select %[[VAL_34]], %[[VAL_35]], %[[VAL_33]] : i32
// CHECK:               memref.store %[[VAL_36]], %[[VAL_8]]{{\[}}%[[VAL_12]]] : memref<1xi32>
// CHECK:               scf.yield %[[VAL_30]] : i32
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
