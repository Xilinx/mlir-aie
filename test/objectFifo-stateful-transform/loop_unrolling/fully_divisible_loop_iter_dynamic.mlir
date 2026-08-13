//===- fully_divisible_loop_iter_dynamic.mlir -------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Dynamic counterpart of fully_divisible_loop_iter.mlir. Under dynamic lowering
// (the aiecc driver default) the loop is preserved (step 1, single body) with a
// runtime buffer-index switch, instead of being unrolled by the buffer depth.

// RUN: aie-opt --aie-objectFifo-stateful-transform --aie-objectFifo-unroll="default-dynamic=true" %s | FileCheck %s

// The loop is NOT unrolled: original step of 1 is kept (static lowering would
// rewrite this to step 2 and emit two acquire/release bodies).
// CHECK-LABEL:   aie.device(xcvc1902) {
// CHECK:           %[[T12:.*]] = aie.tile(1, 2)
// CHECK:           %[[B0:.*]] = aie.buffer(%[[T12]]) {sym_name = "loop_of_buff_0"} : memref<16xi32>
// CHECK:           %[[B1:.*]] = aie.buffer(%[[T12]]) {sym_name = "loop_of_buff_1"} : memref<16xi32>
// CHECK:           %[[L0:.*]] = aie.lock(%[[T12]]) {init = 0 : i32, sym_name = "loop_of_lock_0"}
// CHECK:           %[[L1:.*]] = aie.lock(%[[T12]]) {init = 0 : i32, sym_name = "loop_of_lock_1"}
// CHECK:           func.func @some_work(%{{.*}}: memref<16xi32>, %{{.*}}: index) {
// CHECK:             return
// CHECK:           }
// CHECK:           %{{.*}} = aie.core(%[[T12]]) {
// CHECK:             %[[C4:.*]] = arith.constant 4 : index
// CHECK:             %[[C1:.*]] = arith.constant 1 : index
// CHECK:             %[[C0:.*]] = arith.constant 0 : index
// CHECK:             %[[C0I:.*]] = arith.constant 0 : i32
// CHECK:             %[[C1I:.*]] = arith.constant 1 : i32
// CHECK:             %[[C2I:.*]] = arith.constant 2 : i32
// CHECK:             %{{.*}} = scf.for %[[IV:.*]] = %[[C0]] to %[[C4]] step %[[C1]] iter_args(%[[IDX:.*]] = %[[C0I]]) -> (i32) {
// CHECK:               %[[LC0:.*]] = arith.index_cast %[[IDX]] : i32 to index
// CHECK:               %[[LSEL:.*]] = scf.index_switch %[[LC0]] -> index
// CHECK:               case 0 {
// CHECK:                 scf.yield %[[L0]] : index
// CHECK:               }
// CHECK:               case 1 {
// CHECK:                 scf.yield %[[L1]] : index
// CHECK:               }
// CHECK:               default {
// CHECK:                 scf.yield %[[L0]] : index
// CHECK:               }
// CHECK:               aie.use_lock(%[[LSEL]], Acquire, %[[C0I]])
// CHECK:               %[[BC:.*]] = arith.index_cast %[[IDX]] : i32 to index
// CHECK:               %[[BSEL:.*]] = scf.index_switch %[[BC]] -> memref<16xi32>
// CHECK:               case 0 {
// CHECK:                 scf.yield %[[B0]] : memref<16xi32>
// CHECK:               }
// CHECK:               case 1 {
// CHECK:                 scf.yield %[[B1]] : memref<16xi32>
// CHECK:               }
// CHECK:               default {
// CHECK:                 scf.yield %[[B0]] : memref<16xi32>
// CHECK:               }
// CHECK:               func.call @some_work(%[[BSEL]], %[[IV]]) : (memref<16xi32>, index) -> ()
// CHECK:               %[[LC1:.*]] = arith.index_cast %[[IDX]] : i32 to index
// CHECK:               %[[RSEL:.*]] = scf.index_switch %[[LC1]] -> index
// CHECK:               case 0 {
// CHECK:                 scf.yield %[[L0]] : index
// CHECK:               }
// CHECK:               case 1 {
// CHECK:                 scf.yield %[[L1]] : index
// CHECK:               }
// CHECK:               default {
// CHECK:                 scf.yield %[[L0]] : index
// CHECK:               }
// CHECK:               aie.use_lock(%[[RSEL]], Release, %[[C1I]])
// CHECK:               %[[NX:.*]] = arith.addi %[[IDX]], %[[C1I]] : i32
// CHECK:               %[[CMP:.*]] = arith.cmpi sge, %[[NX]], %[[C2I]] : i32
// CHECK:               %[[SEL:.*]] = arith.select %[[CMP]], %[[C0I]], %[[NX]] : i32
// CHECK:               scf.yield %[[SEL]] : i32
// CHECK:             }
// CHECK:             aie.end
// CHECK:           }
// CHECK:         }

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
