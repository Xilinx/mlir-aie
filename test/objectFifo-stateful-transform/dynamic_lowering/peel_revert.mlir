//===- peel_revert.mlir -------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// The release here happens under a condition that differs from iteration to
// iteration, so the core holds a different number of elements each time around
// and peeling the first iteration cannot make the acquires constant. The peel
// is attempted and then thrown away, leaving one loop that still computes its
// acquire as `max(3 - h, 0)`.
//
// Both runs below are checked against the same expectations: peeling enabled
// has to reproduce exactly what peeling disabled produces, which is what makes
// this a check of the revert rather than of the peel.

// RUN: aie-opt --aie-objectFifo-stateful-transform="skip-verify=true" --aie-objectFifo-unroll %s | FileCheck %s
// RUN: aie-opt --aie-objectFifo-stateful-transform="skip-verify=true" --aie-objectFifo-unroll="peel-first-iteration=false" %s | FileCheck %s

// CHECK-LABEL:   aie.device(npu2) {
// CHECK-DAG:           %[[T2:.*]] = aie.tile(0, 2)
// CHECK:           %{{.*}} = aie.core(%[[T2]]) {
// Nothing is acquired ahead of the loop: the peeled copy was discarded.
// CHECK-NOT:         aie.use_lock
// CHECK:             %{{.*}}:2 = scf.for %{{.*}} iter_args(%{{.*}} = %{{.*}}, %[[HELD:.*]] = %{{.*}}) -> (i32, i32) {
// CHECK:               %[[DELTA:.*]] = arith.subi %{{.*}}, %[[HELD]] : i32
// CHECK:               %[[ACQ:.*]] = arith.maxsi %[[DELTA]], %{{.*}} : i32
// CHECK:               aie.use_lock(%{{.*}}, AcquireGreaterEqual, %[[ACQ]])
// CHECK:               %{{.*}} = scf.if
// CHECK:             }
// CHECK-NOT:         scf.for
// CHECK:             aie.end

module {
  aie.device(npu2) {
    %tile_0_1 = aie.tile(0, 1)
    %tile_0_2 = aie.tile(0, 2)

    aie.objectfifo @fifo(%tile_0_1, {%tile_0_2}, 4 : i32) : !aie.objectfifo<memref<8xi8>>

    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c7 = arith.constant 7 : index
      %c14 = arith.constant 14 : index
      scf.for %arg0 = %c0 to %c14 step %c1 {
        %dyn = arith.cmpi slt, %arg0, %c7 : index
        %x_obj0, %x_obj1, %x_obj2 = aie.objectfifo.acquire @fifo(Consume) : memref<8xi8>, memref<8xi8>, memref<8xi8>
        scf.if %dyn {
          aie.objectfifo.release @fifo(Consume) [1]
        }
      }
      aie.end
    }
  }
}
