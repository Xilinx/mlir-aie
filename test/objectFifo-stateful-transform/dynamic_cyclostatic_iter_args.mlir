//===- dynamic_cyclostatic_iter_args.mlir ---------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// Pre-implementation spec for the cyclostatic-acquire peel rewrite.
//
// When the loop carries iter_args via scf.yield, peeling iteration 0 must:
//   1. Substitute the iter_args init values for the iter_arg block args in
//      the cloned iter-0 body.
//   2. Capture the scf.yield results from the peeled iter-0 body and pass
//      them as the iter_args init of the trimmed loop.
//
// Without this, the trimmed loop would re-start from the original init,
// silently discarding iter 0's computation.

// RUN: aie-opt --aie-objectFifo-stateful-transform="dynamic-objFifos=true" %s | FileCheck %s

// CHECK: aie.core
// Iter 0 peeled body sees the user's original iter_args init (constant 7).
// It computes iter_arg + 1 and that becomes the trimmed loop's init.
// Use a constant the lowering won't fold away so we can grep for it.
// CHECK-DAG: %[[INIT:.+]] = arith.constant 7 : i32
// CHECK-DAG: %[[STEP:.+]] = arith.constant 1 : i32

// The peeled iter-0 body executes the user's add and uses the init value.
// (Order in IR is INIT, then peeled body, then trimmed loop.)
// CHECK: aie.use_lock(%{{.*}}_cons_cons_lock_0, AcquireGreaterEqual, 3)
// CHECK: %[[AFTER0:.+]] = arith.addi %[[INIT]], %[[STEP]] : i32
// CHECK: aie.use_lock(%{{.*}}_cons_prod_lock_0, Release, 1)

// The trimmed loop's iter_args init is the AFTER0 SSA value from the peel,
// NOT the original constant 7.
// CHECK: scf.for {{.*}} iter_args(%{{.*}} = %[[AFTER0]]) -> (i32)
// CHECK: aie.use_lock(%{{.*}}_cons_cons_lock_0, AcquireGreaterEqual, 1)
// CHECK: arith.addi
// CHECK: scf.yield

module {
  aie.device(npu2) {
    %tile_0_1 = aie.tile(0, 1)
    %tile_0_2 = aie.tile(0, 2)

    aie.objectfifo @fifo(%tile_0_1, {%tile_0_2}, 3 : i32) : !aie.objectfifo<memref<8xi8>>

    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c14 = arith.constant 14 : index
      %init = arith.constant 7 : i32
      %step_i32 = arith.constant 1 : i32
      %final = scf.for %arg0 = %c0 to %c14 step %c1 iter_args(%acc = %init) -> (i32) {
        %x = aie.objectfifo.acquire @fifo(Consume, 3) : !aie.objectfifosubview<memref<8xi8>>
        %new_acc = arith.addi %acc, %step_i32 : i32
        aie.objectfifo.release @fifo(Consume, 1)
        scf.yield %new_acc : i32
      }
      aie.objectfifo.release @fifo(Consume, 2)
      aie.end
    }
  }
}
