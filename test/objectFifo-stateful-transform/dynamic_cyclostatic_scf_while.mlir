//===- dynamic_cyclostatic_scf_while.mlir ---------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// Same cyclostatic pattern as dynamic_cyclostatic_acquire_in_loop.mlir, but
// the surrounding loop is scf.while instead of scf.for. The peel:
//   1. clones the (side-effect-free) before-region inline to compute the
//      iter-0 cond and forwarded values;
//   2. wraps the iter-0 after-region clone + a fresh trimmed scf.while in
//      an scf.if guarded by the iter-0 cond;
//   3. else-branch yields iter0Vals (what the original while would have
//      returned if the cond was false on entry);
//   4. rewires external uses of the original whileOp's results to the
//      ifOp's results, then erases the original whileOp.
//
// The test exercises (3) and (4) by consuming the while's result after the
// loop (`memref.store %r, %buf`) — that consumer must end up reading the
// ifOp's result, not a dangling reference to the deleted whileOp.

// RUN: aie-opt --aie-objectFifo-stateful-transform="dynamic-objFifos=true" %s | FileCheck %s

// CHECK: aie.core
// Iter-0 cond cloned at top, then scf.if guards the peel + trimmed while.
// CHECK: %[[ITER0_COND:.*]] = arith.cmpi
// CHECK: %[[IFRES:.*]] = scf.if %[[ITER0_COND]]
// Peeled iter-0: full-size acquire and user release inside the then branch.
// CHECK: aie.use_lock(%{{.*}}_cons_cons_lock_0, AcquireGreaterEqual, 3)
// CHECK: aie.use_lock(%{{.*}}_cons_prod_lock_0, Release, 1)
// Trimmed scf.while inside the then branch, steady-state delta of 1.
// CHECK: scf.while
// CHECK: aie.use_lock(%{{.*}}_cons_cons_lock_0, AcquireGreaterEqual, 1)
// CHECK: aie.use_lock(%{{.*}}_cons_prod_lock_0, Release, 1)
// Else branch yields the iter-0 forwarded values (the original init).
// CHECK: else
// CHECK: scf.yield %c0
// External consumer of the original while's result reads from the ifOp.
// CHECK: memref.store %[[IFRES]]
// CHECK: aie.use_lock(%{{.*}}_cons_prod_lock_0, Release, 2)

module {
  aie.device(npu2) {
    %tile_0_1 = aie.tile(0, 1)
    %tile_0_2 = aie.tile(0, 2)
    %buf = aie.buffer(%tile_0_2) {sym_name = "buf"} : memref<1xindex>

    aie.objectfifo @fifo(%tile_0_1, {%tile_0_2}, 3 : i32) : !aie.objectfifo<memref<8xi8>>

    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c14 = arith.constant 14 : index
      %r = scf.while (%arg0 = %c0) : (index) -> index {
        %cond = arith.cmpi slt, %arg0, %c14 : index
        scf.condition(%cond) %arg0 : index
      } do {
      ^bb0(%arg1: index):
        %x = aie.objectfifo.acquire @fifo(Consume, 3) : !aie.objectfifosubview<memref<8xi8>>
        aie.objectfifo.release @fifo(Consume, 1)
        %next = arith.addi %arg1, %c1 : index
        scf.yield %next : index
      }
      // External consumer of the while result — must be rewired to the
      // peel's outer scf.if result (not left dangling against the erased
      // original whileOp).
      memref.store %r, %buf[%c0] : memref<1xindex>
      aie.objectfifo.release @fifo(Consume, 2)
      aie.end
    }
  }
}
