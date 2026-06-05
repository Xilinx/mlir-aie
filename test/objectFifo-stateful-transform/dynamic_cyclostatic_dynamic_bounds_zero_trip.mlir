//===- dynamic_cyclostatic_dynamic_bounds_zero_trip.mlir ------*- MLIR -*-===//
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
// When the loop's trip count is not statically provable to be >= 1, the
// peeled iteration 0 must be wrapped in an `scf.if (ub - lb >= step)`
// guard. Otherwise a zero-trip loop would over-acquire and deadlock — a
// regression from today's broken-but-not-deadlocked behavior on zero-trip.
//
// The user's trailing `release(carry)` is left where they wrote it. If they
// wrote it unconditionally and the loop can run zero times, that is a
// pre-existing bug in user code, not introduced by peeling.
//
// E2E coverage gap: a runtime zero-trip test would need first-class
// runtime scalar args on @iron.jit to drive the trip count from the host
// (project_iron_jit_rtp deferred work). For now the IR shape is pinned by
// the scf.if + arith.cmpi check below.

// RUN: aie-opt --aie-objectFifo-stateful-transform="dynamic-objFifos=true" %s | FileCheck %s

// CHECK: aie.core
// The peeled iter-0 body must be guarded by an scf.if that fires only when
// the loop body executes at least once: (ub - lb) >= step.
// CHECK: arith.subi
// CHECK: arith.cmpi sge
// CHECK: scf.if
// Inside the guard: the peeled iter-0 acquire (full size, 3).
// CHECK: aie.use_lock(%{{.*}}_cons_cons_lock_0, AcquireGreaterEqual, 3)
// CHECK: aie.use_lock(%{{.*}}_cons_prod_lock_0, Release, 1)
// Trimmed loop: per-iter delta of 1 (3 - carry=2).
// CHECK: scf.for
// CHECK-NEXT: aie.use_lock(%{{.*}}_cons_cons_lock_0, AcquireGreaterEqual, 1)
// CHECK: aie.use_lock(%{{.*}}_cons_prod_lock_0, Release, 1)
// Trailing drain release(2) is user code, emitted unconditionally as
// written. (If the user's loop can be zero-trip, the trailing release is
// their bug; peel does not fix or hide it.)
// CHECK: aie.use_lock(%{{.*}}_cons_prod_lock_0, Release, 2)

module {
  aie.device(npu2) {
    %tile_0_1 = aie.tile(0, 1)
    %tile_0_2 = aie.tile(0, 2)

    aie.objectfifo @fifo(%tile_0_1, {%tile_0_2}, 3 : i32) : !aie.objectfifo<memref<8xi8>>
    %ub_buf = aie.buffer(%tile_0_2) {sym_name = "ub_buf"} : memref<1xindex>

    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      // Dynamic upper bound loaded from a buffer so the trip count is not
      // statically known to be >= 1.
      %ub = memref.load %ub_buf[%c0] : memref<1xindex>
      scf.for %arg0 = %c0 to %ub step %c1 {
        %x = aie.objectfifo.acquire @fifo(Consume, 3) : !aie.objectfifosubview<memref<8xi8>>
        aie.objectfifo.release @fifo(Consume, 1)
      }
      aie.objectfifo.release @fifo(Consume, 2)
      aie.end
    }
  }
}
