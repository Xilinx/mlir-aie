//===- dynamic_cyclostatic_both_outer_inner.mlir --------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// Nested loops with cyclostatic patterns on BOTH levels, different fifos:
//   outer carry on W: acquire(2) + release(1) per outer iter, drain at end.
//   inner carry on X: acquire(3) + release(1) per inner iter, drain at end.
//
// The fix must hoist a pre-acquire before each loop independently, processing
// innermost-first so that the inner pattern's hoisted acquire is not
// double-counted as part of the outer body's analysis.

// RUN: aie-opt --aie-objectFifo-stateful-transform="dynamic-objFifos=true" %s | FileCheck %s

// CHECK: aie.core
// Peeled outer iter-0 body: contains user's W acquire (full size 2), then
// peeled inner iter-0 (X AcqGE(3), Release X 1), then trimmed inner for
// (X AcqGE(1), Release X 1), then user's post-inner releases.
// CHECK: aie.use_lock(%inOF_W{{.*}}_cons_cons_lock_0, AcquireGreaterEqual, 2)
// Peeled inner iter-0 (inside peeled outer iter-0).
// CHECK: aie.use_lock(%inOF_X{{.*}}_cons_cons_lock_0, AcquireGreaterEqual, 3)
// CHECK: aie.use_lock(%inOF_X{{.*}}_cons_prod_lock_0, Release, 1)
// Trimmed inner for inside peeled outer iter-0: per-iter delta of 1.
// CHECK: scf.for
// CHECK: aie.use_lock(%inOF_X{{.*}}_cons_cons_lock_0, AcquireGreaterEqual, 1)
// CHECK: aie.use_lock(%inOF_X{{.*}}_cons_prod_lock_0, Release, 1)
// Post-inner X drain (release 2) and W release inside peeled outer iter-0.
// CHECK: aie.use_lock(%inOF_X{{.*}}_cons_prod_lock_0, Release, 2)
// CHECK: aie.use_lock(%inOF_W{{.*}}_cons_prod_lock_0, Release, 1)
// Trimmed outer for: per-iter W delta of 1.
// CHECK: scf.for
// CHECK: aie.use_lock(%inOF_W{{.*}}_cons_cons_lock_0, AcquireGreaterEqual, 1)
// Trailing drain release after the trimmed outer for.
// CHECK: aie.use_lock(%inOF_W{{.*}}_cons_prod_lock_0, Release, 1)

module {
  aie.device(npu2) {
    %tile_0_1 = aie.tile(0, 1)
    %tile_0_2 = aie.tile(0, 2)

    aie.objectfifo @inOF_W(%tile_0_1, {%tile_0_2}, 3 : i32) : !aie.objectfifo<memref<8xi8>>
    aie.objectfifo @inOF_X(%tile_0_1, {%tile_0_2}, 3 : i32) : !aie.objectfifo<memref<8xi8>>

    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c14 = arith.constant 14 : index
      scf.for %arg0 = %c0 to %c14 step %c1 {
        %w = aie.objectfifo.acquire @inOF_W(Consume, 2) : !aie.objectfifosubview<memref<8xi8>>
        scf.for %arg1 = %c0 to %c14 step %c1 {
          %x = aie.objectfifo.acquire @inOF_X(Consume, 3) : !aie.objectfifosubview<memref<8xi8>>
          aie.objectfifo.release @inOF_X(Consume, 1)
        }
        aie.objectfifo.release @inOF_X(Consume, 2)
        aie.objectfifo.release @inOF_W(Consume, 1)
      }
      aie.objectfifo.release @inOF_W(Consume, 1)
      aie.end
    }
  }
}
