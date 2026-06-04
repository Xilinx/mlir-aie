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
// Outer-loop pre-acquire of W (carry 2 - 1 = 1).
// CHECK: aie.use_lock(%inOF_W{{.*}}_cons_cons_lock_0, AcquireGreaterEqual, 1)
// CHECK: scf.for
// Inside outer body: steady-state acquire delta on W is 1.
// CHECK-NEXT: aie.use_lock(%inOF_W{{.*}}_cons_cons_lock_0, AcquireGreaterEqual, 1)
// Inner-loop pre-acquire of X (carry 3 - 1 = 2).
// CHECK: aie.use_lock(%inOF_X{{.*}}_cons_cons_lock_0, AcquireGreaterEqual, 2)
// CHECK: scf.for
// CHECK-NEXT: aie.use_lock(%inOF_X{{.*}}_cons_cons_lock_0, AcquireGreaterEqual, 1)
// Body release of X.
// CHECK: aie.use_lock(%inOF_X{{.*}}_cons_prod_lock_0, Release, 1)
// Drain inner X carry.
// CHECK: aie.use_lock(%inOF_X{{.*}}_cons_prod_lock_0, Release, 2)
// Body release of W.
// CHECK: aie.use_lock(%inOF_W{{.*}}_cons_prod_lock_0, Release, 1)
// Drain outer W carry.
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
