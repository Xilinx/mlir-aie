//===- dynamic_cyclostatic_multi_acquire_same_fifo.mlir -------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// Same fifo acquired multiple times in body (delayed-release pattern):
//   acquire(3)   // "ensure I hold 3"
//   acquire(5)   // "ensure I hold 5"   (incremental: +2)
//   release(1)
//
// Per AIE2_delayed_release.mlir semantics, the max-held count in the body is
// 5, and the per-iter release is 1, so carry = max_acq - sum_rel = 5 - 1 = 4.
// The fix must hoist `acquire(4)` before the loop. In-body acquires then
// become AcquireGreaterEqual(1) and AcquireGreaterEqual(2) — exactly the
// deltas to grow held from 4 -> 5, then refill to 5 after release.
//
// NOTE: This pattern requires fifo depth >= 5; use depth 5 here so the test
// is well-formed.

// RUN: aie-opt --aie-objectFifo-stateful-transform="dynamic-objFifos=true" %s | FileCheck %s

// CHECK: aie.core
// Hoisted pre-acquire of (max_acq - sum_rel) = 4.
// CHECK: aie.use_lock(%{{.*}}_cons_cons_lock_0, AcquireGreaterEqual, 4)
// CHECK: scf.for
// First in-body acquire: grow held from 4 -> max(3,5) = 5 - 4 = 1 if first
// in-body acquire is acq(3), then 5 - 3 = 2 (which itself becomes AcqGE 2)?
// The crisp invariant is: AcqGE values inside the loop must SUM to the per-
// iter release amount (1), so the loop is in steady state.
// CHECK-NEXT: aie.use_lock(%{{.*}}_cons_cons_lock_0, AcquireGreaterEqual,
// CHECK: aie.use_lock(%{{.*}}_cons_prod_lock_0, Release, 1)
// CHECK: aie.use_lock(%{{.*}}_cons_prod_lock_0, Release, 4)

module {
  aie.device(npu2) {
    %tile_0_1 = aie.tile(0, 1)
    %tile_0_2 = aie.tile(0, 2)

    aie.objectfifo @fifo(%tile_0_1, {%tile_0_2}, 5 : i32) : !aie.objectfifo<memref<8xi8>>

    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c14 = arith.constant 14 : index
      scf.for %arg0 = %c0 to %c14 step %c1 {
        %a = aie.objectfifo.acquire @fifo(Consume, 3) : !aie.objectfifosubview<memref<8xi8>>
        %b = aie.objectfifo.acquire @fifo(Consume, 5) : !aie.objectfifosubview<memref<8xi8>>
        aie.objectfifo.release @fifo(Consume, 1)
      }
      aie.objectfifo.release @fifo(Consume, 4)
      aie.end
    }
  }
}
