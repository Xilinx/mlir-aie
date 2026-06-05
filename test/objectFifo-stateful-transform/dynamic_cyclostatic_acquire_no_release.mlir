//===- dynamic_cyclostatic_acquire_no_release.mlir ------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// Edge case: loop body has acquire(N) but NO release.
//
// `acquire(N)` semantics: "ensure I hold at least N items." If the loop body
// acquires N once and never releases, every iteration's acquire is a no-op
// after the first (held already == N). The peel correctly handles this:
// iter-0 takes the full acquire(N); the trimmed loop's acquires lower to
// `AcquireGreaterEqual(0)` deltas, which the lock lowering elides.
//
// Net effect: the lowered loop body contains NO use_lock op for this fifo,
// which is exactly what the source program means.

// RUN: aie-opt --aie-objectFifo-stateful-transform="dynamic-objFifos=true" %s | FileCheck %s

// CHECK: aie.core
// Peeled iter-0 has the user's full acquire(3).
// CHECK: aie.use_lock(%{{.*}}_cons_cons_lock_0, AcquireGreaterEqual, 3)
// Trimmed loop body has NO use_lock for the fifo's consumer side: the
// per-iter delta is 0 and the lock lowering elides the no-op.
// CHECK: scf.for
// CHECK-NOT: aie.use_lock
// CHECK: }

module {
  aie.device(npu2) {
    %tile_0_1 = aie.tile(0, 1)
    %tile_0_2 = aie.tile(0, 2)

    aie.objectfifo @fifo(%tile_0_1, {%tile_0_2}, 4 : i32) : !aie.objectfifo<memref<8xi8>>

    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c14 = arith.constant 14 : index
      scf.for %arg0 = %c0 to %c14 step %c1 {
        %x = aie.objectfifo.acquire @fifo(Consume, 3) : !aie.objectfifosubview<memref<8xi8>>
      }
      aie.end
    }
  }
}
