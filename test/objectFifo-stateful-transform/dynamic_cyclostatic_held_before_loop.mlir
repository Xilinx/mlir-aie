//===- dynamic_cyclostatic_held_before_loop.mlir --------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// The peel decision subtracts the pre-loop "held" count from the in-body
// carry: if the user already holds N items before entering the loop, the
// in-body steady-state acquire is already a delta of (carry - N) and no
// peel is needed.
//
// Here a pre-loop acquire(2) leaves held=2 on entry; the loop body's
// acquire(3) - release(1) has carry 2; carry - held = 0 → no peel.
// (Without this analysis the peel would fire and produce an off-by-one
// duplicate acquire that desynchronizes the consumer-side lock.)

// RUN: aie-opt --aie-objectFifo-stateful-transform="dynamic-objFifos=true" %s | FileCheck %s

// CHECK: aie.core

// Pre-loop user acquire (full size from cold start).
// CHECK: aie.use_lock(%{{.*}}_cons_cons_lock_0, AcquireGreaterEqual, 2)

// No peeled iter-0 here — the next op must be the scf.for itself, not a
// second AcquireGreaterEqual.
// CHECK-NOT: aie.use_lock(%{{.*}}_cons_cons_lock_0, AcquireGreaterEqual
// CHECK: scf.for

// Inside the loop: in-body acquire emits as delta-from-held = 3 - 2 = 1.
// CHECK: aie.use_lock(%{{.*}}_cons_cons_lock_0, AcquireGreaterEqual, 1)
// CHECK: aie.use_lock(%{{.*}}_cons_prod_lock_0, Release, 1)

// Trailing drain.
// CHECK: aie.use_lock(%{{.*}}_cons_prod_lock_0, Release, 2)

module {
  aie.device(npu2) {
    %tile_0_1 = aie.tile(0, 1)
    %tile_0_2 = aie.tile(0, 2)

    aie.objectfifo @fifo(%tile_0_1, {%tile_0_2}, 3 : i32) : !aie.objectfifo<memref<8xi8>>

    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c14 = arith.constant 14 : index
      // Pre-loop acquire establishes held = 2.
      %pre = aie.objectfifo.acquire @fifo(Consume, 2) : !aie.objectfifosubview<memref<8xi8>>
      scf.for %arg0 = %c0 to %c14 step %c1 {
        %x = aie.objectfifo.acquire @fifo(Consume, 3) : !aie.objectfifosubview<memref<8xi8>>
        aie.objectfifo.release @fifo(Consume, 1)
      }
      aie.objectfifo.release @fifo(Consume, 2)
      aie.end
    }
  }
}
