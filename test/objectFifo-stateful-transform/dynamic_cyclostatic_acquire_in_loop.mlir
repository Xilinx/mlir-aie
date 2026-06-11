//===- dynamic_cyclostatic_acquire_in_loop.mlir ---------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// Regression for https://github.com/Xilinx/mlir-aie/issues/2463.
//
// ObjectFifo `acquire(N)` means "ensure the core holds N total items"
// (see test/objectFifo-stateful-transform/access_patterns/AIE2_delayed_release.mlir).
// The emitted `aie.use_lock(..., AcquireGreaterEqual, K)` value K is the
// delta `N - currently_held`, not N itself.
//
// In the cyclostatic pattern below:
//   for i in 0..14:
//     acquire(3)   // hold 3
//     release(1)   // drop 1, still holding 2
//
// At loop-body entry:
//   - First iteration:  held = 0  => emit AcquireGreaterEqual(3).
//   - Iterations 1..13: held = 2  => emit AcquireGreaterEqual(1).
//
// The static-unroll path (`dynamic-objFifos=false`) does this correctly: it
// unrolls by LCM(consumer fifo sizes) and emits the per-position delta.
//
// The dynamic-objFifos path (default) keeps the scf.for and emits the user's
// literal value `AcquireGreaterEqual(3)` on every iteration. That over-
// acquires by 2 per iter, exhausting the producer pool after ~2 iterations
// and deadlocking on hardware.
//
// This test asserts the correct dynamic-path lowering. It will fail until the
// dynamic-objFifos lowering subtracts the steady-state held count (or peels
// the first iteration).

// RUN: aie-opt --aie-objectFifo-stateful-transform="dynamic-objFifos=true" %s | FileCheck %s

// The inner scf.for is preserved.
// CHECK: aie.core
// CHECK: scf.for
// CHECK: scf.for {{.*}} to {{.*}} step

// Inside the inner loop body, the steady-state acquire delta is 1, NOT 3:
// the previous iteration's release(1) left 2 items held, and acquire(3)
// only needs 1 more.
// CHECK-NEXT: aie.use_lock(%{{.*}}_cons_cons_lock_0, AcquireGreaterEqual, 1)

// The release(1) inside the body stays at 1.
// CHECK: aie.use_lock(%{{.*}}_cons_prod_lock_0, Release, 1)

// The trailing release(2) after the loop stays at 2.
// CHECK: aie.use_lock(%{{.*}}_cons_prod_lock_0, Release, 2)

module {
  aie.device(npu2) {
    %tile_0_1 = aie.tile(0, 1)
    %tile_0_2 = aie.tile(0, 2)

    aie.objectfifo @fifo(%tile_0_1, {%tile_0_2}, 3 : i32) : !aie.objectfifo<memref<8xi8>>

    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %cmax = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      %c14 = arith.constant 14 : index
      scf.for %arg0 = %c0 to %cmax step %c1 {
        scf.for %arg1 = %c0 to %c14 step %c1 {
          %x = aie.objectfifo.acquire @fifo(Consume, 3) : !aie.objectfifosubview<memref<8xi8>>
          aie.objectfifo.release @fifo(Consume, 1)
        }
        aie.objectfifo.release @fifo(Consume, 2)
      }
      aie.end
    }
  }
}
