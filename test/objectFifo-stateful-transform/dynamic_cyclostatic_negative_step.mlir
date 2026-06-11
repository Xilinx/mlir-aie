//===- dynamic_cyclostatic_negative_step.mlir -----------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// scf.for with non-constant step (sign unknown at compile time): the peel
// rewrite must bail out and leave the loop intact, because emitting a
// positive-step runtime guard against a negative-step loop would skip the
// trimmed loop body entirely after rewrite.
//
// The lock-lowering still produces correct (just unoptimized) IR for the
// in-body acquire: AcquireGreaterEqual emits at the user's literal value.

// RUN: aie-opt --aie-objectFifo-stateful-transform="dynamic-objFifos=true" %s | FileCheck %s

// CHECK: aie.core
// No pre-loop peeled iter-0 — peel bailed out on unknown step sign.
// CHECK-NOT: aie.use_lock(%{{.*}}_cons_cons_lock_0, AcquireGreaterEqual, 3)
// The scf.for is preserved as-is, no scf.if guard wrapping a peeled body.
// CHECK: scf.for
// In-body acquire emits at the user's literal size (unoptimized but correct).
// CHECK: aie.use_lock(%{{.*}}_cons_cons_lock_0, AcquireGreaterEqual, 3)
// CHECK: aie.use_lock(%{{.*}}_cons_prod_lock_0, Release, 1)
// Trailing user drain.
// CHECK: aie.use_lock(%{{.*}}_cons_prod_lock_0, Release, 2)

module {
  aie.device(npu2) {
    %tile_0_1 = aie.tile(0, 1)
    %tile_0_2 = aie.tile(0, 2)

    aie.objectfifo @fifo(%tile_0_1, {%tile_0_2}, 3 : i32) : !aie.objectfifo<memref<8xi8>>
    %step_buf = aie.buffer(%tile_0_2) {sym_name = "step_buf"} : memref<1xindex>

    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c14 = arith.constant 14 : index
      // Step is loaded at runtime → sign cannot be proven at compile time,
      // so peel must bail out.
      %step = memref.load %step_buf[%c0] : memref<1xindex>
      scf.for %arg0 = %c0 to %c14 step %step {
        %x = aie.objectfifo.acquire @fifo(Consume, 3) : !aie.objectfifosubview<memref<8xi8>>
        aie.objectfifo.release @fifo(Consume, 1)
      }
      aie.objectfifo.release @fifo(Consume, 2)
      aie.end
    }
  }
}
