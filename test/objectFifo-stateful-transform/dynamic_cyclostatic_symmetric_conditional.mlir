//===- dynamic_cyclostatic_symmetric_conditional.mlir --------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// A (fifo, port) is used unconditionally AND inside an scf.if whose *both*
// branches have the SAME non-zero net effect (each acquires 2 and releases 1,
// i.e. net +1). Because every branch contributes the same delta, that delta
// is branch-independent (deterministic) and folds into the loop's static
// cyclostatic carry exactly like an unconditional net +1. The pass must
// therefore NOT emit "cannot statically analyze cyclostatic acquire pattern";
// instead it must peel iteration 0 so the steady-state AcquireGreaterEqual
// value becomes the per-iteration delta.
//
// This exercises the net-equal folding: only a conditional whose branches
// *disagree* is unanalyzable; matching (even non-zero) branch nets are fine.

// RUN: aie-opt --aie-objectFifo-stateful-transform="dynamic-objFifos=true" %s | FileCheck %s

// The peeled iteration 0 (including its clone of the conditional) is emitted
// *before* the trimmed loop, so an scf.if appears ahead of the scf.for:
// CHECK: aie.core
// CHECK: scf.if
// CHECK: aie.use_lock(%{{.*}}_cons_cons_lock_0, AcquireGreaterEqual, 2)
// CHECK: scf.for
// The steady-state body retains the unconditional acquire and the conditional:
// CHECK: aie.use_lock(%{{.*}}_cons_cons_lock_0, AcquireGreaterEqual, 1)
// CHECK: scf.if
// CHECK: aie.use_lock(%{{.*}}_cons_cons_lock_0, AcquireGreaterEqual, 2)

module {
  aie.device(npu2) {
    %tile_0_1 = aie.tile(0, 1)
    %tile_0_2 = aie.tile(0, 2)

    aie.objectfifo @fifo(%tile_0_1, {%tile_0_2}, 4 : i32) : !aie.objectfifo<memref<8xi8>>

    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c14 = arith.constant 14 : index
      %true = arith.constant true
      scf.for %arg0 = %c0 to %c14 step %c1 {
        // Unconditional, balanced: net 0.
        %u = aie.objectfifo.acquire @fifo(Consume, 1) : !aie.objectfifosubview<memref<8xi8>>
        aie.objectfifo.release @fifo(Consume, 1)
        // Conditional: both branches share the same non-zero net (+1).
        scf.if %true {
          %a = aie.objectfifo.acquire @fifo(Consume, 2) : !aie.objectfifosubview<memref<8xi8>>
          aie.objectfifo.release @fifo(Consume, 1)
        } else {
          %b = aie.objectfifo.acquire @fifo(Consume, 2) : !aie.objectfifosubview<memref<8xi8>>
          aie.objectfifo.release @fifo(Consume, 1)
        }
      }
      aie.end
    }
  }
}
