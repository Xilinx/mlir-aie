//===- dynamic_cyclostatic_balanced_conditional.mlir ---------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Regression test for the multi-head-attention lowering (see commit that
// stabilized cyclostatic pattern analysis). A (fifo, port) is used BOTH
// unconditionally AND inside an scf.if, but every occurrence is *balanced*
// (acquire count == release count within its scope). A balanced conditional
// branch contributes zero net carry, so it cannot make the straight-line
// carry analysis unsound: the pass must NOT emit the "cannot statically
// analyze cyclostatic acquire pattern" diagnostic, must NOT peel, and must
// lower normally.
//
// Before the fix, the mere co-occurrence of conditional + unconditional
// acq/rel on the same fifo tripped a hard error even though the program is
// well-formed.

// RUN: aie-opt --aie-objectFifo-stateful-transform="dynamic-objFifos=true" --aie-assign-lock-ids %s | FileCheck %s

// CHECK: aie.core
// The loop is preserved as-is (no peeled iter-0 cloned before it).
// CHECK: scf.for
// The conditional acq/rel inside the loop body is preserved.
// CHECK: scf.if
// Both the unconditional and conditional acquires lower to a plain
// AcquireGreaterEqual(1) / Release(1) pair; no hoisted peel-acquire appears.
// CHECK: %{{.*}} = arith.constant 1 : i32
// CHECK: aie.use_lock(%{{.*}}_cons_cons_lock_0, AcquireGreaterEqual, %{{.*}})
// CHECK: %{{.*}} = arith.constant 1 : i32
// CHECK: aie.use_lock(%{{.*}}_cons_prod_lock_0, Release, %{{.*}})

module {
  aie.device(npu2) {
    %tile_0_1 = aie.tile(0, 1)
    %tile_0_2 = aie.tile(0, 2)

    aie.objectfifo @fifo(%tile_0_1, {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<8xi8>>

    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c14 = arith.constant 14 : index
      %true = arith.constant true
      scf.for %arg0 = %c0 to %c14 step %c1 {
        // Unconditional, balanced: acquire 1, release 1 -> net 0.
        %a = aie.objectfifo.acquire @fifo(Consume, 1) : !aie.objectfifosubview<memref<8xi8>>
        aie.objectfifo.release @fifo(Consume, 1)
        // Conditional, balanced: acquire 1, release 1 -> net 0 per branch.
        scf.if %true {
          %b = aie.objectfifo.acquire @fifo(Consume, 1) : !aie.objectfifosubview<memref<8xi8>>
          aie.objectfifo.release @fifo(Consume, 1)
        }
      }
      aie.end
    }
  }
}
