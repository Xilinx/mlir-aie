//===- dynamic_cyclostatic_pure_conditional.mlir --------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// Edge case: ALL acq/rel of (fifo, port) in the loop body are inside
// scf.if. Sibling test dynamic_cyclostatic_conditional_release.mlir covers
// the MIXED case (some conditional, some not), which triggers a diagnostic.
//
// Pure-conditional is well-formed: each conditional path is its own
// straight-line acq/rel sequence that the lock-lowering tracks correctly on
// its own. There is no cyclostatic carry to analyze (no unconditional ops
// to compute max_acq / sum_rel over), so peel must NOT fire and must NOT
// emit a diagnostic. The lowering proceeds as normal.

// RUN: aie-opt --aie-objectFifo-stateful-transform="dynamic-objFifos=true" %s | FileCheck %s

// CHECK: aie.core
// scf.for is preserved as-is (no peeled iter-0 cloned before it).
// CHECK: scf.for
// scf.if inside the loop body is preserved.
// CHECK: scf.if
// The user's conditional acq/rel lowered to a conditional use_lock.
// CHECK: aie.use_lock(%{{.*}}_cons_cons_lock_0, AcquireGreaterEqual, 3)
// CHECK: aie.use_lock(%{{.*}}_cons_prod_lock_0, Release, 1)

module {
  aie.device(npu2) {
    %tile_0_1 = aie.tile(0, 1)
    %tile_0_2 = aie.tile(0, 2)

    aie.objectfifo @fifo(%tile_0_1, {%tile_0_2}, 3 : i32) : !aie.objectfifo<memref<8xi8>>

    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c14 = arith.constant 14 : index
      %true = arith.constant true
      scf.for %arg0 = %c0 to %c14 step %c1 {
        scf.if %true {
          %x = aie.objectfifo.acquire @fifo(Consume, 3) : !aie.objectfifosubview<memref<8xi8>>
          aie.objectfifo.release @fifo(Consume, 1)
        }
      }
      aie.end
    }
  }
}
