//===- dynamic_cyclostatic_scf_while_side_effects.mlir --------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// scf.while whose before-region contains a side-effecting op (memref.load).
// Cloning the before-region would execute the load twice, which can change
// program semantics if the loaded value is expected to vary between calls.
// The peel skips this loop with a warning; the lowering still produces
// correct (un-optimized) code.

// RUN: aie-opt --aie-objectFifo-stateful-transform="dynamic-objFifos=true" %s 2>&1 | FileCheck %s

// CHECK: warning: {{.*}}cyclostatic acquire peel skipped{{.*}}scf.while before-region has side effects
// CHECK: aie.core
// scf.while is preserved unchanged (no scf.if wrapper, no peeled iter-0).
// CHECK: scf.while
// In-body acquire stays at the user's literal value (no per-iter delta).
// CHECK: aie.use_lock(%{{.*}}_cons_cons_lock_0, AcquireGreaterEqual, 3)

module {
  aie.device(npu2) {
    %tile_0_1 = aie.tile(0, 1)
    %tile_0_2 = aie.tile(0, 2)
    %flag = aie.buffer(%tile_0_2) {sym_name = "flag"} : memref<1xi32>

    aie.objectfifo @fifo(%tile_0_1, {%tile_0_2}, 3 : i32) : !aie.objectfifo<memref<8xi8>>

    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c0_i32 = arith.constant 0 : i32
      scf.while : () -> () {
        // Side-effecting op in before-region.
        %v = memref.load %flag[%c0] : memref<1xi32>
        %cond = arith.cmpi ne, %v, %c0_i32 : i32
        scf.condition(%cond)
      } do {
        %x = aie.objectfifo.acquire @fifo(Consume, 3) : !aie.objectfifosubview<memref<8xi8>>
        aie.objectfifo.release @fifo(Consume, 1)
        scf.yield
      }
      aie.end
    }
  }
}
