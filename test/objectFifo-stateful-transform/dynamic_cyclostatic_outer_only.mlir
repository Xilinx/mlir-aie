//===- dynamic_cyclostatic_outer_only.mlir --------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// Cyclostatic acquire(3)+release(1) on the OUTER loop. Inner loop has no
// fifo ops (just per-element work over the held window).
//
// Correct lowering should hoist `acquire(2)` before the outer loop so the
// outer body's `acquire(3)` lowers to `AcquireGreaterEqual(1)`. After the
// outer loop, the trailing release(2) drains the carry.

// RUN: aie-opt --aie-objectFifo-stateful-transform="dynamic-objFifos=true" %s | FileCheck %s

// CHECK: aie.core
// Peeled iter-0 body (full-size acquire) before the trimmed loop.
// CHECK: aie.use_lock(%{{.*}}_cons_cons_lock_0, AcquireGreaterEqual, 3)
// CHECK: aie.use_lock(%{{.*}}_cons_prod_lock_0, Release, 1)
// Trimmed loop: per-iter delta of 1.
// CHECK: scf.for
// CHECK: aie.use_lock(%{{.*}}_cons_cons_lock_0, AcquireGreaterEqual, 1)
// CHECK: aie.use_lock(%{{.*}}_cons_prod_lock_0, Release, 1)
// Trailing drain release(2) after the trimmed loop.
// CHECK: aie.use_lock(%{{.*}}_cons_prod_lock_0, Release, 2)

module {
  aie.device(npu2) {
    %tile_0_1 = aie.tile(0, 1)
    %tile_0_2 = aie.tile(0, 2)
    %buf = aie.buffer(%tile_0_2) {sym_name = "buf"} : memref<8xi8>

    aie.objectfifo @fifo(%tile_0_1, {%tile_0_2}, 3 : i32) : !aie.objectfifo<memref<8xi8>>

    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      %c14 = arith.constant 14 : index
      scf.for %arg0 = %c0 to %c14 step %c1 {
        %x = aie.objectfifo.acquire @fifo(Consume, 3) : !aie.objectfifosubview<memref<8xi8>>
        %x0 = aie.objectfifo.subview.access %x[0] : !aie.objectfifosubview<memref<8xi8>> -> memref<8xi8>
        // Inner loop does per-byte work but no fifo ops.
        scf.for %arg1 = %c0 to %c8 step %c1 {
          %v = memref.load %x0[%arg1] : memref<8xi8>
          memref.store %v, %buf[%arg1] : memref<8xi8>
        }
        aie.objectfifo.release @fifo(Consume, 1)
      }
      aie.objectfifo.release @fifo(Consume, 2)
      aie.end
    }
  }
}
