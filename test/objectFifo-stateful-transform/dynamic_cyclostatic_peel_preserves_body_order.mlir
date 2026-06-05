//===- dynamic_cyclostatic_peel_preserves_body_order.mlir -----*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// A naive "hoist one acquire(carry) per fifo" rewrite groups the pre-loop
// acquires by fifo and reorders them relative to each other and to in-body
// ops on other fifos. That can deadlock cross-core when the relative order
// of acq/rel between fifos was load-bearing. Peel iter-0 must clone the
// body verbatim so the order is preserved.
//
// This test uses an interleaving that no per-fifo hoist could reproduce:
//     acquire fifoX(3)
//     acquire fifoY(2)
//     release fifoX(1)
//     acquire fifoX(3)     <-- second acquire of fifoX BETWEEN Y's acq/rel
//     release fifoY(1)
//     release fifoX(1)
// The peeled iter-0 must contain exactly this op sequence.

// RUN: aie-opt --aie-objectFifo-stateful-transform="dynamic-objFifos=true" %s | FileCheck %s

// CHECK: aie.core

// Peeled iter-0: exact source order. The lock-lowering tracks "currently
// held" and emits each AcquireGE as a delta from the prior held count, so
// the second X acquire (after one release) emits AcquireGE 1 rather than
// 3 — but it stays in source position between Y's acquire and release.
// CHECK:      aie.use_lock(%fifoX_cons_cons_lock_0, AcquireGreaterEqual, 3)
// CHECK:      aie.use_lock(%fifoY_cons_cons_lock_0, AcquireGreaterEqual, 2)
// CHECK:      aie.use_lock(%fifoX_cons_prod_lock_0, Release, 1)
// CHECK:      aie.use_lock(%fifoX_cons_cons_lock_0, AcquireGreaterEqual, 1)
// CHECK:      aie.use_lock(%fifoY_cons_prod_lock_0, Release, 1)
// CHECK:      aie.use_lock(%fifoX_cons_prod_lock_0, Release, 1)

// Trimmed loop: same interleaving, per-iter deltas.
// CHECK: scf.for
// CHECK:      aie.use_lock(%fifoX_cons_cons_lock_0, AcquireGreaterEqual, 1)
// CHECK:      aie.use_lock(%fifoY_cons_cons_lock_0, AcquireGreaterEqual, 1)
// CHECK:      aie.use_lock(%fifoX_cons_prod_lock_0, Release, 1)
// CHECK:      aie.use_lock(%fifoX_cons_cons_lock_0, AcquireGreaterEqual, 1)
// CHECK:      aie.use_lock(%fifoY_cons_prod_lock_0, Release, 1)
// CHECK:      aie.use_lock(%fifoX_cons_prod_lock_0, Release, 1)

// User trailing drains.
// CHECK: aie.use_lock(%fifoX_cons_prod_lock_0, Release, 2)
// CHECK: aie.use_lock(%fifoY_cons_prod_lock_0, Release, 1)

module {
  aie.device(npu2) {
    %tile_0_1 = aie.tile(0, 1)
    %tile_0_2 = aie.tile(0, 2)

    aie.objectfifo @fifoX(%tile_0_1, {%tile_0_2}, 3 : i32) : !aie.objectfifo<memref<8xi8>>
    aie.objectfifo @fifoY(%tile_0_1, {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<8xi8>>

    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c14 = arith.constant 14 : index
      scf.for %arg0 = %c0 to %c14 step %c1 {
        %x1 = aie.objectfifo.acquire @fifoX(Consume, 3) : !aie.objectfifosubview<memref<8xi8>>
        %y  = aie.objectfifo.acquire @fifoY(Consume, 2) : !aie.objectfifosubview<memref<8xi8>>
        aie.objectfifo.release @fifoX(Consume, 1)
        %x2 = aie.objectfifo.acquire @fifoX(Consume, 3) : !aie.objectfifosubview<memref<8xi8>>
        aie.objectfifo.release @fifoY(Consume, 1)
        aie.objectfifo.release @fifoX(Consume, 1)
      }
      aie.objectfifo.release @fifoX(Consume, 2)
      aie.objectfifo.release @fifoY(Consume, 1)
      aie.end
    }
  }
}
