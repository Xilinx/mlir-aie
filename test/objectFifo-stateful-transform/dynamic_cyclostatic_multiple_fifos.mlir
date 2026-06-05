//===- dynamic_cyclostatic_multiple_fifos.mlir ----------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// Two fifos in the same inner loop body, each cyclostatic with a different
// carry: X acquires 3 / releases 1 (carry 2), Y acquires 2 / releases 1
// (carry 1). The fix must hoist independent pre-acquires for each fifo.

// RUN: aie-opt --aie-objectFifo-stateful-transform="dynamic-objFifos=true" %s | FileCheck %s

// CHECK: aie.core
// Peeled iter-0 body: full-size acquires in source order (X before Y), then
// releases in source order.
// CHECK: aie.use_lock(%fifoX_cons_cons_lock_0, AcquireGreaterEqual, 3)
// CHECK: aie.use_lock(%fifoY_cons_cons_lock_0, AcquireGreaterEqual, 2)
// CHECK: aie.use_lock(%fifoX_cons_prod_lock_0, Release, 1)
// CHECK: aie.use_lock(%fifoY_cons_prod_lock_0, Release, 1)
// Trimmed loop: per-iter deltas (X: 3-2=1; Y: 2-1=1) in source order.
// CHECK: scf.for
// CHECK: aie.use_lock(%fifoX_cons_cons_lock_0, AcquireGreaterEqual, 1)
// CHECK: aie.use_lock(%fifoY_cons_cons_lock_0, AcquireGreaterEqual, 1)
// CHECK: aie.use_lock(%fifoX_cons_prod_lock_0, Release, 1)
// CHECK: aie.use_lock(%fifoY_cons_prod_lock_0, Release, 1)
// Trailing user releases stay as written.
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
        %x = aie.objectfifo.acquire @fifoX(Consume, 3) : !aie.objectfifosubview<memref<8xi8>>
        %y = aie.objectfifo.acquire @fifoY(Consume, 2) : !aie.objectfifosubview<memref<8xi8>>
        aie.objectfifo.release @fifoX(Consume, 1)
        aie.objectfifo.release @fifoY(Consume, 1)
      }
      aie.objectfifo.release @fifoX(Consume, 2)
      aie.objectfifo.release @fifoY(Consume, 1)
      aie.end
    }
  }
}
