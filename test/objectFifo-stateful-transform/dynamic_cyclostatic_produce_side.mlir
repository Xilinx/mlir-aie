//===- dynamic_cyclostatic_produce_side.mlir ------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// Producer-side cyclostatic: a core acquires multiple producer slots and
// releases them one at a time per iteration. Same logic as consumer-side,
// just on the opposite port.

// RUN: aie-opt --aie-objectFifo-stateful-transform="dynamic-objFifos=true" %s | FileCheck %s

// CHECK: aie.core
// Peeled iter-0: producer-side prod_lock full acquire(3).
// CHECK: aie.use_lock(%{{.*}}_prod_lock_0, AcquireGreaterEqual, 3)
// Per-iter release into consumer.
// CHECK: aie.use_lock(%{{.*}}_cons_lock_0, Release, 1)
// Trimmed loop: per-iter delta of 1 (3 - carry=2).
// CHECK: scf.for
// CHECK: aie.use_lock(%{{.*}}_prod_lock_0, AcquireGreaterEqual, 1)
// CHECK: aie.use_lock(%{{.*}}_cons_lock_0, Release, 1)
// Trailing drain release(2).
// CHECK: aie.use_lock(%{{.*}}_cons_lock_0, Release, 2)

module {
  aie.device(npu2) {
    %tile_0_2 = aie.tile(0, 2)
    %tile_0_3 = aie.tile(0, 3)

    aie.objectfifo @fifo(%tile_0_2, {%tile_0_3}, 3 : i32) : !aie.objectfifo<memref<8xi8>>

    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c14 = arith.constant 14 : index
      scf.for %arg0 = %c0 to %c14 step %c1 {
        %x = aie.objectfifo.acquire @fifo(Produce, 3) : !aie.objectfifosubview<memref<8xi8>>
        aie.objectfifo.release @fifo(Produce, 1)
      }
      aie.objectfifo.release @fifo(Produce, 2)
      aie.end
    }
  }
}
