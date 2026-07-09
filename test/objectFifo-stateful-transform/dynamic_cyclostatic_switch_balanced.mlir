//===- dynamic_cyclostatic_switch_balanced.mlir --------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// The net-equal cyclostatic analysis treats scf.index_switch the same way as
// scf.if: every case region *and* the default region is a branch, and they
// are compared against each other. Here a (fifo, port) is used unconditionally
// and inside an scf.index_switch whose every branch (case 0, case 1, default)
// is balanced (acquire 1 / release 1, net 0). All branches agree, so the
// conditional contributes zero deterministic carry and the pass must NOT emit
// "cannot statically analyze cyclostatic acquire pattern"; the loop lowers
// normally with the switch preserved.

// RUN: aie-opt --aie-objectFifo-stateful-transform="dynamic-objFifos=true" %s | FileCheck %s

// CHECK: aie.core
// CHECK: scf.for
// CHECK: aie.use_lock(%{{.*}}_cons_cons_lock_0, AcquireGreaterEqual, 1)
// CHECK: aie.use_lock(%{{.*}}_cons_prod_lock_0, Release, 1)
// CHECK: scf.index_switch

module {
  aie.device(npu2) {
    %tile_0_1 = aie.tile(0, 1)
    %tile_0_2 = aie.tile(0, 2)

    aie.objectfifo @fifo(%tile_0_1, {%tile_0_2}, 4 : i32) : !aie.objectfifo<memref<8xi8>>

    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c14 = arith.constant 14 : index
      scf.for %arg0 = %c0 to %c14 step %c1 {
        // Unconditional, balanced: net 0.
        %u = aie.objectfifo.acquire @fifo(Consume, 1) : !aie.objectfifosubview<memref<8xi8>>
        aie.objectfifo.release @fifo(Consume, 1)
        // Every branch (including default) has the same net 0.
        scf.index_switch %arg0
        case 0 {
          %a = aie.objectfifo.acquire @fifo(Consume, 1) : !aie.objectfifosubview<memref<8xi8>>
          aie.objectfifo.release @fifo(Consume, 1)
          scf.yield
        }
        case 1 {
          %b = aie.objectfifo.acquire @fifo(Consume, 1) : !aie.objectfifosubview<memref<8xi8>>
          aie.objectfifo.release @fifo(Consume, 1)
          scf.yield
        }
        default {
          %d = aie.objectfifo.acquire @fifo(Consume, 1) : !aie.objectfifosubview<memref<8xi8>>
          aie.objectfifo.release @fifo(Consume, 1)
          scf.yield
        }
      }
      aie.end
    }
  }
}
