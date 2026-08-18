//===- lower_set_lock_in_loop.mlir -----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --aie-lower-set-lock %s | FileCheck %s

// aiex.set_lock lowers to a single write32, so it is valid anywhere a write is
// valid inside a runtime sequence -- including inside a rolled loop or a
// select arm, not only as a direct child of the sequence. A control program
// that keeps its loop rolled (a runtime-bound trip count, which is the point
// of the dynamic BD pool path) has to release a lock once per iteration.

// CHECK-LABEL: @lock_in_runtime_bound_loop
// CHECK:         scf.for
// 126976 = 0x0001F000, lock 0 in a compute tile's local address space.
// CHECK-DAG:       %[[V0:.*]] = arith.constant 1 : i32
// CHECK-DAG:       %[[A0:.*]] = arith.constant 126976 : i32
// CHECK:           aiex.npu.write32(%[[A0]], %[[V0]]) {column = 2 : i32, row = 2 : i32} : i32, i32
module @lock_in_runtime_bound_loop {
  aie.device(npu2) {
    %tile22 = aie.tile(2, 2)
    %lock22_0 = aie.lock(%tile22, 0) {init = 0 : i32}
    aie.runtime_sequence(%n: index) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      scf.for %i = %c0 to %n step %c1 {
        aiex.set_lock(%lock22_0, 1)
      }
    }
  }
}

// -----

// Two levels down: a select arm inside a loop, which is where a rolled
// per-iteration mode select puts its arming.

// CHECK-LABEL: @lock_in_select_arm
// CHECK:         scf.for
// CHECK:           scf.index_switch
// 786480 = 0x000C0030, lock 3 in a memtile's local address space.
// CHECK-DAG:         %[[V1:.*]] = arith.constant 1 : i32
// CHECK-DAG:         %[[A1:.*]] = arith.constant 786480 : i32
// CHECK:             aiex.npu.write32(%[[A1]], %[[V1]]) {column = 1 : i32, row = 1 : i32} : i32, i32
module @lock_in_select_arm {
  aie.device(npu2) {
    %memtile11 = aie.tile(1, 1)
    %lock11_3 = aie.lock(%memtile11, 3) {init = 0 : i32}
    aie.runtime_sequence(%n: index, %sel: index) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      scf.for %i = %c0 to %n step %c1 {
        scf.index_switch %sel
        case 0 {
          aiex.set_lock(%lock11_3, 1)
          scf.yield
        }
        default {
          scf.yield
        }
      }
    }
  }
}
