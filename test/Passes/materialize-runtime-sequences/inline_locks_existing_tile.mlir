//===- inline_locks_existing_tile.mlir --------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-materialize-runtime-sequences %s | FileCheck %s

// The caller declares the callee's tile itself, after other ops, as the control
// overlay's import does; the lock cloned in uses that tile, so it must follow it.

module {
  // CHECK-LABEL: aie.device(npu2) {
  // CHECK: %[[TILE:.*]] = aie.tile(0, 2)
  // CHECK-NEXT: aie.lock(%[[TILE]], 4)
  aie.device(npu2) {
    %shim = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)
    aie.runtime_sequence(%arg0: memref<64xi32>) {
      aiex.configure @callee {
        aiex.run @sequence(%arg0) : (memref<64xi32>)
      }
    }
  }

  aie.device(npu2) @callee {
    %tile_0_2 = aie.tile(0, 2)
    %lock = aie.lock(%tile_0_2, 4) {init = 0 : i32}
    aie.runtime_sequence (%arg0: memref<64xi32>) {
      %one = arith.constant 1 : i32
      aiex.set_lock(%lock, %one)
    }
  }
}
