//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.

// RUN: aie-opt --aie-test-runtime-bd-liveness --verify-diagnostics --split-input-file %s

// A task configured inside a loop whose handle is never synced (await/free)
// accumulates one held BD per iteration. This is a bug for a constant trip
// count and impossible (unbounded BDs) for a runtime one, so it is rejected.
// Top-level leaks are NOT rejected (see liveness-rotation-depth.mlir): only
// loop-carried leaks error.

// Constant trip count, no sync inside the loop, handle escapes via loop result.
aie.device(npu2) {
  %tile_0_0 = aie.tile(0, 0)
  // expected-remark@+1 {{bd-peak: tile(0,0) peak=2}}
  aie.runtime_sequence(%arg0: memref<8xi16>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    // Dead prologue: %cur is never used in the body, so %zero is never freed.
    // Top-level leak => legal (remark only), not an error.
    // expected-remark@+1 {{bd-liveness: leaked}}
    %zero = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<8xi16>, 0, 8)
      aie.end
    }
    %last = scf.for %i = %c0 to %c4 step %c1 iter_args(%cur = %zero) -> (index) {
      // expected-error@+1 {{buffer descriptor configured in a loop is never completed}}
      %t = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%arg0 : memref<8xi16>, 0, 8)
        aie.end
      }
      aiex.dma_start_task(%t)
      scf.yield %t : index
    }
    aiex.dma_free_task(%last)
  }
}

// -----

// Simplest loop leak: configure + start in the loop body, no sync at all.
aie.device(npu2) {
  %tile_0_0 = aie.tile(0, 0)
  // expected-remark@+1 {{bd-peak: tile(0,0) peak=1}}
  aie.runtime_sequence(%arg0: memref<8xi16>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    scf.for %i = %c0 to %c4 step %c1 {
      // expected-error@+1 {{buffer descriptor configured in a loop is never completed}}
      %t = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%arg0 : memref<8xi16>, 0, 8)
        aie.end
      }
      aiex.dma_start_task(%t)
    }
  }
}
