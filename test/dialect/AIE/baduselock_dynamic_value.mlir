//===- baduselock_dynamic_value.mlir ---------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Verifier tests for lock values used inside DMA/BD blocks. Locks there are
// configured via MMIO, so the value must be a compile-time constant defined by
// an arith.constant.

// RUN: aie-opt -split-input-file -verify-diagnostics %s

// A non-constant lock value inside a DMA/BD block is rejected.
module {
  aie.device(npu1) {
    %tile = aie.tile(1, 2)
    %lock = aie.lock(%tile, 0)
    %mem = aie.mem(%tile) {
      %dma = aie.dma_start(MM2S, 0, ^bd, ^end)
    ^bd:
      %v = arith.constant 1 : i32
      %w = arith.addi %v, %v : i32
      // expected-error@+1 {{lock value in a DMA/BD block must be a compile-time constant (defined by an arith.constant)}}
      aie.use_lock(%lock, Acquire, %w)
      aie.next_bd ^end
    ^end:
      aie.end
    }
  }
}
