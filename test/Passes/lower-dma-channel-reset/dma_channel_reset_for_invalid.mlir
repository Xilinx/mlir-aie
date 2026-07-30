//===- dma_channel_reset_for_invalid.mlir -----------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt -split-input-file -verify-diagnostics --aie-lower-dma-channel-reset %s

// By this stage the objectFIFO transform must already have emitted the re-arm
// binding. A dma_channel_reset_for whose symbol does not resolve to one is an
// error (no later pass will resolve it).
module {
  aie.device(npu2) {
    aie.runtime_sequence() {
      // expected-error @+1 {{could not resolve 'missing' to an aie.objectfifo_rearm_binding}}
      aiex.dma_channel_reset_for(@missing)
    }
  }
}

// -----

// A binding whose head_bd_ids / repeat_counts were never populated (this pass
// must run after --aie-assign-bd-ids, which folds them onto the binding) cannot
// re-push its channels: fail with an actionable diagnostic rather than emit a
// bogus START_QUEUE write.
module {
  aie.device(npu2) {
    %t03 = aie.tile(0, 3)
    %pl = aie.lock(%t03, 0) {init = 1 : i32}
    aie.objectfifo_rearm_binding @of_rearm channels(%t03 : index) locks(%pl : index) {channel_dirs = array<i32: 0>, channel_indices = array<i32: 0>, lock_inits = array<i32: 1>}
    aie.runtime_sequence() {
      // expected-error @+1 {{has no head_bd_ids/repeat_counts; run --aie-assign-bd-ids before this pass, or set them on the binding}}
      aiex.dma_channel_reset_for(@of_rearm)
    }
  }
}
