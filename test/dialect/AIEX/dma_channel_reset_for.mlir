//===- dma_channel_reset_for.mlir -------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file %s | FileCheck %s

// aiex.dma_channel_reset_for names an objectFIFO by symbol. Its verifier is
// deferring (mirrors dma_configure_task_for): before the objectFIFO transform
// the symbol resolves to an aie.objectfifo; after it, to the fifo's
// aie.objectfifo_rearm_binding. Both resolve clean.

// CHECK-LABEL: aie.device(npu2)
// CHECK: aiex.dma_channel_reset_for(@of)
module {
  aie.device(npu2) {
    %t0 = aie.tile(0, 0)
    %t2 = aie.tile(0, 2)
    aie.objectfifo @of (%t0, {%t2}, 2 : i32) : !aie.objectfifo<memref<64xi16>>
    aie.runtime_sequence() {
      aiex.dma_channel_reset_for(@of)
    }
  }
}

// -----

// Post-transform shape: the symbol resolves to the re-arm binding.
// CHECK-LABEL: aie.device(npu2)
// CHECK: aiex.dma_channel_reset_for(@of_rearm)
module {
  aie.device(npu2) {
    %ct = aie.tile(0, 3)
    %pl = aie.lock(%ct, 0) {init = 2 : i32}
    aie.objectfifo_rearm_binding @of_rearm channels(%ct : index) locks(%pl : index) {channel_dirs = array<i32: 0>, channel_indices = array<i32: 0>, lock_inits = array<i32: 2>}
    aie.runtime_sequence() {
      aiex.dma_channel_reset_for(@of_rearm)
    }
  }
}

// -----

// An as-yet-unresolvable symbol is deferred (a later pass resolves it), so the
// verifier does not reject it here.
// CHECK-LABEL: aie.device(npu2)
// CHECK: aiex.dma_channel_reset_for(@not_yet_lowered)
module {
  aie.device(npu2) {
    aie.runtime_sequence() {
      aiex.dma_channel_reset_for(@not_yet_lowered)
    }
  }
}
