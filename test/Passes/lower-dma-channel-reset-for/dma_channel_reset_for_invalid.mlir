//===- dma_channel_reset_for_invalid.mlir -----------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt -split-input-file -verify-diagnostics --aie-lower-dma-channel-reset-for %s

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

// A binding endpoint whose resident DMA channel is absent cannot be re-pushed:
// fail rather than emit a bogus START_QUEUE write.
module {
  aie.device(npu2) {
    %t03 = aie.tile(0, 3)
    %pl = aie.lock(%t03, 0) {init = 1 : i32}
    aie.objectfifo_rearm_binding @of_rearm channels(%t03 : index) locks(%pl : index) {channel_dirs = array<i32: 0>, channel_indices = array<i32: 0>, lock_inits = array<i32: 1>}
    aie.runtime_sequence() {
      // expected-error @+1 {{could not find a resident DMA channel for endpoint on tile (0, 3)}}
      aiex.dma_channel_reset_for(@of_rearm)
    }
  }
}

// -----

// A resident channel whose head BD id has not been assigned yet (this pass must
// run after --aie-assign-bd-ids) gets a distinct, actionable diagnostic.
module {
  aie.device(npu2) {
    %t03 = aie.tile(0, 3)
    %buf = aie.buffer(%t03) : memref<64xi32>
    %pl = aie.lock(%t03, 0) {init = 1 : i32}
    %mem = aie.mem(%t03) {
      %s = aie.dma_start(S2MM, 0, ^bd, ^end)
    ^bd:
      %c1 = arith.constant 1 : i32
      aie.dma_bd(%buf : memref<64xi32> offset = 0 len = 64)
      aie.next_bd ^bd
    ^end:
      aie.end
    }
    aie.objectfifo_rearm_binding @of_rearm channels(%t03 : index) locks(%pl : index) {channel_dirs = array<i32: 0>, channel_indices = array<i32: 0>, lock_inits = array<i32: 1>}
    aie.runtime_sequence() {
      // expected-error @+1 {{has no assigned BD id; run --aie-assign-bd-ids first}}
      aiex.dma_channel_reset_for(@of_rearm)
    }
  }
}

// -----

// A resident channel whose repeat count exceeds the 8-bit START_QUEUE repeat
// field is rejected rather than truncated into a wrong re-push (aie.dma_start
// does not bound repeat_count; aie.dma_bd already bounds the head BD id).
module {
  aie.device(npu2) {
    %t03 = aie.tile(0, 3)
    %buf = aie.buffer(%t03) : memref<64xi32>
    %pl = aie.lock(%t03, 0) {init = 1 : i32}
    %mem = aie.mem(%t03) {
      %s = aie.dma_start(S2MM, 0, ^bd, ^end, repeat_count = 300)
    ^bd:
      aie.dma_bd(%buf : memref<64xi32> offset = 0 len = 64) {bd_id = 2 : i32, next_bd_id = 2 : i32}
      aie.next_bd ^bd
    ^end:
      aie.end
    }
    aie.objectfifo_rearm_binding @of_rearm channels(%t03 : index) locks(%pl : index) {channel_dirs = array<i32: 0>, channel_indices = array<i32: 0>, lock_inits = array<i32: 1>}
    aie.runtime_sequence() {
      // expected-error @+1 {{repeat count 300 on tile (0, 3) does not fit the 8-bit START_QUEUE repeat field}}
      aiex.dma_channel_reset_for(@of_rearm)
    }
  }
}
