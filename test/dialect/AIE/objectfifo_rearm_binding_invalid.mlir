//===- objectfifo_rearm_binding_invalid.mlir -------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt -split-input-file -verify-diagnostics %s

// The parallel arrays must match the operand counts: one channel_dirs entry per
// channel tile.
module {
  aie.device(npu2) {
    %ct = aie.tile(0, 3)
    // expected-error @+1 {{expected one channel_dirs entry per channel tile}}
    aie.objectfifo_rearm_binding @b channels(%ct : index) locks() {channel_dirs = array<i32: 0, 1>, channel_indices = array<i32: 0>, lock_inits = array<i32>}
  }
}

// -----

// One channel_indices entry per channel tile.
module {
  aie.device(npu2) {
    %ct = aie.tile(0, 3)
    // expected-error @+1 {{expected one channel_indices entry per channel tile}}
    aie.objectfifo_rearm_binding @b channels(%ct : index) locks() {channel_dirs = array<i32: 0>, channel_indices = array<i32: 0, 1>, lock_inits = array<i32>}
  }
}

// -----

// One lock_inits entry per lock.
module {
  aie.device(npu2) {
    %ct = aie.tile(0, 3)
    %l = aie.lock(%ct, 0) {init = 2 : i32}
    // expected-error @+1 {{expected one lock_inits entry per lock}}
    aie.objectfifo_rearm_binding @b channels() locks(%l : index) {channel_dirs = array<i32>, channel_indices = array<i32>, lock_inits = array<i32: 2, 0>}
  }
}

// -----

// channel_dirs entries must be 0 (S2MM) or 1 (MM2S).
module {
  aie.device(npu2) {
    %ct = aie.tile(0, 3)
    // expected-error @+1 {{channel_dirs entries must be 0 (S2MM) or 1 (MM2S)}}
    aie.objectfifo_rearm_binding @b channels(%ct : index) locks() {channel_dirs = array<i32: 2>, channel_indices = array<i32: 0>, lock_inits = array<i32>}
  }
}

// -----

// channel_tiles operands must be aie.tile values (the lowering resolves them to
// a TileOp); an arbitrary index is rejected rather than crashing the lowering.
module {
  aie.device(npu2) {
    %c = arith.constant 0 : index
    // expected-error @+1 {{channel_tiles operands must be aie.tile values}}
    aie.objectfifo_rearm_binding @b channels(%c : index) locks() {channel_dirs = array<i32: 0>, channel_indices = array<i32: 0>, lock_inits = array<i32>}
  }
}

// -----

// locks operands must be aie.lock values (the lowering emits set_lock on them).
module {
  aie.device(npu2) {
    %c = arith.constant 7 : index
    // expected-error @+1 {{locks operands must be aie.lock values}}
    aie.objectfifo_rearm_binding @b channels() locks(%c : index) {channel_dirs = array<i32>, channel_indices = array<i32>, lock_inits = array<i32: 1>}
  }
}

// -----

// channel_tiles must be non-shim: a shim DMA endpoint is host-managed, so a
// binding (even hand-authored) naming a shim tile is rejected rather than
// re-arming the host's channel.
module {
  aie.device(npu2) {
    %shim = aie.tile(0, 0)
    // expected-error @+1 {{channel_tiles must be non-shim tiles}}
    aie.objectfifo_rearm_binding @b channels(%shim : index) locks() {channel_dirs = array<i32: 0>, channel_indices = array<i32: 0>, lock_inits = array<i32>}
  }
}

// -----

// head_bd_ids and repeat_counts travel as a pair: the re-push needs both, so one
// without the other is rejected.
module {
  aie.device(npu2) {
    %ct = aie.tile(0, 3)
    // expected-error @+1 {{head_bd_ids and repeat_counts must both be set or both be absent}}
    aie.objectfifo_rearm_binding @b channels(%ct : index) locks() {channel_dirs = array<i32: 0>, channel_indices = array<i32: 0>, lock_inits = array<i32>, head_bd_ids = array<i32: 5>}
  }
}

// -----

// One head_bd_ids entry per channel tile.
module {
  aie.device(npu2) {
    %ct = aie.tile(0, 3)
    // expected-error @+1 {{expected one head_bd_ids entry per channel tile}}
    aie.objectfifo_rearm_binding @b channels(%ct : index) locks() {channel_dirs = array<i32: 0>, channel_indices = array<i32: 0>, lock_inits = array<i32>, head_bd_ids = array<i32: 5, 6>, repeat_counts = array<i32: 0, 0>}
  }
}

// -----

// One repeat_counts entry per channel tile.
module {
  aie.device(npu2) {
    %ct = aie.tile(0, 3)
    // expected-error @+1 {{expected one repeat_counts entry per channel tile}}
    aie.objectfifo_rearm_binding @b channels(%ct : index) locks() {channel_dirs = array<i32: 0>, channel_indices = array<i32: 0>, lock_inits = array<i32>, head_bd_ids = array<i32: 5>, repeat_counts = array<i32: 0, 1>}
  }
}
