//===- bad_memtile_bd_pool_runtime.mlir ------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// An AIE2 memtile channel can only start on a BD of its parity pool: even
// channels use BDs 0-23, odd channels BDs 24-47. A queue push or a re-arm
// binding head BD outside the pool is rejected.

// RUN: aie-opt --split-input-file --verify-diagnostics %s

aie.device(npu1) {
  aie.runtime_sequence(%arg0: memref<16xi32>) {
    %c30 = arith.constant 30 : i32
    %c0 = arith.constant 0 : i32
    // expected-error@+1 {{'aiex.npu.push_queue' op BD ID 30 is not accessible from channel (S2MM, 0)}}
    aiex.npu.push_queue(0, 1, S2MM : 0) bd_id %c30 repeat %c0 {issue_token = false} : i32, i32
  }
}

// -----

aie.device(npu2) {
  aie.runtime_sequence(%arg0: memref<16xi32>) {
    %c5 = arith.constant 5 : i32
    %c0 = arith.constant 0 : i32
    // expected-error@+1 {{'aiex.npu.push_queue' op BD ID 5 is not accessible from channel (MM2S, 3)}}
    aiex.npu.push_queue(0, 1, MM2S : 3) bd_id %c5 repeat %c0 {issue_token = false} : i32, i32
  }
}

// -----

// Pushes in the right pool, and any BD on a core or shim tile, are accepted.
aie.device(npu1) {
  aie.runtime_sequence(%arg0: memref<16xi32>) {
    %c0 = arith.constant 0 : i32
    %c23 = arith.constant 23 : i32
    %c24 = arith.constant 24 : i32
    %c15 = arith.constant 15 : i32
    aiex.npu.push_queue(0, 1, S2MM : 0) bd_id %c23 repeat %c0 {issue_token = false} : i32, i32
    aiex.npu.push_queue(0, 1, MM2S : 5) bd_id %c24 repeat %c0 {issue_token = false} : i32, i32
    aiex.npu.push_queue(0, 2, S2MM : 1) bd_id %c15 repeat %c0 {issue_token = false} : i32, i32
    aiex.npu.push_queue(0, 0, MM2S : 1) bd_id %c15 repeat %c0 {issue_token = false} : i32, i32
  }
}

// -----

aie.device(npu2) {
  %mt = aie.tile(0, 1)
  %pl = aie.lock(%mt, 0) {init = 2 : i32}
  %cl = aie.lock(%mt, 1) {init = 0 : i32}
  // expected-error@+1 {{'aie.objectfifo_rearm_binding' op head BD 24 is not accessible from channel 0 of tile (0, 1)}}
  aie.objectfifo_rearm_binding @r channels(%mt : index) locks(%pl, %cl : index, index) {channel_dirs = array<i32: 0>, channel_indices = array<i32: 0>, lock_inits = array<i32: 2, 0>, head_bd_ids = array<i32: 24>, repeat_counts = array<i32: 0>}
}
