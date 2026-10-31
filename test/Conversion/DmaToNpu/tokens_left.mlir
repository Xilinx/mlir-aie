//===- tokens_left.mlir -----------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-dma-to-npu --verify-diagnostics --split-input-file %s

// A task-completion token no await consumes outlives its runtime sequence,
// and the next command's await on that channel takes it: measured on Strix,
// that await returned when the earlier transfer landed, with its own output
// unwritten. A sequence may not end with a token outstanding on a channel any
// sequence of the device awaits; on a channel nothing awaits, a token is
// never taken. An await consumes the oldest token, so only counts matter.

// Two token-issuing pushes, one sync.
aie.device(npu2) {
  // expected-error@+1 {{ends with 1 task-completion token no await consumes on tile (0,0) MM2S channel 0}}
  aie.runtime_sequence @two_pushes_one_sync() {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    aiex.npu.push_queue (0, 0, MM2S:0) bd_id %c0 repeat %c0 {issue_token = true} : i32, i32
    aiex.npu.push_queue (0, 0, MM2S:0) bd_id %c1 repeat %c0 {issue_token = true} : i32, i32
    aiex.npu.sync(%c0, %c0, %c1, %c0, %c1, %c1) : i32, i32, i32, i32, i32, i32
  }
}

// -----

// An S2MM copy issues a token whether or not it says so: two outputs, one
// wait.
aie.device(npu2) {
  %tile_0_0 = aie.tile(0, 0)
  aie.shim_dma_allocation @out (%tile_0_0, S2MM, 0)
  // expected-error@+1 {{ends with 1 task-completion token no await consumes on tile (0,0) S2MM channel 0}}
  aie.runtime_sequence @two_outputs_one_wait(%arg0: memref<512xi32>) {
    aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0][1, 1, 1, 256][0, 0, 0, 1]) {id = 0 : i64, metadata = @out} : memref<512xi32>
    aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 256][1, 1, 1, 256][0, 0, 0, 1]) {id = 1 : i64, metadata = @out} : memref<512xi32>
    aiex.npu.dma_wait {symbol = @out}
  }
}

// -----

// The input's token lands on a channel nothing awaits, so nothing can take
// it: accepted.
aie.device(npu2) {
  %tile_0_0 = aie.tile(0, 0)
  aie.shim_dma_allocation @in (%tile_0_0, MM2S, 0)
  aie.shim_dma_allocation @out (%tile_0_0, S2MM, 0)
  aie.runtime_sequence @unawaited_input_token(%a: memref<256xi32>, %b: memref<256xi32>) {
    aiex.npu.dma_memcpy_nd(%b[0, 0, 0, 0][1, 1, 1, 256][0, 0, 0, 1]) {id = 1 : i64, metadata = @out} : memref<256xi32>
    aiex.npu.dma_memcpy_nd(%a[0, 0, 0, 0][1, 1, 1, 256][0, 0, 0, 1]) {id = 0 : i64, issue_token = true, metadata = @in} : memref<256xi32>
    aiex.npu.dma_wait {symbol = @out}
  }
}

// -----

// Another sequence awaits the channel, so a token this one leaves there would
// satisfy that await when it runs next.
aie.device(npu2) {
  // expected-error@+1 {{ends with 1 task-completion token no await consumes on tile (0,0) MM2S channel 0}}
  aie.runtime_sequence @leaves() {
    %c0 = arith.constant 0 : i32
    aiex.npu.push_queue (0, 0, MM2S:0) bd_id %c0 repeat %c0 {issue_token = true} : i32, i32
  }
  aie.runtime_sequence @awaits() {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    aiex.npu.push_queue (0, 0, MM2S:0) bd_id %c1 repeat %c0 {issue_token = true} : i32, i32
    aiex.npu.sync(%c0, %c0, %c1, %c0, %c1, %c1) : i32, i32, i32, i32, i32, i32
  }
}

// -----

// A token consumed on one branch only.
aie.device(npu2) {
  // expected-error@+1 {{ends with 1 task-completion token no await consumes on tile (0,0) MM2S channel 0}}
  aie.runtime_sequence @one_branch(%cond: i1) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    aiex.npu.push_queue (0, 0, MM2S:0) bd_id %c0 repeat %c0 {issue_token = true} : i32, i32
    scf.if %cond {
      aiex.npu.sync(%c0, %c0, %c1, %c0, %c1, %c1) : i32, i32, i32, i32, i32, i32
    }
  }
}

// -----

// Each branch consumes its own token: accepted.
aie.device(npu2) {
  aie.runtime_sequence @both_branches(%cond: i1) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    aiex.npu.push_queue (0, 0, MM2S:0) bd_id %c0 repeat %c0 {issue_token = true} : i32, i32
    scf.if %cond {
      aiex.npu.sync(%c0, %c0, %c1, %c0, %c1, %c1) : i32, i32, i32, i32, i32, i32
    } else {
      aiex.npu.push_queue (0, 0, MM2S:0) bd_id %c1 repeat %c0 {issue_token = false} : i32, i32
      aiex.npu.sync(%c0, %c0, %c1, %c0, %c1, %c1) : i32, i32, i32, i32, i32, i32
    }
  }
}

// -----

// A runtime loop that pushes a token every pass and syncs once after: what it
// leaves grows with its trip count.
aie.device(npu2) {
  aie.runtime_sequence @await_once_after_loop(%n: index) {
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    // expected-error@+1 {{changes how many task-completion tokens are outstanding on tile (0,0) MM2S channel 0 in each pass}}
    scf.for %i = %zero to %n step %one {
      aiex.npu.push_queue (0, 0, MM2S:0) bd_id %c0 repeat %c0 {issue_token = true} : i32, i32
    }
    aiex.npu.sync(%c0, %c0, %c1, %c0, %c1, %c1) : i32, i32, i32, i32, i32, i32
  }
}

// -----

// A ping-pong: each pass pushes the next transfer and syncs the previous one,
// and the last is synced after the loop. Accepted.
aie.device(npu2) {
  aie.runtime_sequence @pingpong(%n: index) {
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    aiex.npu.push_queue (0, 0, MM2S:0) bd_id %c0 repeat %c0 {issue_token = true} : i32, i32
    scf.for %i = %zero to %n step %one {
      aiex.npu.push_queue (0, 0, MM2S:0) bd_id %c1 repeat %c0 {issue_token = true} : i32, i32
      aiex.npu.sync(%c0, %c0, %c1, %c0, %c1, %c1) : i32, i32, i32, i32, i32, i32
    }
    aiex.npu.sync(%c0, %c0, %c1, %c0, %c1, %c1) : i32, i32, i32, i32, i32, i32
  }
}

// -----

// A loop with a known trip count is followed pass by pass: three tokens
// pushed in it, three syncs after. Accepted.
aie.device(npu2) {
  aie.runtime_sequence @known_trips() {
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %three = arith.constant 3 : index
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    scf.for %i = %zero to %three step %one {
      aiex.npu.push_queue (0, 0, MM2S:0) bd_id %c0 repeat %c0 {issue_token = true} : i32, i32
    }
    aiex.npu.sync(%c0, %c0, %c1, %c0, %c1, %c1) : i32, i32, i32, i32, i32, i32
    aiex.npu.sync(%c0, %c0, %c1, %c0, %c1, %c1) : i32, i32, i32, i32, i32, i32
    aiex.npu.sync(%c0, %c0, %c1, %c0, %c1, %c1) : i32, i32, i32, i32, i32, i32
  }
}
