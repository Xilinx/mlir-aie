// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: aie-opt --aie-dma-to-npu --split-input-file %s | FileCheck %s

// Each channel is analyzed over its own ops. A loop or branch holding only
// another channel's pushes leaves this channel's queue as it was, and each
// channel's fifth push is guarded where it is, by a poll on its own status.
// CHECK-LABEL: @interleaved
// CHECK-COUNT-4: aiex.npu.write32
// CHECK: scf.for
// CHECK-NOT: aiex.npu.maskpoll
// CHECK: aiex.npu.sync
// CHECK: scf.if
// CHECK-COUNT-4: aiex.npu.write32
// CHECK-NOT: aiex.npu.maskpoll
// CHECK: }
// CHECK-NOT: aiex.npu.write32
// CHECK: aiex.npu.maskpoll(%c119336_i32
// CHECK-NOT: aiex.npu.maskpoll
// CHECK: aiex.npu.write32
// CHECK-NOT: aiex.npu.write32
// CHECK: aiex.npu.maskpoll(%c119340_i32
// CHECK-NOT: aiex.npu.maskpoll
// CHECK: aiex.npu.write32
// CHECK-NOT: aiex.npu.maskpoll
aie.device(npu2) {
  aie.runtime_sequence @interleaved(%n: index, %cond: i1) {
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    aiex.npu.push_queue (0, 0, MM2S:0) bd_id %c0 repeat %c0 {issue_token = false} : i32, i32
    aiex.npu.push_queue (0, 0, MM2S:0) bd_id %c0 repeat %c0 {issue_token = false} : i32, i32
    aiex.npu.push_queue (0, 0, MM2S:0) bd_id %c0 repeat %c0 {issue_token = false} : i32, i32
    aiex.npu.push_queue (0, 0, MM2S:0) bd_id %c0 repeat %c0 {issue_token = false} : i32, i32
    scf.for %i = %zero to %n step %one {
      aiex.npu.push_queue (0, 0, MM2S:1) bd_id %c0 repeat %c0 {issue_token = true} : i32, i32
      aiex.npu.sync(%c0, %c0, %c1, %c1, %c1, %c1) : i32, i32, i32, i32, i32, i32
    }
    scf.if %cond {
      aiex.npu.push_queue (0, 0, MM2S:1) bd_id %c0 repeat %c0 {issue_token = false} : i32, i32
      aiex.npu.push_queue (0, 0, MM2S:1) bd_id %c0 repeat %c0 {issue_token = false} : i32, i32
      aiex.npu.push_queue (0, 0, MM2S:1) bd_id %c0 repeat %c0 {issue_token = false} : i32, i32
      aiex.npu.push_queue (0, 0, MM2S:1) bd_id %c0 repeat %c0 {issue_token = false} : i32, i32
    }
    aiex.npu.push_queue (0, 0, MM2S:0) bd_id %c0 repeat %c0 {issue_token = false} : i32, i32
    aiex.npu.push_queue (0, 0, MM2S:1) bd_id %c0 repeat %c0 {issue_token = false} : i32, i32
  }
}

// -----

// A queue-space poll credits only the channel whose status it reads.
// CHECK-LABEL: @poll_other_channel
// CHECK-COUNT-8: aiex.npu.write32
// CHECK-NOT: aiex.npu.write32
// CHECK: aiex.npu.maskpoll(%c119340_i32
// CHECK-NOT: aiex.npu.write32
// CHECK: aiex.npu.maskpoll(%c119336_i32
// CHECK-NOT: aiex.npu.maskpoll
// CHECK: aiex.npu.write32
// CHECK-NOT: aiex.npu.maskpoll
// CHECK: aiex.npu.write32
// CHECK-NOT: aiex.npu.maskpoll
aie.device(npu2) {
  aie.runtime_sequence @poll_other_channel() {
    %c0 = arith.constant 0 : i32
    %status = arith.constant 119340 : i32
    %mask = arith.constant 4194304 : i32
    aiex.npu.push_queue (0, 0, MM2S:0) bd_id %c0 repeat %c0 {issue_token = false} : i32, i32
    aiex.npu.push_queue (0, 0, MM2S:0) bd_id %c0 repeat %c0 {issue_token = false} : i32, i32
    aiex.npu.push_queue (0, 0, MM2S:0) bd_id %c0 repeat %c0 {issue_token = false} : i32, i32
    aiex.npu.push_queue (0, 0, MM2S:0) bd_id %c0 repeat %c0 {issue_token = false} : i32, i32
    aiex.npu.push_queue (0, 0, MM2S:1) bd_id %c0 repeat %c0 {issue_token = false} : i32, i32
    aiex.npu.push_queue (0, 0, MM2S:1) bd_id %c0 repeat %c0 {issue_token = false} : i32, i32
    aiex.npu.push_queue (0, 0, MM2S:1) bd_id %c0 repeat %c0 {issue_token = false} : i32, i32
    aiex.npu.push_queue (0, 0, MM2S:1) bd_id %c0 repeat %c0 {issue_token = false} : i32, i32
    aiex.npu.maskpoll(%status, %c0, %mask) : i32, i32, i32
    aiex.npu.push_queue (0, 0, MM2S:0) bd_id %c0 repeat %c0 {issue_token = false} : i32, i32
    aiex.npu.push_queue (0, 0, MM2S:1) bd_id %c0 repeat %c0 {issue_token = false} : i32, i32
  }
}
