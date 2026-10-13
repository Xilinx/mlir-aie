// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: aie-opt --aie-dma-to-npu --split-input-file %s | FileCheck %s
// RUN: aie-opt --aie-dma-to-npu='enforce-queue-depth=false' --split-input-file %s 2>/dev/null | FileCheck %s --check-prefix=OFF

// The inner loop can fill the queue before the outer loop's sync. Walking its
// body just once would incorrectly make this look like push-then-await.
// CHECK-LABEL: @nested
// CHECK: scf.for
// CHECK: scf.for
// CHECK: aiex.npu.maskpoll
// CHECK: aiex.npu.write32
// CHECK: aiex.npu.sync
// OFF-LABEL: @nested
// OFF-NOT: aiex.npu.maskpoll
aie.device(npu2) {
  aie.runtime_sequence @nested(%n: index, %m: index) {
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    scf.for %i = %zero to %n step %one {
      scf.for %j = %zero to %m step %one {
        aiex.npu.push_queue (0, 0, MM2S:0) bd_id %c0 repeat %c0 {issue_token = true} : i32, i32
      }
      aiex.npu.sync(%c0, %c0, %c1, %c0, %c1, %c1) : i32, i32, i32, i32, i32, i32
    }
  }
}

// -----

// A runtime loop may execute zero times: its waits do not necessarily drain
// the prefix. Keep the entry state among the possible exits.
// CHECK-LABEL: @possibly_zero
// CHECK-NOT: aiex.npu.maskpoll
// CHECK: scf.for
// CHECK: aiex.npu.sync
// CHECK: }
// CHECK: aiex.npu.maskpoll
// CHECK: aiex.npu.write32
// OFF-LABEL: @possibly_zero
// OFF-NOT: aiex.npu.maskpoll
aie.device(npu2) {
  aie.runtime_sequence @possibly_zero(%n: index) {
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    aiex.npu.push_queue (0, 0, MM2S:0) bd_id %c0 repeat %c0 {issue_token = true} : i32, i32
    aiex.npu.push_queue (0, 0, MM2S:0) bd_id %c0 repeat %c0 {issue_token = true} : i32, i32
    aiex.npu.push_queue (0, 0, MM2S:0) bd_id %c0 repeat %c0 {issue_token = true} : i32, i32
    aiex.npu.push_queue (0, 0, MM2S:0) bd_id %c0 repeat %c0 {issue_token = true} : i32, i32
    scf.for %i = %zero to %n step %one {
      aiex.npu.sync(%c0, %c0, %c1, %c0, %c1, %c1) : i32, i32, i32, i32, i32, i32
    }
    aiex.npu.push_queue (0, 0, MM2S:0) bd_id %c0 repeat %c0 {issue_token = false} : i32, i32
  }
}

// -----

// Statically empty loops have no queue effects, including their nested loops.
// CHECK-LABEL: @known_zero
// CHECK-NOT: aiex.npu.maskpoll
// OFF-LABEL: @known_zero
// OFF-NOT: aiex.npu.maskpoll
aie.device(npu2) {
  aie.runtime_sequence @known_zero(%n: index) {
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %c0 = arith.constant 0 : i32
    scf.for %i = %zero to %zero step %one {
      scf.for %j = %zero to %n step %one {
        aiex.npu.push_queue (0, 0, MM2S:0) bd_id %c0 repeat %c0 {issue_token = false} : i32, i32
      }
    }
    aiex.npu.push_queue (0, 0, MM2S:0) bd_id %c0 repeat %c0 {issue_token = false} : i32, i32
    aiex.npu.push_queue (0, 0, MM2S:0) bd_id %c0 repeat %c0 {issue_token = false} : i32, i32
    aiex.npu.push_queue (0, 0, MM2S:0) bd_id %c0 repeat %c0 {issue_token = false} : i32, i32
    aiex.npu.push_queue (0, 0, MM2S:0) bd_id %c0 repeat %c0 {issue_token = false} : i32, i32
  }
}

// -----

// A wait on only one branch cannot make the following push safe.
// CHECK-LABEL: @conditional_wait
// CHECK-NOT: aiex.npu.maskpoll
// CHECK: scf.if
// CHECK: aiex.npu.sync
// CHECK: }
// CHECK: aiex.npu.maskpoll
// CHECK: aiex.npu.write32
// OFF-LABEL: @conditional_wait
// OFF-NOT: aiex.npu.maskpoll
aie.device(npu2) {
  aie.runtime_sequence @conditional_wait(%cond: i1) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    aiex.npu.push_queue (0, 0, MM2S:0) bd_id %c0 repeat %c0 {issue_token = true} : i32, i32
    aiex.npu.push_queue (0, 0, MM2S:0) bd_id %c0 repeat %c0 {issue_token = true} : i32, i32
    aiex.npu.push_queue (0, 0, MM2S:0) bd_id %c0 repeat %c0 {issue_token = true} : i32, i32
    aiex.npu.push_queue (0, 0, MM2S:0) bd_id %c0 repeat %c0 {issue_token = true} : i32, i32
    scf.if %cond {
      aiex.npu.sync(%c0, %c0, %c1, %c0, %c1, %c1) : i32, i32, i32, i32, i32, i32
    }
    aiex.npu.push_queue (0, 0, MM2S:0) bd_id %c0 repeat %c0 {issue_token = false} : i32, i32
  }
}

// -----

// Mutually exclusive branch pushes are not added together.
// CHECK-LABEL: @exclusive_pushes
// CHECK-NOT: aiex.npu.maskpoll
// OFF-LABEL: @exclusive_pushes
// OFF-NOT: aiex.npu.maskpoll
aie.device(npu2) {
  aie.runtime_sequence @exclusive_pushes(%cond: i1) {
    %c0 = arith.constant 0 : i32
    scf.if %cond {
      aiex.npu.push_queue (0, 0, MM2S:0) bd_id %c0 repeat %c0 {issue_token = false} : i32, i32
      aiex.npu.push_queue (0, 0, MM2S:0) bd_id %c0 repeat %c0 {issue_token = false} : i32, i32
    } else {
      aiex.npu.push_queue (0, 0, MM2S:0) bd_id %c0 repeat %c0 {issue_token = false} : i32, i32
      aiex.npu.push_queue (0, 0, MM2S:0) bd_id %c0 repeat %c0 {issue_token = false} : i32, i32
    }
    aiex.npu.push_queue (0, 0, MM2S:0) bd_id %c0 repeat %c0 {issue_token = false} : i32, i32
  }
}

// -----

// A queue-space poll can complete the oldest push, but does not consume its
// token. The sync cannot be credited to the fifth push's token instead.
// CHECK-LABEL: @poll_preserves_pending_token
// CHECK-COUNT-4: aiex.npu.write32
// CHECK: aiex.npu.maskpoll
// CHECK: aiex.npu.write32
// CHECK: aiex.npu.sync
// CHECK: aiex.npu.maskpoll
// CHECK: aiex.npu.write32
// OFF-LABEL: @poll_preserves_pending_token
// OFF-NOT: aiex.npu.maskpoll
aie.device(npu2) {
  aie.runtime_sequence @poll_preserves_pending_token() {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    aiex.npu.push_queue (0, 0, MM2S:0) bd_id %c0 repeat %c0 {issue_token = true} : i32, i32
    aiex.npu.push_queue (0, 0, MM2S:0) bd_id %c0 repeat %c0 {issue_token = false} : i32, i32
    aiex.npu.push_queue (0, 0, MM2S:0) bd_id %c0 repeat %c0 {issue_token = false} : i32, i32
    aiex.npu.push_queue (0, 0, MM2S:0) bd_id %c0 repeat %c0 {issue_token = false} : i32, i32
    aiex.npu.push_queue (0, 0, MM2S:0) bd_id %c0 repeat %c0 {issue_token = true} : i32, i32
    aiex.npu.sync(%c0, %c0, %c1, %c0, %c1, %c1) : i32, i32, i32, i32, i32, i32
    aiex.npu.push_queue (0, 0, MM2S:0) bd_id %c0 repeat %c0 {issue_token = false} : i32, i32
  }
}

// -----

// Keep token order as well as occupancy at joins. The then-path token retires
// only its first push; the else-path token would retire both.
// CHECK-LABEL: @branch_token_order
// CHECK: scf.if
// CHECK: aiex.npu.sync
// CHECK-COUNT-3: aiex.npu.write32
// CHECK: aiex.npu.maskpoll
// CHECK: aiex.npu.write32
// OFF-LABEL: @branch_token_order
// OFF-NOT: aiex.npu.maskpoll
aie.device(npu2) {
  aie.runtime_sequence @branch_token_order(%cond: i1) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    scf.if %cond {
      aiex.npu.push_queue (0, 0, MM2S:0) bd_id %c0 repeat %c0 {issue_token = true} : i32, i32
      aiex.npu.push_queue (0, 0, MM2S:0) bd_id %c0 repeat %c0 {issue_token = false} : i32, i32
    } else {
      aiex.npu.push_queue (0, 0, MM2S:0) bd_id %c0 repeat %c0 {issue_token = false} : i32, i32
      aiex.npu.push_queue (0, 0, MM2S:0) bd_id %c0 repeat %c0 {issue_token = true} : i32, i32
    }
    aiex.npu.sync(%c0, %c0, %c1, %c0, %c1, %c1) : i32, i32, i32, i32, i32, i32
    aiex.npu.push_queue (0, 0, MM2S:0) bd_id %c0 repeat %c0 {issue_token = false} : i32, i32
    aiex.npu.push_queue (0, 0, MM2S:0) bd_id %c0 repeat %c0 {issue_token = false} : i32, i32
    aiex.npu.push_queue (0, 0, MM2S:0) bd_id %c0 repeat %c0 {issue_token = false} : i32, i32
    aiex.npu.push_queue (0, 0, MM2S:0) bd_id %c0 repeat %c0 {issue_token = false} : i32, i32
  }
}

// -----

// Existing queue-space polls are executed every iteration and must not be
// duplicated, even when the loop's entry state already has a full queue.
// CHECK-LABEL: @existing_loop_guard
// CHECK-COUNT-4: aiex.npu.write32
// CHECK: scf.for
// CHECK: aiex.npu.maskpoll
// CHECK-NOT: aiex.npu.maskpoll
// CHECK: aiex.npu.write32
// CHECK-NOT: aiex.npu.maskpoll
// OFF-LABEL: @existing_loop_guard
// OFF: aiex.npu.maskpoll
// OFF-NOT: aiex.npu.maskpoll
aie.device(npu2) {
  aie.runtime_sequence @existing_loop_guard(%n: index) {
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %c0 = arith.constant 0 : i32
    %status = arith.constant 119336 : i32
    %mask = arith.constant 4194304 : i32
    aiex.npu.push_queue (0, 0, MM2S:0) bd_id %c0 repeat %c0 {issue_token = false} : i32, i32
    aiex.npu.push_queue (0, 0, MM2S:0) bd_id %c0 repeat %c0 {issue_token = false} : i32, i32
    aiex.npu.push_queue (0, 0, MM2S:0) bd_id %c0 repeat %c0 {issue_token = false} : i32, i32
    aiex.npu.push_queue (0, 0, MM2S:0) bd_id %c0 repeat %c0 {issue_token = false} : i32, i32
    scf.for %i = %zero to %n step %one {
      aiex.npu.maskpoll(%status, %c0, %mask) : i32, i32, i32
      aiex.npu.push_queue (0, 0, MM2S:0) bd_id %c0 repeat %c0 {issue_token = false} : i32, i32
    }
  }
}

// -----

// Seven independent branches have 128 combined queue states, but only two per
// channel. Analyze channels separately: there is no overflow or need to widen
// to full queues merely because there are many independent branches.
// CHECK-LABEL: @bounded_disjunctions
// CHECK-NOT: aiex.npu.maskpoll
// OFF-LABEL: @bounded_disjunctions
// OFF-NOT: aiex.npu.maskpoll
aie.device(npu2) {
  aie.runtime_sequence @bounded_disjunctions(%a: i1, %b: i1, %c: i1, %d: i1, %e: i1, %f: i1, %g: i1) {
    %c0 = arith.constant 0 : i32
    scf.if %a {
      aiex.npu.push_queue (0, 0, MM2S:0) bd_id %c0 repeat %c0 {issue_token = false} : i32, i32
    }
    scf.if %b {
      aiex.npu.push_queue (0, 0, MM2S:1) bd_id %c0 repeat %c0 {issue_token = false} : i32, i32
    }
    scf.if %c {
      aiex.npu.push_queue (0, 2, MM2S:0) bd_id %c0 repeat %c0 {issue_token = false} : i32, i32
    }
    scf.if %d {
      aiex.npu.push_queue (0, 2, MM2S:1) bd_id %c0 repeat %c0 {issue_token = false} : i32, i32
    }
    scf.if %e {
      aiex.npu.push_queue (0, 3, MM2S:0) bd_id %c0 repeat %c0 {issue_token = false} : i32, i32
    }
    scf.if %f {
      aiex.npu.push_queue (0, 3, MM2S:1) bd_id %c0 repeat %c0 {issue_token = false} : i32, i32
    }
    scf.if %g {
      aiex.npu.push_queue (0, 4, MM2S:0) bd_id %c0 repeat %c0 {issue_token = false} : i32, i32
    }
    aiex.npu.push_queue (0, 0, MM2S:0) bd_id %c0 repeat %c0 {issue_token = false} : i32, i32
  }
}
