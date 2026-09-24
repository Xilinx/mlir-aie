//===- decompose_large_dma_task_nd.mlir -----------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A task's descriptor may give more dimensions than a buffer descriptor holds
// (3 plus the iteration dimension) when they are all constant.
// aie-decompose-large-dma-bd reduces them. Every dimension past the innermost
// three is iterated, and one pass over the pattern runs the descriptor once per
// index of them all. A dimension that continues the one inside it merges with
// it, and a unit one drops. If more than one iteration dimension is left, the
// ones past the fourth are split off into a 4-dimension piece per index,
// outermost slowest, each at its own offset. Pieces run as separate tasks, since
// each one runs a part of a pass. A piece a descriptor still cannot hold is
// decomposed further.
//
// RUN: aie-opt --pass-pipeline='any(aie.device(aie-decompose-large-dma-bd))' \
// RUN:   --split-input-file %s | FileCheck %s
// RUN: aie-opt --pass-pipeline='any(aie.device(aie-substitute-shim-dma-allocations,aie-decompose-large-dma-bd,aie-assign-runtime-sequence-bd-ids,aie-dma-tasks-to-npu))' \
// RUN:   --split-input-file %s | FileCheck %s --check-prefix=LOWERED

// 12 iterations over [3 x 4]: the stride-262144 dimension does not continue the
// stride-32 one, so it splits into 3 pieces of 4 iterations. The token moves to
// the last piece, which the await names.
// CHECK-LABEL: @peel_5d
// CHECK:         %[[P0:.*]] = aiex.dma_configure_task_for @a
// CHECK-NEXT:      aie.dma_bd({{.*}} offset = 0 len = 2048 sizes = [4, 2, 32, 32] strides = [32, 4096, 1024, 1])
// CHECK-NEXT:      aie.end
// CHECK-NEXT:    } {repeat_count = 3 : i32}
// CHECK-NEXT:    aiex.dma_start_task(%[[P0]])
// CHECK-NEXT:    %[[P1:.*]] = aiex.dma_configure_task_for @a
// CHECK-NEXT:      aie.dma_bd({{.*}} offset = 262144 len = 2048 sizes = [4, 2, 32, 32] strides = [32, 4096, 1024, 1])
// CHECK-NEXT:      aie.end
// CHECK-NEXT:    } {repeat_count = 3 : i32}
// CHECK-NEXT:    aiex.dma_start_task(%[[P1]])
// CHECK-NEXT:    %[[P2:.*]] = aiex.dma_configure_task_for @a
// CHECK-NEXT:      aie.dma_bd({{.*}} offset = 524288 len = 2048 sizes = [4, 2, 32, 32] strides = [32, 4096, 1024, 1])
// CHECK-NEXT:      aie.end
// CHECK-NEXT:    } {issue_token = true, repeat_count = 3 : i32}
// CHECK-NEXT:    aiex.dma_start_task(%[[P2]])
// CHECK-NEXT:    aiex.dma_await_task(%[[P2]])
// CHECK-NEXT:    }
// LOWERED-LABEL: @peel_5d
// LOWERED-COUNT-3: aiex.npu.writebd
// LOWERED-NOT:     aiex.npu.writebd
// LOWERED:         aiex.npu.sync
// LOWERED-NEXT:    }
module {
  aie.device(npu2_1col) {
    %t = aie.tile(0, 0)
    aie.shim_dma_allocation @a (%t, MM2S, 0)
    aie.runtime_sequence @peel_5d(%in: memref<1048576xbf16>) {
      %0 = aiex.dma_configure_task_for @a {
        aie.dma_bd(%in : memref<1048576xbf16> offset = 0 len = 2048 sizes = [3, 4, 2, 32, 32] strides = [262144, 32, 4096, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 11 : i32}
      aiex.dma_start_task(%0)
      aiex.dma_await_task(%0)
    }
  }
}

// -----

// Stride 300 continues [3 x 100], so the two iteration dimensions merge into
// one of 6 and the descriptor stays whole.
// CHECK-LABEL: @merge_5d
// CHECK:         %[[T:.*]] = aiex.dma_configure_task_for @a
// CHECK-NEXT:      aie.dma_bd({{.*}} offset = 0 len = 128 sizes = [6, 1, 8, 16] strides = [100, 0, 32, 1])
// CHECK-NEXT:      aie.end
// CHECK-NEXT:    } {issue_token = true, repeat_count = 5 : i32}
// CHECK-NEXT:    aiex.dma_start_task(%[[T]])
// CHECK-NEXT:    aiex.dma_await_task(%[[T]])
module {
  aie.device(npu2_1col) {
    %t = aie.tile(0, 0)
    aie.shim_dma_allocation @a (%t, MM2S, 0)
    aie.runtime_sequence @merge_5d(%in: memref<4096xi32>) {
      %0 = aiex.dma_configure_task_for @a {
        aie.dma_bd(%in : memref<4096xi32> offset = 0 len = 128 sizes = [2, 3, 1, 8, 16] strides = [300, 100, 0, 32, 1])
        aie.end
      } {issue_token = true, repeat_count = 5 : i32}
      aiex.dma_start_task(%0)
      aiex.dma_await_task(%0)
    }
  }
}

// -----

// A unit iteration dimension drops.
// CHECK-LABEL: @unit_5d
// CHECK:         aiex.dma_configure_task_for @a
// CHECK-NEXT:      aie.dma_bd({{.*}} offset = 0 len = 128 sizes = [3, 1, 8, 16] strides = [200, 0, 32, 1])
// CHECK-NEXT:      aie.end
// CHECK-NEXT:    } {issue_token = true, repeat_count = 2 : i32}
module {
  aie.device(npu2_1col) {
    %t = aie.tile(0, 0)
    aie.shim_dma_allocation @a (%t, MM2S, 0)
    aie.runtime_sequence @unit_5d(%in: memref<4096xi32>) {
      %0 = aiex.dma_configure_task_for @a {
        aie.dma_bd(%in : memref<4096xi32> offset = 0 len = 128 sizes = [1, 3, 1, 8, 16] strides = [0, 200, 0, 32, 1])
        aie.end
      } {issue_token = true, repeat_count = 2 : i32}
      aiex.dma_start_task(%0)
      aiex.dma_await_task(%0)
    }
  }
}

// -----

// Two dimensions past the fourth, [3 x 2] at strides 20000 and 7000, split off
// into 6 pieces from the base offset of 16, the outer one slowest. A free
// frees every piece. Past the channel's 4 queue slots the compiler waits for
// queue space before each push.
// CHECK-LABEL: @peel_6d
// CHECK:         %[[P0:.*]] = aiex.dma_configure_task_for @a
// CHECK-NEXT:      aie.dma_bd({{.*}} offset = 16 len = 128 sizes = [2, 1, 8, 16] strides = [1000, 0, 32, 1])
// CHECK:         } {repeat_count = 1 : i32}
// CHECK:           aie.dma_bd({{.*}} offset = 7016 len = 128
// CHECK:           aie.dma_bd({{.*}} offset = 20016 len = 128
// CHECK:           aie.dma_bd({{.*}} offset = 27016 len = 128
// CHECK:           aie.dma_bd({{.*}} offset = 40016 len = 128
// CHECK:         %[[P5:.*]] = aiex.dma_configure_task_for @a
// CHECK-NEXT:      aie.dma_bd({{.*}} offset = 47016 len = 128 sizes = [2, 1, 8, 16] strides = [1000, 0, 32, 1])
// CHECK-NEXT:      aie.end
// CHECK-NEXT:    } {issue_token = true, repeat_count = 1 : i32}
// CHECK-NEXT:    aiex.dma_start_task(%[[P5]])
// CHECK-NEXT:    aiex.dma_await_task(%[[P5]])
// CHECK-NEXT:    aiex.dma_free_task(%[[P0]])
// CHECK-COUNT-5: aiex.dma_free_task
// CHECK-NEXT:    }
// LOWERED-LABEL: @peel_6d
// LOWERED-COUNT-4: aiex.npu.push_queue
// LOWERED:         aiex.npu.maskpoll
// LOWERED:         aiex.npu.push_queue
// LOWERED:         aiex.npu.maskpoll
// LOWERED:         aiex.npu.push_queue
// LOWERED-NOT:     aiex.npu.push_queue
// LOWERED:         aiex.npu.sync
module {
  aie.device(npu2_1col) {
    %t = aie.tile(0, 0)
    aie.shim_dma_allocation @a (%t, MM2S, 0)
    aie.runtime_sequence @peel_6d(%in: memref<65536xi32>) {
      %0 = aiex.dma_configure_task_for @a {
        aie.dma_bd(%in : memref<65536xi32> offset = 16 len = 128 sizes = [3, 2, 2, 1, 8, 16] strides = [20000, 7000, 1000, 0, 32, 1])
        aie.end
      } {issue_token = true, repeat_count = 11 : i32}
      aiex.dma_start_task(%0)
      aiex.dma_await_task(%0)
      aiex.dma_free_task(%0)
    }
  }
}

// -----

// Each of the 2 pieces is [2 x 1031 x 2] at stride 3, which a descriptor
// cannot hold and cannot factor: each index of its iteration dimension is
// sliced into 1023 + 8, in order.
// CHECK-LABEL: @peel_then_slice
// CHECK:         aie.dma_bd({{.*}} offset = 0 len = 2046 sizes = [1, 1, 1023, 2] strides = [3500, 0, 3, 1])
// CHECK:         aie.dma_bd({{.*}} offset = 3069 len = 16 sizes = [1, 1, 8, 2] strides = [3500, 0, 3, 1])
// CHECK:         aie.dma_bd({{.*}} offset = 3500 len = 2046 sizes = [1, 1, 1023, 2]
// CHECK:         aie.dma_bd({{.*}} offset = 6569 len = 16 sizes = [1, 1, 8, 2]
// CHECK:         aie.dma_bd({{.*}} offset = 9000 len = 2046 sizes = [1, 1, 1023, 2]
// CHECK:         aie.dma_bd({{.*}} offset = 12069 len = 16 sizes = [1, 1, 8, 2]
// CHECK:         aie.dma_bd({{.*}} offset = 12500 len = 2046 sizes = [1, 1, 1023, 2]
// CHECK:         %[[L:.*]] = aiex.dma_configure_task_for @a
// CHECK-NEXT:      aie.dma_bd({{.*}} offset = 15569 len = 16 sizes = [1, 1, 8, 2]
// CHECK-NEXT:      aie.end
// CHECK-NEXT:    } {issue_token = true}
// CHECK-NEXT:    aiex.dma_start_task(%[[L]])
// CHECK-NEXT:    aiex.dma_await_task(%[[L]])
// LOWERED-LABEL: @peel_then_slice
// LOWERED-COUNT-8: aiex.npu.push_queue
// LOWERED-NOT:     aiex.npu.push_queue
// LOWERED:         aiex.npu.sync
module {
  aie.device(npu2_1col) {
    %t = aie.tile(0, 0)
    aie.shim_dma_allocation @a (%t, MM2S, 0)
    aie.runtime_sequence @peel_then_slice(%in: memref<65536xi32>) {
      %0 = aiex.dma_configure_task_for @a {
        aie.dma_bd(%in : memref<65536xi32> offset = 0 len = 2062 sizes = [2, 2, 1, 1031, 2] strides = [9000, 3500, 0, 3, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%0)
      aiex.dma_await_task(%0)
    }
  }
}

// -----

// A repeat count of 7 over [2 x 2] runs 2 passes: each piece is restarted for
// the second, and only the last start issues the token.
// CHECK-LABEL: @two_passes
// CHECK:         %[[P0:.*]] = aiex.dma_configure_task_for @a
// CHECK-NEXT:      aie.dma_bd({{.*}} offset = 0 len = 128 sizes = [2, 1, 8, 16] strides = [3500, 0, 32, 1])
// CHECK-NEXT:      aie.end
// CHECK-NEXT:    } {repeat_count = 1 : i32}
// CHECK-NEXT:    aiex.dma_start_task(%[[P0]])
// CHECK-NEXT:    %[[P1:.*]] = aiex.dma_configure_task_for @a
// CHECK-NEXT:      aie.dma_bd({{.*}} offset = 9000 len = 128 sizes = [2, 1, 8, 16] strides = [3500, 0, 32, 1])
// CHECK-NEXT:      aie.end
// CHECK-NEXT:    } {issue_token = true, repeat_count = 1 : i32}
// CHECK-NEXT:    aiex.dma_start_task(%[[P1]]) {no_token}
// CHECK-NEXT:    aiex.dma_start_task(%[[P0]])
// CHECK-NEXT:    aiex.dma_start_task(%[[P1]])
// CHECK-NEXT:    aiex.dma_await_task(%[[P1]])
module {
  aie.device(npu2_1col) {
    %t = aie.tile(0, 0)
    aie.shim_dma_allocation @a (%t, MM2S, 0)
    aie.runtime_sequence @two_passes(%in: memref<65536xi32>) {
      %tk = aiex.dma_configure_task_for @a {
        aie.dma_bd(%in : memref<65536xi32> offset = 0 len = 128 sizes = [2, 2, 1, 8, 16] strides = [9000, 3500, 0, 32, 1])
        aie.end
      } {issue_token = true, repeat_count = 7 : i32}
      aiex.dma_start_task(%tk)
      aiex.dma_await_task(%tk)
    }
  }
}
