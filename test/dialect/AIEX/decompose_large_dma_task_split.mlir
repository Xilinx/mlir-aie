//===- decompose_large_dma_task_split.mlir ---------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// aie-decompose-large-dma-bd issues a task-path transfer as one task per slice
// when it slices into more descriptors than the channel queues (4 on npu2), or
// when its slices need repeat counts of their own. Only the last slice issues
// the token: an await of the task awaits it, and a free frees every slice.
//
// Each slice then waits for the ones before it, so its pushes are interleaved
// with the transfers issued alongside it: slice 0 keeps its place, and the rest
// follow the last start issued alongside it, ordered by how far through its
// transfer each is. A transfer issued ahead of its counterpart would otherwise
// have the compiler's waits for its later slices block the counterpart's
// pushes.
//
// An i32 transfer of p x 2 elements at stride 3, p prime, cannot be factored,
// so it is sliced into ceil(p / 1023) descriptors: p = 4099 gives 5, 5119
// gives 6, 8179 gives 8 and 17393 gives 18.
//
// RUN: aie-opt --pass-pipeline='any(aie.device(aie-decompose-large-dma-bd))' \
// RUN:   --split-input-file %s | FileCheck %s
// RUN: aie-opt --pass-pipeline='any(aie.device(aie-substitute-shim-dma-allocations,aie-decompose-large-dma-bd,aie-assign-runtime-sequence-bd-ids))' \
// RUN:   --split-input-file %s | FileCheck %s --check-prefix=LOWERED

// A transfer of 5 slices issued after a whole one. Only the later task
// splits, and its slices stay in order. The token moves to the last slice,
// which the await names; the free frees every slice.
// CHECK-LABEL: @split_after_whole
// CHECK:         %[[C:.*]] = aiex.dma_configure_task_for @c
// CHECK-NEXT:      aie.dma_bd({{.*}} offset = 0 len
// CHECK-NEXT:      aie.end
// CHECK-NEXT:    } {issue_token = true}
// CHECK-NEXT:    aiex.dma_start_task(%[[C]])
// CHECK-NEXT:    %[[A0:.*]] = aiex.dma_configure_task_for @a
// CHECK-NEXT:      aie.dma_bd({{.*}} offset = 0 len
// CHECK-NEXT:      aie.end
// CHECK-NEXT:    }{{$}}
// CHECK-NEXT:    aiex.dma_start_task(%[[A0]])
// CHECK-NEXT:    %[[A1:.*]] = aiex.dma_configure_task_for @a
// CHECK-NEXT:      aie.dma_bd({{.*}} offset = 3069 len
// CHECK-NEXT:      aie.end
// CHECK-NEXT:    }{{$}}
// CHECK-NEXT:    aiex.dma_start_task(%[[A1]])
// CHECK-NEXT:    %[[A2:.*]] = aiex.dma_configure_task_for @a
// CHECK-NEXT:      aie.dma_bd({{.*}} offset = 6138 len
// CHECK-NEXT:      aie.end
// CHECK-NEXT:    }{{$}}
// CHECK-NEXT:    aiex.dma_start_task(%[[A2]])
// CHECK-NEXT:    %[[A3:.*]] = aiex.dma_configure_task_for @a
// CHECK-NEXT:      aie.dma_bd({{.*}} offset = 9207 len
// CHECK-NEXT:      aie.end
// CHECK-NEXT:    }{{$}}
// CHECK-NEXT:    aiex.dma_start_task(%[[A3]])
// CHECK-NEXT:    %[[A4:.*]] = aiex.dma_configure_task_for @a
// CHECK-NEXT:      aie.dma_bd({{.*}} offset = 12276 len
// CHECK-NEXT:      aie.end
// CHECK-NEXT:    } {issue_token = true}
// CHECK-NEXT:    aiex.dma_start_task(%[[A4]])
// CHECK-NEXT:    aiex.dma_await_task(%[[C]])
// CHECK-NEXT:    aiex.dma_await_task(%[[A4]])
// CHECK-NEXT:    aiex.dma_free_task(%[[A0]])
// CHECK-NEXT:    aiex.dma_free_task(%[[A1]])
// CHECK-NEXT:    aiex.dma_free_task(%[[A2]])
// CHECK-NEXT:    aiex.dma_free_task(%[[A3]])
// CHECK-NEXT:    aiex.dma_free_task(%[[A4]])
// CHECK-NEXT:    }
module {
  aie.device(npu2_1col) {
    %t = aie.tile(0, 0)
    aie.shim_dma_allocation @a (%t, MM2S, 0)
    aie.shim_dma_allocation @c (%t, S2MM, 0)
    aie.runtime_sequence @split_after_whole(%in: memref<32768xi32>, %out: memref<32768xi32>) {
      %c = aiex.dma_configure_task_for @c {
        aie.dma_bd(%out : memref<32768xi32> offset = 0 len = 64 sizes = [1, 1, 1, 64] strides = [0, 0, 0, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c)
      %a = aiex.dma_configure_task_for @a {
        aie.dma_bd(%in : memref<32768xi32> offset = 0 len = 8198 sizes = [1, 1, 4099, 2] strides = [0, 0, 3, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%a)
      aiex.dma_await_task(%c)
      aiex.dma_await_task(%a)
      aiex.dma_free_task(%a)
    }
  }
}

// -----

// Two 5-slice transfers side by side alternate slice by slice.
// CHECK-LABEL: @both_split
// CHECK:         %[[C0:.*]] = aiex.dma_configure_task_for @c
// CHECK-NEXT:      aie.dma_bd({{.*}} offset = 0 len
// CHECK-NEXT:      aie.end
// CHECK-NEXT:    }{{$}}
// CHECK-NEXT:    aiex.dma_start_task(%[[C0]])
// CHECK-NEXT:    %[[A0:.*]] = aiex.dma_configure_task_for @a
// CHECK-NEXT:      aie.dma_bd({{.*}} offset = 0 len
// CHECK-NEXT:      aie.end
// CHECK-NEXT:    }{{$}}
// CHECK-NEXT:    aiex.dma_start_task(%[[A0]])
// CHECK-NEXT:    %[[C1:.*]] = aiex.dma_configure_task_for @c
// CHECK-NEXT:      aie.dma_bd({{.*}} offset = 3069 len
// CHECK-NEXT:      aie.end
// CHECK-NEXT:    }{{$}}
// CHECK-NEXT:    aiex.dma_start_task(%[[C1]])
// CHECK-NEXT:    %[[A1:.*]] = aiex.dma_configure_task_for @a
// CHECK-NEXT:      aie.dma_bd({{.*}} offset = 3069 len
// CHECK-NEXT:      aie.end
// CHECK-NEXT:    }{{$}}
// CHECK-NEXT:    aiex.dma_start_task(%[[A1]])
// CHECK-NEXT:    %[[C2:.*]] = aiex.dma_configure_task_for @c
// CHECK-NEXT:      aie.dma_bd({{.*}} offset = 6138 len
// CHECK-NEXT:      aie.end
// CHECK-NEXT:    }{{$}}
// CHECK-NEXT:    aiex.dma_start_task(%[[C2]])
// CHECK-NEXT:    %[[A2:.*]] = aiex.dma_configure_task_for @a
// CHECK-NEXT:      aie.dma_bd({{.*}} offset = 6138 len
// CHECK-NEXT:      aie.end
// CHECK-NEXT:    }{{$}}
// CHECK-NEXT:    aiex.dma_start_task(%[[A2]])
// CHECK-NEXT:    %[[C3:.*]] = aiex.dma_configure_task_for @c
// CHECK-NEXT:      aie.dma_bd({{.*}} offset = 9207 len
// CHECK-NEXT:      aie.end
// CHECK-NEXT:    }{{$}}
// CHECK-NEXT:    aiex.dma_start_task(%[[C3]])
// CHECK-NEXT:    %[[A3:.*]] = aiex.dma_configure_task_for @a
// CHECK-NEXT:      aie.dma_bd({{.*}} offset = 9207 len
// CHECK-NEXT:      aie.end
// CHECK-NEXT:    }{{$}}
// CHECK-NEXT:    aiex.dma_start_task(%[[A3]])
// CHECK-NEXT:    %[[C4:.*]] = aiex.dma_configure_task_for @c
// CHECK-NEXT:      aie.dma_bd({{.*}} offset = 12276 len
// CHECK-NEXT:      aie.end
// CHECK-NEXT:    } {issue_token = true}
// CHECK-NEXT:    aiex.dma_start_task(%[[C4]])
// CHECK-NEXT:    %[[A4:.*]] = aiex.dma_configure_task_for @a
// CHECK-NEXT:      aie.dma_bd({{.*}} offset = 12276 len
// CHECK-NEXT:      aie.end
// CHECK-NEXT:    }{{$}}
// CHECK-NEXT:    aiex.dma_start_task(%[[A4]])
// CHECK-NEXT:    aiex.dma_await_task(%[[C4]])
// CHECK-NEXT:    }
module {
  aie.device(npu2_1col) {
    %t = aie.tile(0, 0)
    aie.shim_dma_allocation @a (%t, MM2S, 0)
    aie.shim_dma_allocation @c (%t, S2MM, 0)
    aie.runtime_sequence @both_split(%in: memref<32768xi32>, %out: memref<32768xi32>) {
      %c = aiex.dma_configure_task_for @c {
        aie.dma_bd(%out : memref<32768xi32> offset = 0 len = 8198 sizes = [1, 1, 4099, 2] strides = [0, 0, 3, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c)
      %a = aiex.dma_configure_task_for @a {
        aie.dma_bd(%in : memref<32768xi32> offset = 0 len = 8198 sizes = [1, 1, 4099, 2] strides = [0, 0, 3, 1])
        aie.end
      }
      aiex.dma_start_task(%a)
      aiex.dma_await_task(%c)
    }
  }
}

// -----

// A split transfer issued before a whole one: the whole one moves up to right
// after the first slice. Slices only ever move later, so it is the slices
// that follow it, not it that moves ahead of them.
// CHECK-LABEL: @split_before_whole
// CHECK:         %[[C0:.*]] = aiex.dma_configure_task_for @c
// CHECK-NEXT:      aie.dma_bd({{.*}} offset = 0 len
// CHECK-NEXT:      aie.end
// CHECK-NEXT:    }{{$}}
// CHECK-NEXT:    aiex.dma_start_task(%[[C0]])
// CHECK-NEXT:    %[[A:.*]] = aiex.dma_configure_task_for @a
// CHECK-NEXT:      aie.dma_bd({{.*}} offset = 0 len
// CHECK-NEXT:      aie.end
// CHECK-NEXT:    }{{$}}
// CHECK-NEXT:    aiex.dma_start_task(%[[A]])
// CHECK-NEXT:    %[[C1:.*]] = aiex.dma_configure_task_for @c
// CHECK-NEXT:      aie.dma_bd({{.*}} offset = 3069 len
// CHECK-NEXT:      aie.end
// CHECK-NEXT:    }{{$}}
// CHECK-NEXT:    aiex.dma_start_task(%[[C1]])
// CHECK-NEXT:    %[[C2:.*]] = aiex.dma_configure_task_for @c
// CHECK-NEXT:      aie.dma_bd({{.*}} offset = 6138 len
// CHECK-NEXT:      aie.end
// CHECK-NEXT:    }{{$}}
// CHECK-NEXT:    aiex.dma_start_task(%[[C2]])
// CHECK-NEXT:    %[[C3:.*]] = aiex.dma_configure_task_for @c
// CHECK-NEXT:      aie.dma_bd({{.*}} offset = 9207 len
// CHECK-NEXT:      aie.end
// CHECK-NEXT:    }{{$}}
// CHECK-NEXT:    aiex.dma_start_task(%[[C3]])
// CHECK-NEXT:    %[[C4:.*]] = aiex.dma_configure_task_for @c
// CHECK-NEXT:      aie.dma_bd({{.*}} offset = 12276 len
// CHECK-NEXT:      aie.end
// CHECK-NEXT:    } {issue_token = true}
// CHECK-NEXT:    aiex.dma_start_task(%[[C4]])
// CHECK-NEXT:    aiex.dma_await_task(%[[C4]])
// CHECK-NEXT:    }
module {
  aie.device(npu2_1col) {
    %t = aie.tile(0, 0)
    aie.shim_dma_allocation @a (%t, MM2S, 0)
    aie.shim_dma_allocation @c (%t, S2MM, 0)
    aie.runtime_sequence @split_before_whole(%in: memref<32768xi32>, %out: memref<32768xi32>) {
      %c = aiex.dma_configure_task_for @c {
        aie.dma_bd(%out : memref<32768xi32> offset = 0 len = 8198 sizes = [1, 1, 4099, 2] strides = [0, 0, 3, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c)
      %a = aiex.dma_configure_task_for @a {
        aie.dma_bd(%in : memref<32768xi32> offset = 0 len = 64 sizes = [1, 1, 1, 64] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%a)
      aiex.dma_await_task(%c)
    }
  }
}

// -----

// A runtime parameter write between a split transfer and its counterpart does
// not end the round: the counterpart still moves up to right after the first
// slice, past the write, and the write stays ahead of it.
// CHECK-LABEL: @past_parameters
// CHECK:         %[[A0:.*]] = aiex.dma_configure_task_for @a
// CHECK-NEXT:      aie.dma_bd({{.*}} offset = 0 len
// CHECK-NEXT:      aie.end
// CHECK-NEXT:    }{{$}}
// CHECK-NEXT:    aiex.dma_start_task(%[[A0]])
// CHECK:         aiex.npu.rtp_write(@rtp, 0, %{{.*}})
// CHECK-NEXT:    %[[C:.*]] = aiex.dma_configure_task_for @c
// CHECK-NEXT:      aie.dma_bd({{.*}} offset = 0 len
// CHECK-NEXT:      aie.end
// CHECK-NEXT:    } {issue_token = true}
// CHECK-NEXT:    aiex.dma_start_task(%[[C]])
// CHECK-NEXT:    %[[A1:.*]] = aiex.dma_configure_task_for @a
// CHECK-NEXT:      aie.dma_bd({{.*}} offset = 3069 len
// CHECK-COUNT-4: aiex.dma_start_task
// CHECK-NEXT:    aiex.dma_await_task(%[[C]])
module {
  aie.device(npu2_1col) {
    %t = aie.tile(0, 0)
    %core = aie.tile(0, 2)
    %rtp = aie.buffer(%core) {sym_name = "rtp", address = 1024 : i32} : memref<4xi32>
    aie.shim_dma_allocation @a (%t, MM2S, 0)
    aie.shim_dma_allocation @c (%t, S2MM, 0)
    aie.runtime_sequence @past_parameters(%in: memref<32768xi32>, %out: memref<32768xi32>) {
      %a = aiex.dma_configure_task_for @a {
        aie.dma_bd(%in : memref<32768xi32> offset = 0 len = 8198 sizes = [1, 1, 4099, 2] strides = [0, 0, 3, 1])
        aie.end
      }
      aiex.dma_start_task(%a)
      %v = arith.constant 7 : i32
      aiex.npu.rtp_write(@rtp, 0, %v) : i32
      %c = aiex.dma_configure_task_for @c {
        aie.dma_bd(%out : memref<32768xi32> offset = 0 len = 64 sizes = [1, 1, 1, 64] strides = [0, 0, 0, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c)
      aiex.dma_await_task(%c)
    }
  }
}

// -----

// Transfers of 6 and 8 slices merge by fraction done: after the first slices,
// a1 (1/8) c1 (1/6) a2 (2/8) c2 (2/6) a3 (3/8), then c3 and a4 tie at 1/2 and
// keep program order, then a5 (5/8) c4 (4/6) a6 (6/8) c5 (5/6) a7 (7/8).
// CHECK-LABEL: @unequal
// CHECK:         %[[C0:.*]] = aiex.dma_configure_task_for @c
// CHECK-NEXT:      aie.dma_bd({{.*}} offset = 0 len
// CHECK-NEXT:      aie.end
// CHECK-NEXT:    }{{$}}
// CHECK-NEXT:    aiex.dma_start_task(%[[C0]])
// CHECK-NEXT:    %[[A0:.*]] = aiex.dma_configure_task_for @a
// CHECK-NEXT:      aie.dma_bd({{.*}} offset = 0 len
// CHECK-NEXT:      aie.end
// CHECK-NEXT:    }{{$}}
// CHECK-NEXT:    aiex.dma_start_task(%[[A0]])
// CHECK-NEXT:    %[[A1:.*]] = aiex.dma_configure_task_for @a
// CHECK-NEXT:      aie.dma_bd({{.*}} offset = 3069 len
// CHECK-NEXT:      aie.end
// CHECK-NEXT:    }{{$}}
// CHECK-NEXT:    aiex.dma_start_task(%[[A1]])
// CHECK-NEXT:    %[[C1:.*]] = aiex.dma_configure_task_for @c
// CHECK-NEXT:      aie.dma_bd({{.*}} offset = 3069 len
// CHECK-NEXT:      aie.end
// CHECK-NEXT:    }{{$}}
// CHECK-NEXT:    aiex.dma_start_task(%[[C1]])
// CHECK-NEXT:    %[[A2:.*]] = aiex.dma_configure_task_for @a
// CHECK-NEXT:      aie.dma_bd({{.*}} offset = 6138 len
// CHECK-NEXT:      aie.end
// CHECK-NEXT:    }{{$}}
// CHECK-NEXT:    aiex.dma_start_task(%[[A2]])
// CHECK-NEXT:    %[[C2:.*]] = aiex.dma_configure_task_for @c
// CHECK-NEXT:      aie.dma_bd({{.*}} offset = 6138 len
// CHECK-NEXT:      aie.end
// CHECK-NEXT:    }{{$}}
// CHECK-NEXT:    aiex.dma_start_task(%[[C2]])
// CHECK-NEXT:    %[[A3:.*]] = aiex.dma_configure_task_for @a
// CHECK-NEXT:      aie.dma_bd({{.*}} offset = 9207 len
// CHECK-NEXT:      aie.end
// CHECK-NEXT:    }{{$}}
// CHECK-NEXT:    aiex.dma_start_task(%[[A3]])
// CHECK-NEXT:    %[[C3:.*]] = aiex.dma_configure_task_for @c
// CHECK-NEXT:      aie.dma_bd({{.*}} offset = 9207 len
// CHECK-NEXT:      aie.end
// CHECK-NEXT:    }{{$}}
// CHECK-NEXT:    aiex.dma_start_task(%[[C3]])
// CHECK-NEXT:    %[[A4:.*]] = aiex.dma_configure_task_for @a
// CHECK-NEXT:      aie.dma_bd({{.*}} offset = 12276 len
// CHECK-NEXT:      aie.end
// CHECK-NEXT:    }{{$}}
// CHECK-NEXT:    aiex.dma_start_task(%[[A4]])
// CHECK-NEXT:    %[[A5:.*]] = aiex.dma_configure_task_for @a
// CHECK-NEXT:      aie.dma_bd({{.*}} offset = 15345 len
// CHECK-NEXT:      aie.end
// CHECK-NEXT:    }{{$}}
// CHECK-NEXT:    aiex.dma_start_task(%[[A5]])
// CHECK-NEXT:    %[[C4:.*]] = aiex.dma_configure_task_for @c
// CHECK-NEXT:      aie.dma_bd({{.*}} offset = 12276 len
// CHECK-NEXT:      aie.end
// CHECK-NEXT:    }{{$}}
// CHECK-NEXT:    aiex.dma_start_task(%[[C4]])
// CHECK-NEXT:    %[[A6:.*]] = aiex.dma_configure_task_for @a
// CHECK-NEXT:      aie.dma_bd({{.*}} offset = 18414 len
// CHECK-NEXT:      aie.end
// CHECK-NEXT:    }{{$}}
// CHECK-NEXT:    aiex.dma_start_task(%[[A6]])
// CHECK-NEXT:    %[[C5:.*]] = aiex.dma_configure_task_for @c
// CHECK-NEXT:      aie.dma_bd({{.*}} offset = 15345 len
// CHECK-NEXT:      aie.end
// CHECK-NEXT:    } {issue_token = true}
// CHECK-NEXT:    aiex.dma_start_task(%[[C5]])
// CHECK-NEXT:    %[[A7:.*]] = aiex.dma_configure_task_for @a
// CHECK-NEXT:      aie.dma_bd({{.*}} offset = 21483 len
// CHECK-NEXT:      aie.end
// CHECK-NEXT:    }{{$}}
// CHECK-NEXT:    aiex.dma_start_task(%[[A7]])
// CHECK-NEXT:    aiex.dma_await_task(%[[C5]])
// CHECK-NEXT:    }
module {
  aie.device(npu2_1col) {
    %t = aie.tile(0, 0)
    aie.shim_dma_allocation @a (%t, MM2S, 0)
    aie.shim_dma_allocation @c (%t, S2MM, 0)
    aie.runtime_sequence @unequal(%in: memref<32768xi32>, %out: memref<32768xi32>) {
      %c = aiex.dma_configure_task_for @c {
        aie.dma_bd(%out : memref<32768xi32> offset = 0 len = 10238 sizes = [1, 1, 5119, 2] strides = [0, 0, 3, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c)
      %a = aiex.dma_configure_task_for @a {
        aie.dma_bd(%in : memref<32768xi32> offset = 0 len = 16358 sizes = [1, 1, 8179, 2] strides = [0, 0, 3, 1])
        aie.end
      }
      aiex.dma_start_task(%a)
      aiex.dma_await_task(%c)
    }
  }
}

// -----

// Slices do not move past a later start on a channel their transfer shares:
// the second transfer on @a ends the round, so the first's slices all go out
// before it. Nor do they move past an await, a sync, or a poll: the first
// transfer on @c is placed whole before the await, and the second one after it.
// CHECK-LABEL: @rounds
// CHECK:         %[[A0:.*]] = aiex.dma_configure_task_for @a
// CHECK-NEXT:      aie.dma_bd({{.*}} offset = 0 len
// CHECK-NEXT:      aie.end
// CHECK-NEXT:    }{{$}}
// CHECK-NEXT:    aiex.dma_start_task(%[[A0]])
// CHECK-NEXT:    %[[A1:.*]] = aiex.dma_configure_task_for @a
// CHECK-NEXT:      aie.dma_bd({{.*}} offset = 3069 len
// CHECK-NEXT:      aie.end
// CHECK-NEXT:    }{{$}}
// CHECK-NEXT:    aiex.dma_start_task(%[[A1]])
// CHECK-NEXT:    %[[A2:.*]] = aiex.dma_configure_task_for @a
// CHECK-NEXT:      aie.dma_bd({{.*}} offset = 6138 len
// CHECK-NEXT:      aie.end
// CHECK-NEXT:    }{{$}}
// CHECK-NEXT:    aiex.dma_start_task(%[[A2]])
// CHECK-NEXT:    %[[A3:.*]] = aiex.dma_configure_task_for @a
// CHECK-NEXT:      aie.dma_bd({{.*}} offset = 9207 len
// CHECK-NEXT:      aie.end
// CHECK-NEXT:    }{{$}}
// CHECK-NEXT:    aiex.dma_start_task(%[[A3]])
// CHECK-NEXT:    %[[A4:.*]] = aiex.dma_configure_task_for @a
// CHECK-NEXT:      aie.dma_bd({{.*}} offset = 12276 len
// CHECK-NEXT:      aie.end
// CHECK-NEXT:    }{{$}}
// CHECK-NEXT:    aiex.dma_start_task(%[[A4]])
// CHECK-NEXT:    %[[B:.*]] = aiex.dma_configure_task_for @a
// CHECK-NEXT:      aie.dma_bd({{.*}} offset = 0 len
// CHECK-NEXT:      aie.end
// CHECK-NEXT:    }{{$}}
// CHECK-NEXT:    aiex.dma_start_task(%[[B]])
// CHECK-NEXT:    %[[C0:.*]] = aiex.dma_configure_task_for @c
// CHECK-NEXT:      aie.dma_bd({{.*}} offset = 0 len
// CHECK-NEXT:      aie.end
// CHECK-NEXT:    }{{$}}
// CHECK-NEXT:    aiex.dma_start_task(%[[C0]])
// CHECK-NEXT:    %[[C1:.*]] = aiex.dma_configure_task_for @c
// CHECK-NEXT:      aie.dma_bd({{.*}} offset = 3069 len
// CHECK-NEXT:      aie.end
// CHECK-NEXT:    }{{$}}
// CHECK-NEXT:    aiex.dma_start_task(%[[C1]])
// CHECK-NEXT:    %[[C2:.*]] = aiex.dma_configure_task_for @c
// CHECK-NEXT:      aie.dma_bd({{.*}} offset = 6138 len
// CHECK-NEXT:      aie.end
// CHECK-NEXT:    }{{$}}
// CHECK-NEXT:    aiex.dma_start_task(%[[C2]])
// CHECK-NEXT:    %[[C3:.*]] = aiex.dma_configure_task_for @c
// CHECK-NEXT:      aie.dma_bd({{.*}} offset = 9207 len
// CHECK-NEXT:      aie.end
// CHECK-NEXT:    }{{$}}
// CHECK-NEXT:    aiex.dma_start_task(%[[C3]])
// CHECK-NEXT:    %[[C4:.*]] = aiex.dma_configure_task_for @c
// CHECK-NEXT:      aie.dma_bd({{.*}} offset = 12276 len
// CHECK-NEXT:      aie.end
// CHECK-NEXT:    } {issue_token = true}
// CHECK-NEXT:    aiex.dma_start_task(%[[C4]])
// CHECK-NEXT:    aiex.dma_await_task(%[[C4]])
// CHECK-NEXT:    %[[D0:.*]] = aiex.dma_configure_task_for @c
// CHECK-NEXT:      aie.dma_bd({{.*}} offset = 0 len
// CHECK-NEXT:      aie.end
// CHECK-NEXT:    }{{$}}
// CHECK-NEXT:    aiex.dma_start_task(%[[D0]])
// CHECK-NEXT:    %[[D1:.*]] = aiex.dma_configure_task_for @c
// CHECK-NEXT:      aie.dma_bd({{.*}} offset = 3069 len
// CHECK-NEXT:      aie.end
// CHECK-NEXT:    }{{$}}
// CHECK-NEXT:    aiex.dma_start_task(%[[D1]])
// CHECK-NEXT:    %[[D2:.*]] = aiex.dma_configure_task_for @c
// CHECK-NEXT:      aie.dma_bd({{.*}} offset = 6138 len
// CHECK-NEXT:      aie.end
// CHECK-NEXT:    }{{$}}
// CHECK-NEXT:    aiex.dma_start_task(%[[D2]])
// CHECK-NEXT:    %[[D3:.*]] = aiex.dma_configure_task_for @c
// CHECK-NEXT:      aie.dma_bd({{.*}} offset = 9207 len
// CHECK-NEXT:      aie.end
// CHECK-NEXT:    }{{$}}
// CHECK-NEXT:    aiex.dma_start_task(%[[D3]])
// CHECK-NEXT:    %[[D4:.*]] = aiex.dma_configure_task_for @c
// CHECK-NEXT:      aie.dma_bd({{.*}} offset = 12276 len
// CHECK-NEXT:      aie.end
// CHECK-NEXT:    }{{$}}
// CHECK-NEXT:    aiex.dma_start_task(%[[D4]])
// CHECK-NEXT:    }
module {
  aie.device(npu2_1col) {
    %t = aie.tile(0, 0)
    aie.shim_dma_allocation @a (%t, MM2S, 0)
    aie.shim_dma_allocation @c (%t, S2MM, 0)
    aie.runtime_sequence @rounds(%in: memref<32768xi32>, %out: memref<32768xi32>) {
      %a = aiex.dma_configure_task_for @a {
        aie.dma_bd(%in : memref<32768xi32> offset = 0 len = 8198 sizes = [1, 1, 4099, 2] strides = [0, 0, 3, 1])
        aie.end
      }
      aiex.dma_start_task(%a)
      %b = aiex.dma_configure_task_for @a {
        aie.dma_bd(%in : memref<32768xi32> offset = 0 len = 64 sizes = [1, 1, 1, 64] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%b)
      %c = aiex.dma_configure_task_for @c {
        aie.dma_bd(%out : memref<32768xi32> offset = 0 len = 8198 sizes = [1, 1, 4099, 2] strides = [0, 0, 3, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c)
      aiex.dma_await_task(%c)
      %d = aiex.dma_configure_task_for @c {
        aie.dma_bd(%out : memref<32768xi32> offset = 0 len = 8198 sizes = [1, 1, 4099, 2] strides = [0, 0, 3, 1])
        aie.end
      }
      aiex.dma_start_task(%d)
    }
  }
}

// -----

// A start that runs the task twice runs every slice twice, one pass after the
// other. Only the last push of the last slice issues the token: an earlier one
// would let the await return a pass early.
// CHECK-LABEL: @two_passes
// CHECK:         %[[A0:.*]] = aiex.dma_configure_task_for @a
// CHECK-NEXT:      aie.dma_bd({{.*}} offset = 0 len
// CHECK-NEXT:      aie.end
// CHECK-NEXT:    }{{$}}
// CHECK-NEXT:    aiex.dma_start_task(%[[A0]]){{$}}
// CHECK-NEXT:    %[[A1:.*]] = aiex.dma_configure_task_for @a
// CHECK-NEXT:      aie.dma_bd({{.*}} offset = 3069 len
// CHECK-NEXT:      aie.end
// CHECK-NEXT:    }{{$}}
// CHECK-NEXT:    aiex.dma_start_task(%[[A1]]){{$}}
// CHECK-NEXT:    %[[A2:.*]] = aiex.dma_configure_task_for @a
// CHECK-NEXT:      aie.dma_bd({{.*}} offset = 6138 len
// CHECK-NEXT:      aie.end
// CHECK-NEXT:    }{{$}}
// CHECK-NEXT:    aiex.dma_start_task(%[[A2]]){{$}}
// CHECK-NEXT:    %[[A3:.*]] = aiex.dma_configure_task_for @a
// CHECK-NEXT:      aie.dma_bd({{.*}} offset = 9207 len
// CHECK-NEXT:      aie.end
// CHECK-NEXT:    }{{$}}
// CHECK-NEXT:    aiex.dma_start_task(%[[A3]]){{$}}
// CHECK-NEXT:    %[[A4:.*]] = aiex.dma_configure_task_for @a
// CHECK-NEXT:      aie.dma_bd({{.*}} offset = 12276 len
// CHECK-NEXT:      aie.end
// CHECK-NEXT:    } {issue_token = true}
// CHECK-NEXT:    aiex.dma_start_task(%[[A4]]) {no_token}
// CHECK-NEXT:    aiex.dma_start_task(%[[A0]]){{$}}
// CHECK-NEXT:    aiex.dma_start_task(%[[A1]]){{$}}
// CHECK-NEXT:    aiex.dma_start_task(%[[A2]]){{$}}
// CHECK-NEXT:    aiex.dma_start_task(%[[A3]]){{$}}
// CHECK-NEXT:    aiex.dma_start_task(%[[A4]]){{$}}
// CHECK-NEXT:    aiex.dma_await_task(%[[A4]])
module {
  aie.device(npu2_1col) {
    %t = aie.tile(0, 0)
    aie.shim_dma_allocation @a (%t, MM2S, 0)
    aie.runtime_sequence @two_passes(%in: memref<32768xi32>) {
      %a = aiex.dma_configure_task_for @a {
        aie.dma_bd(%in : memref<32768xi32> offset = 0 len = 8198 sizes = [1, 1, 4099, 2] strides = [0, 0, 3, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%a) {repeat_count = 1 : i32}
      aiex.dma_await_task(%a)
    }
  }
}

// -----

// A stride past the shim's step field is sliced one iteration at a time, so
// no slice runs the task's four iterations: each gets a task, and a repeat
// count, of its own. Four slices would fit in a chain, but a chain's members
// share its repeat count.
// CHECK-LABEL: @own_repeat
// CHECK:         %[[A0:.*]] = aiex.dma_configure_task_for @a
// CHECK-NEXT:      aie.dma_bd({{.*}} offset = 0 len = 32768 sizes = [1, 1, 64, 512]
// CHECK-NEXT:      aie.end
// CHECK-NEXT:    }{{$}}
// CHECK-NEXT:    aiex.dma_start_task(%[[A0]]){{$}}
// CHECK-NEXT:    %[[A1:.*]] = aiex.dma_configure_task_for @a
// CHECK-NEXT:      aie.dma_bd({{.*}} offset = 4194304 len = 32768 sizes = [1, 1, 64, 512]
// CHECK-NEXT:      aie.end
// CHECK-NEXT:    }{{$}}
// CHECK-NEXT:    aiex.dma_start_task(%[[A1]]){{$}}
// CHECK-NEXT:    %[[A2:.*]] = aiex.dma_configure_task_for @a
// CHECK-NEXT:      aie.dma_bd({{.*}} offset = 8388608 len = 32768 sizes = [1, 1, 64, 512]
// CHECK-NEXT:      aie.end
// CHECK-NEXT:    }{{$}}
// CHECK-NEXT:    aiex.dma_start_task(%[[A2]]){{$}}
// CHECK-NEXT:    %[[A3:.*]] = aiex.dma_configure_task_for @a
// CHECK-NEXT:      aie.dma_bd({{.*}} offset = 12582912 len = 32768 sizes = [1, 1, 64, 512]
// CHECK-NEXT:      aie.end
// CHECK-NEXT:    } {issue_token = true}
// CHECK-NEXT:    aiex.dma_start_task(%[[A3]]){{$}}
// CHECK-NEXT:    aiex.dma_await_task(%[[A3]])
module {
  aie.device(npu2_1col) {
    %t = aie.tile(0, 0)
    aie.shim_dma_allocation @a (%t, MM2S, 0)
    aie.runtime_sequence @own_repeat(%in: memref<16777216xbf16>) {
      %a = aiex.dma_configure_task_for @a {
        aie.dma_bd(%in : memref<16777216xbf16> offset = 0 len = 32768 sizes = [4, 1, 64, 512] strides = [4194304, 0, 8192, 1])
        aie.end
      } {repeat_count = 3 : i32, issue_token = true}
      aiex.dma_start_task(%a)
      aiex.dma_await_task(%a)
    }
  }
}

// -----

// A runtime repeat count cannot be divided into passes, so past the queue
// depth the transfer stays one chain while it fits the tile's descriptors.
// CHECK-LABEL: @runtime_repeat_chain
// CHECK:         %[[A:.*]] = aiex.dma_configure_task_for @a repeat
// CHECK-COUNT-4:   aie.next_bd
// CHECK:         aiex.dma_start_task(%[[A]])
// CHECK-NEXT:    aiex.dma_await_task(%[[A]])
module {
  aie.device(npu2_1col) {
    %t = aie.tile(0, 0)
    aie.shim_dma_allocation @a (%t, MM2S, 0)
    aie.runtime_sequence @runtime_repeat_chain(%in: memref<32768xi32>, %r: i32) {
      %a = aiex.dma_configure_task_for @a repeat %r : i32 {
        aie.dma_bd(%in : memref<32768xi32> offset = 0 len = 8198 sizes = [1, 1, 4099, 2] strides = [0, 0, 3, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%a)
      aiex.dma_await_task(%a)
    }
  }
}

// -----

// A contiguous transfer lowers as a plain length, so only its iteration count
// is limited (to 64 on npu2). 80 iterations are sliced into 64 and 16, each
// its own task with a repeat count running one pass over its slice.
// CHECK-LABEL: @slice_iterations
// CHECK:         %[[A0:.*]] = aiex.dma_configure_task_for @a
// CHECK-NEXT:      aie.dma_bd({{.*}} offset = 0 len = 256 sizes = [64, 1, 1, 256] strides = [1000, 0, 0, 1])
// CHECK-NEXT:      aie.end
// CHECK-NEXT:    } {repeat_count = 63 : i32}
// CHECK-NEXT:    aiex.dma_start_task(%[[A0]]){{$}}
// CHECK-NEXT:    %[[A1:.*]] = aiex.dma_configure_task_for @a
// CHECK-NEXT:      aie.dma_bd({{.*}} offset = 64000 len = 256 sizes = [16, 1, 1, 256] strides = [1000, 0, 0, 1])
// CHECK-NEXT:      aie.end
// CHECK-NEXT:    } {issue_token = true, repeat_count = 15 : i32}
// CHECK-NEXT:    aiex.dma_start_task(%[[A1]]){{$}}
// CHECK-NEXT:    aiex.dma_await_task(%[[A1]])
module {
  aie.device(npu2_1col) {
    %t = aie.tile(0, 0)
    aie.shim_dma_allocation @a (%t, MM2S, 0)
    aie.runtime_sequence @slice_iterations(%in: memref<81920xi32>) {
      %a = aiex.dma_configure_task_for @a {
        aie.dma_bd(%in : memref<81920xi32> offset = 0 len = 256 sizes = [80, 1, 1, 256] strides = [1000, 0, 0, 1])
        aie.end
      } {repeat_count = 79 : i32, issue_token = true}
      aiex.dma_start_task(%a)
      aiex.dma_await_task(%a)
    }
  }
}

// -----

// An iteration dimension of stride 0 repeats the same data, as the repeat
// count does. With it dropped, the descriptor fits without slicing, and each
// of its 80 executions still moves the same 81920 elements.
// CHECK-LABEL: @drop_repeat_dim
// CHECK:         %[[A:.*]] = aiex.dma_configure_task_for @a
// CHECK-NEXT:      aie.dma_bd({{.*}} offset = 0 len = 81920 sizes = [1, 20, 1, 4096] strides = [0, 4096, 0, 1])
// CHECK-NEXT:      aie.end
// CHECK-NEXT:    } {issue_token = true, repeat_count = 79 : i32}
// CHECK-NEXT:    aiex.dma_start_task(%[[A]]){{$}}
// CHECK-NEXT:    aiex.dma_await_task(%[[A]])
module {
  aie.device(npu2_1col) {
    %t = aie.tile(0, 0)
    aie.shim_dma_allocation @a (%t, MM2S, 0)
    aie.runtime_sequence @drop_repeat_dim(%in: memref<81920xi32>) {
      %a = aiex.dma_configure_task_for @a {
        aie.dma_bd(%in : memref<81920xi32> offset = 0 len = 81920 sizes = [80, 20, 1, 4096] strides = [0, 4096, 0, 1])
        aie.end
      } {repeat_count = 79 : i32, issue_token = true}
      aiex.dma_start_task(%a)
      aiex.dma_await_task(%a)
    }
  }
}

// -----

// Eighteen slices each of a fill and a drain, alternating, through one shim
// tile's 16 descriptors. As one chain each they could never have fit. As
// separate tasks, each configured only when it is due, they take ids 0 to 15,
// and then aie-assign-runtime-sequence-bd-ids hands out ids of slices that the
// queue-space polls it emits prove finished.
// LOWERED-LABEL: @more_slices_than_bds
// LOWERED:         aie.dma_bd({{.*}} offset = 0 len = 2046 {{.*}}{bd_id = 0 : i32
// LOWERED:         aie.dma_bd({{.*}} offset = 21483 len = 2046 {{.*}}{bd_id = 15 : i32
// LOWERED:         aiex.npu.maskpoll
// LOWERED-NEXT:    aiex.dma_start_task
// LOWERED-NEXT:    aiex.dma_configure_task(
// LOWERED-NEXT:      aie.dma_bd({{.*}} offset = 24552 len = 2046 {{.*}}{bd_id = 0 : i32
// LOWERED:         aie.dma_bd({{.*}} offset = 52173 len = 4 {{.*}}{bd_id =
// LOWERED:         aie.dma_bd({{.*}} offset = 52173 len = 4 {{.*}}{bd_id =
// LOWERED:         aiex.dma_start_task
// LOWERED-NEXT:    aiex.dma_await_task
module {
  aie.device(npu2_1col) {
    %t = aie.tile(0, 0)
    aie.shim_dma_allocation @a (%t, MM2S, 0)
    aie.shim_dma_allocation @c (%t, S2MM, 0)
    aie.runtime_sequence @more_slices_than_bds(%in: memref<65536xi32>, %out: memref<65536xi32>) {
      %a = aiex.dma_configure_task_for @a {
        aie.dma_bd(%in : memref<65536xi32> offset = 0 len = 34786 sizes = [1, 1, 17393, 2] strides = [0, 0, 3, 1])
        aie.end
      }
      aiex.dma_start_task(%a)
      %c = aiex.dma_configure_task_for @c {
        aie.dma_bd(%out : memref<65536xi32> offset = 0 len = 34786 sizes = [1, 1, 17393, 2] strides = [0, 0, 3, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c)
      aiex.dma_await_task(%c)
    }
  }
}
