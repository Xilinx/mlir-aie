//===- decompose_large_dma_task_invalid.mlir ------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// aie-decompose-large-dma-bd rejects the task-path descriptors it cannot decompose.

// RUN: aie-opt --pass-pipeline='any(aie.device(aie-decompose-large-dma-bd))' \
// RUN:   --split-input-file --verify-diagnostics %s

module {
  aie.device(npu2_1col) {
    %t = aie.tile(0, 0)
    aie.shim_dma_allocation @a (%t, MM2S, 0)
    aie.runtime_sequence @ooo_too_large(%in: memref<4096xi32>) {
      %tk = aiex.dma_configure_task_for @a {
        // expected-error@+1 {{splitting an out-of-order buffer descriptor into multiple descriptors is not implemented}}
        aie.dma_bd(%in : memref<4096xi32> offset = 0 len = 2062 sizes = [1, 1, 1031, 2] strides = [0, 0, 3, 1])
          {packet = #aie.packet_info<pkt_type = 0, pkt_id = 1>, out_of_order_id = 5 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%tk)
      aiex.dma_await_task(%tk)
    }
  }
}

// -----

// Decomposition that moves extent into the iteration dimension has to scale the
// task's repeat count to match. A runtime repeat count cannot be scaled at
// compile time, so the BD is rejected rather than left under-running.

module {
  aie.device(npu2_1col) {
    %t = aie.tile(0, 0)
    aie.shim_dma_allocation @a (%t, MM2S, 0)
    aie.runtime_sequence @runtime_repeat(%in: memref<16x16x4096xi8>, %r: i32) {
      %tk = aiex.dma_configure_task_for @a repeat %r : i32 {
        // expected-error@+1 {{cannot decompose a buffer descriptor whose repeat count is a runtime value: decomposition needs to scale it by 8}}
        aie.dma_bd(%in : memref<16x16x4096xi8> offset = 4096 len = 262144 sizes = [1, 8, 8, 4096] strides = [0, 131072, 8192, 1])
          {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%tk)
    }
  }
}

// -----

// A scaled repeat count past the queue's 8-bit field is split into several
// pushes later (dma_task_repeat_split.mlir), but it still has to fit the 32-bit
// attribute. Saying so here names the factor that got us there.

module {
  aie.device(npu2_1col) {
    %t = aie.tile(0, 0)
    aie.shim_dma_allocation @a (%t, MM2S, 0)
    aie.runtime_sequence @repeat_overflows(%in: memref<16x16x4096xi8>) {
      %tk = aiex.dma_configure_task_for @a {
        // expected-error@+1 {{decomposition scales the repeat count by 8 to 17179869183, beyond a 32-bit repeat_count}}
        aie.dma_bd(%in : memref<16x16x4096xi8> offset = 4096 len = 262144 sizes = [1, 8, 8, 4096] strides = [0, 131072, 8192, 1])
          {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 2147483647 : i32}
      aiex.dma_start_task(%tk)
    }
  }
}

// -----

// Sliced one iteration at a time, each slice needs a task of its own, and a
// start then runs every slice once per pass. Six runs of a four-iteration
// transfer end partway through a pass, where no slice boundary falls.

module {
  aie.device(npu2_1col) {
    %t = aie.tile(0, 0)
    aie.shim_dma_allocation @a (%t, MM2S, 0)
    aie.runtime_sequence @partial_pass(%in: memref<16777216xbf16>) {
      %tk = aiex.dma_configure_task_for @a {
        // expected-error@+1 {{cannot split this buffer descriptor: its slices need per-descriptor repeat counts, which a chain shares, and they cannot be separate tasks because a start runs it 6 times, not a whole number of passes over its 4-long iteration dimension}}
        aie.dma_bd(%in : memref<16777216xbf16> offset = 0 len = 32768 sizes = [4, 1, 64, 512] strides = [4194304, 0, 8192, 1])
        aie.end
      } {repeat_count = 3 : i32, issue_token = true}
      aiex.dma_start_task(%tk) {repeat_count = 5 : i32}
      aiex.dma_await_task(%tk)
    }
  }
}

// -----

// A runtime repeat count cannot be divided into passes, so the slices have to
// stay one chain, and 18 of them do not fit the shim tile's 16 descriptors.

module {
  aie.device(npu2_1col) {
    %t = aie.tile(0, 0)
    aie.shim_dma_allocation @a (%t, MM2S, 0)
    aie.runtime_sequence @runtime_repeat_too_many(%in: memref<65536xi32>, %r: i32) {
      %tk = aiex.dma_configure_task_for @a repeat %r : i32 {
        // expected-error@+1 {{cannot split this buffer descriptor: its 18 slices outnumber the tile's 16 buffer descriptors, and they cannot be separate tasks because the task's repeat count is a runtime value}}
        aie.dma_bd(%in : memref<65536xi32> offset = 0 len = 34786 sizes = [1, 1, 17393, 2] strides = [0, 0, 3, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%tk)
      aiex.dma_await_task(%tk)
    }
  }
}

// -----

// Dimensions past a descriptor's 4 are split off only where their offsets can
// be computed: a runtime offset or length leaves them where they are.

module {
  aie.device(npu2_1col) {
    %t = aie.tile(0, 0)
    aie.shim_dma_allocation @a (%t, MM2S, 0)
    aie.runtime_sequence @nd_runtime_offset(%in: memref<65536xi32>, %off: i32) {
      %c128 = arith.constant 128 : i32
      %tk = aiex.dma_configure_task_for @a {
        // expected-error@+1 {{has 5 dimensions, and a buffer descriptor holds 4; the extra ones can only be split off a descriptor whose offset, length, sizes and strides are all constant and that has no padding}}
        aie.dma_bd(%in : memref<65536xi32> offset = %off len = %c128 sizes = [2, 2, 1, 8, 16] strides = [9000, 3500, 0, 32, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%tk)
      aiex.dma_await_task(%tk)
    }
  }
}

// -----

// Pieces of one descriptor in a chain would need the rest of the chain split
// with them.

module {
  aie.device(npu2_1col) {
    %t = aie.tile(0, 0)
    aie.shim_dma_allocation @a (%t, MM2S, 0)
    aie.runtime_sequence @nd_in_chain(%in: memref<65536xi32>) {
      %tk = aiex.dma_configure_task_for @a {
        // expected-error@+1 {{has 5 dimensions, and a buffer descriptor holds 4; the extra ones can only be split off a task's only descriptor}}
        aie.dma_bd(%in : memref<65536xi32> offset = 0 len = 256 sizes = [2, 2, 2, 8, 16] strides = [9000, 3500, 256, 32, 1])
        aie.next_bd ^bd1
      ^bd1:
        aie.dma_bd(%in : memref<65536xi32> offset = 0 len = 128)
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%tk)
      aiex.dma_await_task(%tk)
    }
  }
}

// -----

// Under runtime control flow, a descriptor has to stay one: iteration
// dimensions that merge are accepted, ones that split into pieces are not.

module {
  aie.device(npu2_1col) {
    %t = aie.tile(0, 0)
    aie.shim_dma_allocation @a (%t, MM2S, 0)
    aie.runtime_sequence @nd_in_loop(%in: memref<65536xi32>) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      scf.for %i = %c0 to %c4 step %c1 {
        %merged = aiex.dma_configure_task_for @a {
          aie.dma_bd(%in : memref<65536xi32> offset = 0 len = 256 sizes = [2, 3, 2, 8, 16] strides = [300, 100, 1000, 32, 1])
          aie.end
        } {issue_token = true, repeat_count = 5 : i32}
        aiex.dma_start_task(%merged)
        aiex.dma_await_task(%merged)
        %tk = aiex.dma_configure_task_for @a {
          // expected-error@+1 {{has 5 dimensions, and a buffer descriptor holds 4; the extra ones can only be split off outside runtime control flow, since it splits into 2 descriptors}}
          aie.dma_bd(%in : memref<65536xi32> offset = 0 len = 256 sizes = [2, 2, 2, 8, 16] strides = [9000, 3500, 256, 32, 1])
          aie.end
        } {issue_token = true, repeat_count = 3 : i32}
        aiex.dma_start_task(%tk)
        aiex.dma_await_task(%tk)
      }
    }
  }
}

// -----

// Each piece runs its own part of a pass, so the pieces must be separate tasks,
// which a runtime repeat count cannot be divided among.

module {
  aie.device(npu2_1col) {
    %t = aie.tile(0, 0)
    aie.shim_dma_allocation @a (%t, MM2S, 0)
    aie.runtime_sequence @nd_runtime_repeat(%in: memref<65536xi32>, %r: i32) {
      %tk = aiex.dma_configure_task_for @a repeat %r : i32 {
        // expected-error@+1 {{cannot split this buffer descriptor: its slices need per-descriptor repeat counts, which a chain shares, and they cannot be separate tasks because the task's repeat count is a runtime value}}
        aie.dma_bd(%in : memref<65536xi32> offset = 0 len = 256 sizes = [2, 2, 2, 8, 16] strides = [9000, 3500, 256, 32, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%tk)
      aiex.dma_await_task(%tk)
    }
  }
}

// -----

// Nor can a repeat count that stops partway through a pass over [2 x 2].

module {
  aie.device(npu2_1col) {
    %t = aie.tile(0, 0)
    aie.shim_dma_allocation @a (%t, MM2S, 0)
    aie.runtime_sequence @nd_partial_pass(%in: memref<65536xi32>) {
      %tk = aiex.dma_configure_task_for @a {
        // expected-error@+1 {{cannot split this buffer descriptor: its slices need per-descriptor repeat counts, which a chain shares, and they cannot be separate tasks because a start runs it 6 times, not a whole number of passes over its 4-long iteration dimension}}
        aie.dma_bd(%in : memref<65536xi32> offset = 0 len = 256 sizes = [2, 2, 2, 8, 16] strides = [9000, 3500, 256, 32, 1])
        aie.end
      } {issue_token = true, repeat_count = 5 : i32}
      aiex.dma_start_task(%tk)
      aiex.dma_await_task(%tk)
    }
  }
}

// -----

// Dropping the unit dimension merges every 20 executions into one, which a
// task running 30 cannot be expressed in.

module {
  aie.device(npu2_1col) {
    %t = aie.tile(0, 0)
    aie.shim_dma_allocation @a (%t, MM2S, 0)
    aie.runtime_sequence @squeeze_partial(%in: memref<1310720xi32>) {
      %tk = aiex.dma_configure_task_for @a {
        // expected-error@+1 {{cannot decompose: it merges every 20 executions of the buffer descriptor into 1, and the task runs it 30 times}}
        aie.dma_bd(%in : memref<1310720xi32> offset = 0 len = 1024 sizes = [4, 20, 1, 64, 16] strides = [327680, 16, 81920, 1280, 1])
        aie.end
      } {issue_token = true, repeat_count = 29 : i32}
      aiex.dma_start_task(%tk)
      aiex.dma_await_task(%tk)
    }
  }
}

// -----

// A descriptor takes its locks once per execution, which splitting would
// regroup, so a lock-taking descriptor is not reduced.
module {
  aie.device(npu2_1col) {
    %mt = aie.tile(0, 1)
    %buf = aie.buffer(%mt) : memref<4096xi32>
    %lk = aie.lock(%mt) {init = 0 : i32}
    aie.runtime_sequence @nd_takes_locks(%in: memref<4096xi32>) {
      %c1 = arith.constant 1 : i32
      %tk = aiex.dma_configure_task(%mt, MM2S, 0) {
        aie.use_lock(%lk, AcquireGreaterEqual, %c1)
        // expected-error@+1 {{has 5 dimensions, and a buffer descriptor holds 4; the extra ones can only be split off a descriptor that takes no locks}}
        aie.dma_bd(%buf : memref<4096xi32> offset = 0 len = 256 sizes = [2, 2, 2, 8, 16] strides = [2048, 1024, 256, 32, 1])
        aie.use_lock(%lk, Release, %c1)
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%tk)
      aiex.dma_free_task(%tk)
    }
  }
}
