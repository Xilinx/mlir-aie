//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

// Enforcement pinned off: these stanzas are about what the count reports, and
// with it on the compiler inserts a poll instead of saying anything.
// RUN: aie-opt --aie-unroll-runtime-sequence-loops --canonicalize \
// RUN:         --aie-assign-runtime-sequence-bd-ids='enforce-queue-depth=false' \
// RUN:         --verify-diagnostics --split-input-file %s

// A DMA channel's task queue holds only getDmaTaskQueueDepth() entries (4 on
// AIE2, matching aie-rt's XAIE_DMA_MAX_QUEUE_SIZE). Pushing onto a full queue
// drops the push, so the transfer never runs and anything waiting on it hangs.
// Unlike the TCT FIFO, over-production is not safe here.

// Exactly at the limit: 4 pushes with no await fills the queue but never
// overflows it, so this must stay quiet.
aie.device(npu2) {
  %tile_0_0 = aie.tile(0, 0)
  aie.runtime_sequence @at_limit(%arg0: memref<1024xi32>) {
    %t0 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<1024xi32> offset = 0 len = 256)
      aie.end
    }
    aiex.dma_start_task(%t0)
    %t1 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<1024xi32> offset = 256 len = 256)
      aie.end
    }
    aiex.dma_start_task(%t1)
    %t2 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<1024xi32> offset = 512 len = 256)
      aie.end
    }
    aiex.dma_start_task(%t2)
    %t3 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<1024xi32> offset = 768 len = 256)
      aie.end
    }
    aiex.dma_start_task(%t3)
  }
}

// -----

// One past the limit: the 5th push finds a full queue. Only that push is
// flagged; the first four are legal.
aie.device(npu2) {
  %tile_0_0 = aie.tile(0, 0)
  aie.runtime_sequence @overflow(%arg0: memref<1280xi32>) {
    %t0 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<1280xi32> offset = 0 len = 256)
      aie.end
    }
    aiex.dma_start_task(%t0)
    %t1 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<1280xi32> offset = 256 len = 256)
      aie.end
    }
    aiex.dma_start_task(%t1)
    %t2 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<1280xi32> offset = 512 len = 256)
      aie.end
    }
    aiex.dma_start_task(%t2)
    %t3 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<1280xi32> offset = 768 len = 256)
      aie.end
    }
    aiex.dma_start_task(%t3)
    %t4 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<1280xi32> offset = 1024 len = 256)
      aie.end
    }
    // expected-warning@+1 {{whose task queue is only 4 deep, with 4 push(es) not yet known to have completed}}
    aiex.dma_start_task(%t4)
  }
}

// -----

// Same five pushes, but an await after the fourth drains the queue, so the
// fifth lands in an empty one. Draining is what makes the bound satisfiable.
aie.device(npu2) {
  %tile_0_0 = aie.tile(0, 0)
  aie.runtime_sequence @drained_by_await(%arg0: memref<1280xi32>) {
    %t0 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<1280xi32> offset = 0 len = 256)
      aie.end
    } {issue_token = true}
    aiex.dma_start_task(%t0)
    %t1 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<1280xi32> offset = 256 len = 256)
      aie.end
    } {issue_token = true}
    aiex.dma_start_task(%t1)
    %t2 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<1280xi32> offset = 512 len = 256)
      aie.end
    } {issue_token = true}
    aiex.dma_start_task(%t2)
    %t3 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<1280xi32> offset = 768 len = 256)
      aie.end
    } {issue_token = true}
    aiex.dma_start_task(%t3)
    // Four awaits retire all four pushes: each pops the oldest outstanding
    // token, and every push here issues one.
    aiex.dma_await_task(%t0)
    aiex.dma_await_task(%t1)
    aiex.dma_await_task(%t2)
    aiex.dma_await_task(%t3)
    %t4 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<1280xi32> offset = 1024 len = 256)
      aie.end
    } {issue_token = true}
    aiex.dma_start_task(%t4)
    aiex.dma_await_task(%t4)
  }
}

// -----

// FIFO pop-through. A non-token push cannot be awaited directly, but the
// channel runs its queue in order, so awaiting a later token retires every
// non-token push queued ahead of it. Here A and B (no token) sit in front of C
// (token); awaiting C retires A, B and C, leaving only D outstanding. Four more
// pushes then fit before the queue is full again.
aie.device(npu2) {
  %tile_0_0 = aie.tile(0, 0)
  aie.runtime_sequence @fifo_pop_through(%arg0: memref<2048xi32>) {
    %a = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<2048xi32> offset = 0 len = 256)
      aie.end
    }
    aiex.dma_start_task(%a)
    %b = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<2048xi32> offset = 256 len = 256)
      aie.end
    }
    aiex.dma_start_task(%b)
    %c = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<2048xi32> offset = 512 len = 256)
      aie.end
    } {issue_token = true}
    aiex.dma_start_task(%c)
    %d = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<2048xi32> offset = 768 len = 256)
      aie.end
    }
    aiex.dma_start_task(%d)
    // Retires A, B and C. Queue now holds D alone.
    aiex.dma_await_task(%c)
    %e = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<2048xi32> offset = 1024 len = 256)
      aie.end
    }
    aiex.dma_start_task(%e)
    %f = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<2048xi32> offset = 1280 len = 256)
      aie.end
    }
    aiex.dma_start_task(%f)
    %g = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<2048xi32> offset = 1536 len = 256)
      aie.end
    }
    aiex.dma_start_task(%g)
    // D, E, F, G fill the queue again; this one overflows it.
    %h = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<2048xi32> offset = 1792 len = 256)
      aie.end
    }
    // expected-warning@+1 {{whose task queue is only 4 deep, with 4 push(es) not yet known to have completed}}
    aiex.dma_start_task(%h)
  }
}

// -----

// Independent channels have independent queues: four pushes on each of two
// directions on the same tile is fine, even though the tile sees eight.
aie.device(npu2) {
  %tile_0_0 = aie.tile(0, 0)
  aie.runtime_sequence @per_channel_independent(%arg0: memref<1024xi32>,
                                                %arg1: memref<1024xi32>) {
    %m0 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<1024xi32> offset = 0 len = 256)
      aie.end
    }
    aiex.dma_start_task(%m0)
    %s0 = aiex.dma_configure_task(%tile_0_0, S2MM, 0) {
      aie.dma_bd(%arg1 : memref<1024xi32> offset = 0 len = 256)
      aie.end
    }
    aiex.dma_start_task(%s0)
    %m1 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<1024xi32> offset = 256 len = 256)
      aie.end
    }
    aiex.dma_start_task(%m1)
    %s1 = aiex.dma_configure_task(%tile_0_0, S2MM, 0) {
      aie.dma_bd(%arg1 : memref<1024xi32> offset = 256 len = 256)
      aie.end
    }
    aiex.dma_start_task(%s1)
    %m2 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<1024xi32> offset = 512 len = 256)
      aie.end
    }
    aiex.dma_start_task(%m2)
    %s2 = aiex.dma_configure_task(%tile_0_0, S2MM, 0) {
      aie.dma_bd(%arg1 : memref<1024xi32> offset = 512 len = 256)
      aie.end
    }
    aiex.dma_start_task(%s2)
    %m3 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<1024xi32> offset = 768 len = 256)
      aie.end
    }
    aiex.dma_start_task(%m3)
    %s3 = aiex.dma_configure_task(%tile_0_0, S2MM, 0) {
      aie.dma_bd(%arg1 : memref<1024xi32> offset = 768 len = 256)
      aie.end
    }
    aiex.dma_start_task(%s3)
  }
}

// -----

// Queues are per (tile, direction, channel), and shim tiles are one per
// column, so the same channel number in two columns is two independent
// queues. Four pushes on each must stay quiet.
aie.device(npu2) {
  %tile_0_0 = aie.tile(0, 0)
  %tile_1_0 = aie.tile(1, 0)
  aie.runtime_sequence @per_column_independent(%arg0: memref<1024xi32>) {
    %a0 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<1024xi32> offset = 0 len = 256)
      aie.end
    }
    aiex.dma_start_task(%a0)
    %b0 = aiex.dma_configure_task(%tile_1_0, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<1024xi32> offset = 0 len = 256)
      aie.end
    }
    aiex.dma_start_task(%b0)
    %a1 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<1024xi32> offset = 256 len = 256)
      aie.end
    }
    aiex.dma_start_task(%a1)
    %b1 = aiex.dma_configure_task(%tile_1_0, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<1024xi32> offset = 256 len = 256)
      aie.end
    }
    aiex.dma_start_task(%b1)
    %a2 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<1024xi32> offset = 512 len = 256)
      aie.end
    }
    aiex.dma_start_task(%a2)
    %b2 = aiex.dma_configure_task(%tile_1_0, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<1024xi32> offset = 512 len = 256)
      aie.end
    }
    aiex.dma_start_task(%b2)
    %a3 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<1024xi32> offset = 768 len = 256)
      aie.end
    }
    aiex.dma_start_task(%a3)
    %b3 = aiex.dma_configure_task(%tile_1_0, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<1024xi32> offset = 768 len = 256)
      aie.end
    }
    aiex.dma_start_task(%b3)
  }
}

// -----

// repeat_count does not multiply queue slots: one push with repeat_count=8
// occupies one entry and yields one token (Enable_Token_Issue is a bit in the
// same Task_Queue word). Four such pushes are at the limit, not past it.
aie.device(npu2) {
  %tile_0_0 = aie.tile(0, 0)
  aie.runtime_sequence @repeat_count_is_one_slot(%arg0: memref<1024xi32>) {
    %t0 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<1024xi32> offset = 0 len = 256)
      aie.end
    } {repeat_count = 8 : i32}
    aiex.dma_start_task(%t0)
    %t1 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<1024xi32> offset = 256 len = 256)
      aie.end
    } {repeat_count = 8 : i32}
    aiex.dma_start_task(%t1)
    %t2 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<1024xi32> offset = 512 len = 256)
      aie.end
    } {repeat_count = 8 : i32}
    aiex.dma_start_task(%t2)
    %t3 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<1024xi32> offset = 768 len = 256)
      aie.end
    } {repeat_count = 8 : i32}
    aiex.dma_start_task(%t3)
  }
}

// -----

// A raw aiex.npu.sync drains the queue exactly as dma_await_task does -- it is
// the same hardware event, just spelled at a lower level -- so a sequence that
// mixes the two still gets an accurate count. Only the queue is credited here;
// the token balance that guards await stays with the ops whose flags it can
// read off the IR.
aie.device(npu2) {
  %tile_0_0 = aie.tile(0, 0)
  aie.runtime_sequence @raw_sync_drains(%arg0: memref<1024xi32>) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %t0 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<1024xi32> offset = 0 len = 256)
      aie.end
    } {issue_token = true}
    aiex.dma_start_task(%t0)
    aiex.npu.sync(%c0, %c0, %c1, %c0, %c1, %c1) : i32, i32, i32, i32, i32, i32
    %t1 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<1024xi32> offset = 256 len = 256)
      aie.end
    } {issue_token = true}
    aiex.dma_start_task(%t1)
    aiex.npu.sync(%c0, %c0, %c1, %c0, %c1, %c1) : i32, i32, i32, i32, i32, i32
    %t2 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<1024xi32> offset = 512 len = 256)
      aie.end
    } {issue_token = true}
    aiex.dma_start_task(%t2)
    aiex.npu.sync(%c0, %c0, %c1, %c0, %c1, %c1) : i32, i32, i32, i32, i32, i32
    %t3 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<1024xi32> offset = 768 len = 256)
      aie.end
    } {issue_token = true}
    aiex.dma_start_task(%t3)
    aiex.npu.sync(%c0, %c0, %c1, %c0, %c1, %c1) : i32, i32, i32, i32, i32, i32
    %t4 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<1024xi32> offset = 0 len = 256)
      aie.end
    }
    aiex.dma_start_task(%t4)
  }
}
