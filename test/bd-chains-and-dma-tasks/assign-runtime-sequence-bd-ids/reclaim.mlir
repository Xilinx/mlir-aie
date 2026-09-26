//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

// RUN: aie-opt --aie-assign-runtime-sequence-bd-ids %s | FileCheck %s
// RUN: aie-opt --aie-assign-runtime-sequence-bd-ids --aie-dma-tasks-to-npu \
// RUN:   --aie-dma-to-npu %s | FileCheck %s --check-prefix=LOWERED

// When a tile runs out of BD ids, the pass takes them back from a task that
// was started and never freed. It first looks for a task that a poll already in
// the stream proves finished. Failing that, it inserts a poll of the channel's
// status register proving one finished: a task with j pushes queued behind it
// on its channel is finished once Task_Queue_Size (bits 22:20, not counting the
// running task) is at most j-1, and a task last on its channel once the channel
// is idle (queue size, Channel_Running bit 19 and the stall bits 5:2 all clear,
// mask 0x78003C). Sequences that fit are untouched; see the other tests here.
//
// npu2 shim DMA_MM2S_Status_0 is 0x1D228 (119336); mem tile (0,1)
// DMA_MM2S_Status_0 is 0x1A0680 (1705600).

// Seventeen single-BD tasks on one channel. The queue-space poll before push 16
// leaves at most 4 of the first 15 unfinished, so the first is already proven
// and the 17th task takes its id without a poll of its own.
// CHECK-LABEL: @credit
// CHECK-NOT:   arith.constant {{7340032|6291456|7864380}} : i32
// CHECK-COUNT-12: aiex.npu.maskpoll
// CHECK-NOT:   arith.constant {{7340032|6291456|7864380}} : i32
// CHECK:       {bd_id = 0 : i32}
// CHECK:       aiex.npu.maskpoll
// CHECK-NOT:   aiex.npu.maskpoll

// t0 has j = 1, 2, 3, 4 pushes behind it on MM2S 0; MM2S 1 holds the rest of
// the pool in one chain. The poll bound is the largest 2^k-1 <= j-1, so the
// mask is a whole number of high bits of the field: 22:20 for j = 1, 22:21 for
// j = 2 and 3, 22 for j = 4.
// CHECK-LABEL: @j1
// CHECK-NOT:   aiex.npu.maskpoll
// CHECK:       arith.constant 7340032 : i32
// CHECK-NEXT:  arith.constant 0 : i32
// CHECK-NEXT:  arith.constant 119336 : i32
// CHECK-NEXT:  aiex.npu.maskpoll
// CHECK-NEXT:  aiex.dma_configure_task(%{{.*}}, S2MM, 0)
// CHECK-NEXT:  {bd_id = 0 : i32}
// CHECK-NOT:   aiex.npu.maskpoll

// CHECK-LABEL: @j2
// CHECK-NOT:   aiex.npu.maskpoll
// CHECK:       arith.constant 6291456 : i32
// CHECK-NEXT:  arith.constant 0 : i32
// CHECK-NEXT:  arith.constant 119336 : i32
// CHECK-NEXT:  aiex.npu.maskpoll
// CHECK-NEXT:  aiex.dma_configure_task(%{{.*}}, S2MM, 0)
// CHECK-NEXT:  {bd_id = 0 : i32}

// CHECK-LABEL: @j3
// CHECK-NOT:   aiex.npu.maskpoll
// CHECK:       arith.constant 6291456 : i32
// CHECK-NEXT:  arith.constant 0 : i32
// CHECK-NEXT:  arith.constant 119336 : i32
// CHECK-NEXT:  aiex.npu.maskpoll
// CHECK-NEXT:  aiex.dma_configure_task(%{{.*}}, S2MM, 0)
// CHECK-NEXT:  {bd_id = 0 : i32}

// The queue-space poll before the fifth push leaves all four earlier pushes
// possibly unfinished, so it proves nothing and the pass adds its own.
// CHECK-LABEL: @j4
// CHECK:       aiex.npu.maskpoll
// CHECK:       aiex.dma_start_task
// CHECK:       arith.constant 4194304 : i32
// CHECK-NEXT:  arith.constant 0 : i32
// CHECK-NEXT:  arith.constant 119336 : i32
// CHECK-NEXT:  aiex.npu.maskpoll
// CHECK-NEXT:  aiex.dma_configure_task(%{{.*}}, S2MM, 0)
// CHECK-NEXT:  {bd_id = 0 : i32}

// Every task is last on its channel: the oldest is taken behind an idle poll,
// all four of its ids at once.
// CHECK-LABEL: @idle
// CHECK-NOT:   aiex.npu.maskpoll
// CHECK:       arith.constant 7864380 : i32
// CHECK-NEXT:  arith.constant 0 : i32
// CHECK-NEXT:  arith.constant 119336 : i32
// CHECK-NEXT:  aiex.npu.maskpoll
// CHECK-NEXT:  aiex.dma_configure_task(%{{.*}}, MM2S, 0)
// CHECK-NEXT:  {bd_id = 0 : i32}

// t0 is started again after the allocation, so its id stays with it; t1 goes
// instead, behind an idle poll since nothing is queued after it.
// CHECK-LABEL: @restart
// CHECK-NOT:   aiex.npu.maskpoll
// CHECK:       arith.constant 7864380 : i32
// CHECK-NEXT:  arith.constant 0 : i32
// CHECK-NEXT:  arith.constant 119336 : i32
// CHECK-NEXT:  aiex.npu.maskpoll
// CHECK-NEXT:  aiex.dma_configure_task(%{{.*}}, S2MM, 0)
// CHECK-NEXT:  {bd_id = 1 : i32}

// Freeing a reclaimed task later is a no-op; awaiting it still waits on its
// token.
// CHECK-LABEL: @free_await
// CHECK:       arith.constant 7340032 : i32
// CHECK:       aiex.dma_await_task
// CHECK-NOT:   aiex.dma_free_task
// LOWERED-LABEL: @free_await
// LOWERED:       aiex.npu.maskpoll
// LOWERED:       aiex.npu.sync

// On a mem tile only a task of the same channel parity can give an id to an
// even channel: the older odd-channel o0 is passed over for e0.
// CHECK-LABEL: @parity
// CHECK-NOT:   aiex.npu.maskpoll
// CHECK:       arith.constant 7340032 : i32
// CHECK-NEXT:  arith.constant 0 : i32
// CHECK-NEXT:  arith.constant 1705600 : i32
// CHECK-NEXT:  aiex.npu.maskpoll
// CHECK-NEXT:  aiex.dma_configure_task(%{{.*}}, MM2S, 4)
// CHECK-NEXT:  {bd_id = 0 : i32}

aie.device(npu2) @shim_dev {
  %shim = aie.tile(0, 0)
  aie.runtime_sequence @credit(%arg0: memref<64xi32>) {
    %t0 = aiex.dma_configure_task(%shim, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 8)
      aie.end
    }
    aiex.dma_start_task(%t0)
    %t1 = aiex.dma_configure_task(%shim, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 8)
      aie.end
    }
    aiex.dma_start_task(%t1)
    %t2 = aiex.dma_configure_task(%shim, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 8)
      aie.end
    }
    aiex.dma_start_task(%t2)
    %t3 = aiex.dma_configure_task(%shim, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 8)
      aie.end
    }
    aiex.dma_start_task(%t3)
    %t4 = aiex.dma_configure_task(%shim, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 8)
      aie.end
    }
    aiex.dma_start_task(%t4)
    %t5 = aiex.dma_configure_task(%shim, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 8)
      aie.end
    }
    aiex.dma_start_task(%t5)
    %t6 = aiex.dma_configure_task(%shim, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 8)
      aie.end
    }
    aiex.dma_start_task(%t6)
    %t7 = aiex.dma_configure_task(%shim, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 8)
      aie.end
    }
    aiex.dma_start_task(%t7)
    %t8 = aiex.dma_configure_task(%shim, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 8)
      aie.end
    }
    aiex.dma_start_task(%t8)
    %t9 = aiex.dma_configure_task(%shim, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 8)
      aie.end
    }
    aiex.dma_start_task(%t9)
    %t10 = aiex.dma_configure_task(%shim, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 8)
      aie.end
    }
    aiex.dma_start_task(%t10)
    %t11 = aiex.dma_configure_task(%shim, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 8)
      aie.end
    }
    aiex.dma_start_task(%t11)
    %t12 = aiex.dma_configure_task(%shim, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 8)
      aie.end
    }
    aiex.dma_start_task(%t12)
    %t13 = aiex.dma_configure_task(%shim, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 8)
      aie.end
    }
    aiex.dma_start_task(%t13)
    %t14 = aiex.dma_configure_task(%shim, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 8)
      aie.end
    }
    aiex.dma_start_task(%t14)
    %t15 = aiex.dma_configure_task(%shim, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 8)
      aie.end
    }
    aiex.dma_start_task(%t15)
    %t16 = aiex.dma_configure_task(%shim, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 8)
      aie.end
    }
    aiex.dma_start_task(%t16)
  }
  aie.runtime_sequence @j1(%arg0: memref<64xi32>) {
    %t0 = aiex.dma_configure_task(%shim, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 8)
      aie.end
    }
    aiex.dma_start_task(%t0)
    %t1 = aiex.dma_configure_task(%shim, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 8)
      aie.end
    }
    aiex.dma_start_task(%t1)
    %big = aiex.dma_configure_task(%shim, MM2S, 1) {
      aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 8)
      aie.next_bd ^bd1
    ^bd1:
      aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 8)
      aie.next_bd ^bd2
    ^bd2:
      aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 8)
      aie.next_bd ^bd3
    ^bd3:
      aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 8)
      aie.next_bd ^bd4
    ^bd4:
      aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 8)
      aie.next_bd ^bd5
    ^bd5:
      aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 8)
      aie.next_bd ^bd6
    ^bd6:
      aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 8)
      aie.next_bd ^bd7
    ^bd7:
      aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 8)
      aie.next_bd ^bd8
    ^bd8:
      aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 8)
      aie.next_bd ^bd9
    ^bd9:
      aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 8)
      aie.next_bd ^bd10
    ^bd10:
      aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 8)
      aie.next_bd ^bd11
    ^bd11:
      aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 8)
      aie.next_bd ^bd12
    ^bd12:
      aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 8)
      aie.next_bd ^bd13
    ^bd13:
      aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 8)
      aie.end
    }
    aiex.dma_start_task(%big)
    %new = aiex.dma_configure_task(%shim, S2MM, 0) {
      aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 8)
      aie.end
    }
    aiex.dma_start_task(%new)
  }
  aie.runtime_sequence @j2(%arg0: memref<64xi32>) {
    %t0 = aiex.dma_configure_task(%shim, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 8)
      aie.end
    }
    aiex.dma_start_task(%t0)
    %t1 = aiex.dma_configure_task(%shim, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 8)
      aie.end
    }
    aiex.dma_start_task(%t1)
    %t2 = aiex.dma_configure_task(%shim, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 8)
      aie.end
    }
    aiex.dma_start_task(%t2)
    %big = aiex.dma_configure_task(%shim, MM2S, 1) {
      aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 8)
      aie.next_bd ^bd1
    ^bd1:
      aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 8)
      aie.next_bd ^bd2
    ^bd2:
      aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 8)
      aie.next_bd ^bd3
    ^bd3:
      aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 8)
      aie.next_bd ^bd4
    ^bd4:
      aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 8)
      aie.next_bd ^bd5
    ^bd5:
      aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 8)
      aie.next_bd ^bd6
    ^bd6:
      aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 8)
      aie.next_bd ^bd7
    ^bd7:
      aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 8)
      aie.next_bd ^bd8
    ^bd8:
      aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 8)
      aie.next_bd ^bd9
    ^bd9:
      aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 8)
      aie.next_bd ^bd10
    ^bd10:
      aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 8)
      aie.next_bd ^bd11
    ^bd11:
      aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 8)
      aie.next_bd ^bd12
    ^bd12:
      aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 8)
      aie.end
    }
    aiex.dma_start_task(%big)
    %new = aiex.dma_configure_task(%shim, S2MM, 0) {
      aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 8)
      aie.end
    }
    aiex.dma_start_task(%new)
  }
  aie.runtime_sequence @j3(%arg0: memref<64xi32>) {
    %t0 = aiex.dma_configure_task(%shim, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 8)
      aie.end
    }
    aiex.dma_start_task(%t0)
    %t1 = aiex.dma_configure_task(%shim, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 8)
      aie.end
    }
    aiex.dma_start_task(%t1)
    %t2 = aiex.dma_configure_task(%shim, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 8)
      aie.end
    }
    aiex.dma_start_task(%t2)
    %t3 = aiex.dma_configure_task(%shim, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 8)
      aie.end
    }
    aiex.dma_start_task(%t3)
    %big = aiex.dma_configure_task(%shim, MM2S, 1) {
      aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 8)
      aie.next_bd ^bd1
    ^bd1:
      aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 8)
      aie.next_bd ^bd2
    ^bd2:
      aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 8)
      aie.next_bd ^bd3
    ^bd3:
      aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 8)
      aie.next_bd ^bd4
    ^bd4:
      aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 8)
      aie.next_bd ^bd5
    ^bd5:
      aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 8)
      aie.next_bd ^bd6
    ^bd6:
      aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 8)
      aie.next_bd ^bd7
    ^bd7:
      aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 8)
      aie.next_bd ^bd8
    ^bd8:
      aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 8)
      aie.next_bd ^bd9
    ^bd9:
      aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 8)
      aie.next_bd ^bd10
    ^bd10:
      aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 8)
      aie.next_bd ^bd11
    ^bd11:
      aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 8)
      aie.end
    }
    aiex.dma_start_task(%big)
    %new = aiex.dma_configure_task(%shim, S2MM, 0) {
      aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 8)
      aie.end
    }
    aiex.dma_start_task(%new)
  }
  aie.runtime_sequence @j4(%arg0: memref<64xi32>) {
    %t0 = aiex.dma_configure_task(%shim, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 8)
      aie.end
    }
    aiex.dma_start_task(%t0)
    %t1 = aiex.dma_configure_task(%shim, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 8)
      aie.end
    }
    aiex.dma_start_task(%t1)
    %t2 = aiex.dma_configure_task(%shim, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 8)
      aie.end
    }
    aiex.dma_start_task(%t2)
    %t3 = aiex.dma_configure_task(%shim, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 8)
      aie.end
    }
    aiex.dma_start_task(%t3)
    %t4 = aiex.dma_configure_task(%shim, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 8)
      aie.end
    }
    aiex.dma_start_task(%t4)
    %big = aiex.dma_configure_task(%shim, MM2S, 1) {
      aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 8)
      aie.next_bd ^bd1
    ^bd1:
      aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 8)
      aie.next_bd ^bd2
    ^bd2:
      aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 8)
      aie.next_bd ^bd3
    ^bd3:
      aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 8)
      aie.next_bd ^bd4
    ^bd4:
      aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 8)
      aie.next_bd ^bd5
    ^bd5:
      aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 8)
      aie.next_bd ^bd6
    ^bd6:
      aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 8)
      aie.next_bd ^bd7
    ^bd7:
      aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 8)
      aie.next_bd ^bd8
    ^bd8:
      aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 8)
      aie.next_bd ^bd9
    ^bd9:
      aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 8)
      aie.next_bd ^bd10
    ^bd10:
      aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 8)
      aie.end
    }
    aiex.dma_start_task(%big)
    %new = aiex.dma_configure_task(%shim, S2MM, 0) {
      aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 8)
      aie.end
    }
    aiex.dma_start_task(%new)
  }
  aie.runtime_sequence @idle(%arg0: memref<64xi32>) {
    %t0 = aiex.dma_configure_task(%shim, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 8)
      aie.next_bd ^bd1
    ^bd1:
      aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 8)
      aie.next_bd ^bd2
    ^bd2:
      aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 8)
      aie.next_bd ^bd3
    ^bd3:
      aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 8)
      aie.end
    }
    aiex.dma_start_task(%t0)
    %t1 = aiex.dma_configure_task(%shim, MM2S, 1) {
      aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 8)
      aie.next_bd ^bd1
    ^bd1:
      aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 8)
      aie.next_bd ^bd2
    ^bd2:
      aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 8)
      aie.next_bd ^bd3
    ^bd3:
      aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 8)
      aie.end
    }
    aiex.dma_start_task(%t1)
    %t2 = aiex.dma_configure_task(%shim, S2MM, 0) {
      aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 8)
      aie.next_bd ^bd1
    ^bd1:
      aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 8)
      aie.next_bd ^bd2
    ^bd2:
      aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 8)
      aie.next_bd ^bd3
    ^bd3:
      aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 8)
      aie.end
    }
    aiex.dma_start_task(%t2)
    %t3 = aiex.dma_configure_task(%shim, S2MM, 1) {
      aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 8)
      aie.next_bd ^bd1
    ^bd1:
      aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 8)
      aie.next_bd ^bd2
    ^bd2:
      aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 8)
      aie.next_bd ^bd3
    ^bd3:
      aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 8)
      aie.end
    }
    aiex.dma_start_task(%t3)
    %new = aiex.dma_configure_task(%shim, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 8)
      aie.end
    }
    aiex.dma_start_task(%new)
  }
  aie.runtime_sequence @restart(%arg0: memref<64xi32>) {
    %t0 = aiex.dma_configure_task(%shim, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 8)
      aie.end
    }
    aiex.dma_start_task(%t0)
    %t1 = aiex.dma_configure_task(%shim, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 8)
      aie.end
    }
    aiex.dma_start_task(%t1)
    %big = aiex.dma_configure_task(%shim, MM2S, 1) {
      aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 8)
      aie.next_bd ^bd1
    ^bd1:
      aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 8)
      aie.next_bd ^bd2
    ^bd2:
      aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 8)
      aie.next_bd ^bd3
    ^bd3:
      aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 8)
      aie.next_bd ^bd4
    ^bd4:
      aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 8)
      aie.next_bd ^bd5
    ^bd5:
      aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 8)
      aie.next_bd ^bd6
    ^bd6:
      aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 8)
      aie.next_bd ^bd7
    ^bd7:
      aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 8)
      aie.next_bd ^bd8
    ^bd8:
      aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 8)
      aie.next_bd ^bd9
    ^bd9:
      aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 8)
      aie.next_bd ^bd10
    ^bd10:
      aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 8)
      aie.next_bd ^bd11
    ^bd11:
      aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 8)
      aie.next_bd ^bd12
    ^bd12:
      aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 8)
      aie.next_bd ^bd13
    ^bd13:
      aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 8)
      aie.end
    }
    aiex.dma_start_task(%big)
    %new = aiex.dma_configure_task(%shim, S2MM, 0) {
      aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 8)
      aie.end
    }
    aiex.dma_start_task(%new)
    aiex.dma_start_task(%t0)
  }
  aie.runtime_sequence @free_await(%arg0: memref<64xi32>) {
    %t0 = aiex.dma_configure_task(%shim, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 8)
      aie.end
    } {issue_token = true}
    aiex.dma_start_task(%t0)
    %t1 = aiex.dma_configure_task(%shim, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 8)
      aie.end
    }
    aiex.dma_start_task(%t1)
    %big = aiex.dma_configure_task(%shim, MM2S, 1) {
      aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 8)
      aie.next_bd ^bd1
    ^bd1:
      aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 8)
      aie.next_bd ^bd2
    ^bd2:
      aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 8)
      aie.next_bd ^bd3
    ^bd3:
      aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 8)
      aie.next_bd ^bd4
    ^bd4:
      aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 8)
      aie.next_bd ^bd5
    ^bd5:
      aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 8)
      aie.next_bd ^bd6
    ^bd6:
      aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 8)
      aie.next_bd ^bd7
    ^bd7:
      aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 8)
      aie.next_bd ^bd8
    ^bd8:
      aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 8)
      aie.next_bd ^bd9
    ^bd9:
      aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 8)
      aie.next_bd ^bd10
    ^bd10:
      aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 8)
      aie.next_bd ^bd11
    ^bd11:
      aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 8)
      aie.next_bd ^bd12
    ^bd12:
      aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 8)
      aie.next_bd ^bd13
    ^bd13:
      aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 8)
      aie.end
    }
    aiex.dma_start_task(%big)
    %new = aiex.dma_configure_task(%shim, S2MM, 0) {
      aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 8)
      aie.end
    }
    aiex.dma_start_task(%new)
    aiex.dma_await_task(%t0)
    aiex.dma_free_task(%t0)
  }
}

aie.device(npu2) @mem_dev {
  %mem = aie.tile(0, 1)
  %buf = aie.buffer(%mem) {address = 0 : i32} : memref<64xi32>
  aie.runtime_sequence @parity(%arg0: memref<64xi32>) {
    %o0 = aiex.dma_configure_task(%mem, MM2S, 1) {
      aie.dma_bd(%buf : memref<64xi32> offset = 0 len = 8)
      aie.end
    }
    aiex.dma_start_task(%o0)
    %o1 = aiex.dma_configure_task(%mem, MM2S, 1) {
      aie.dma_bd(%buf : memref<64xi32> offset = 0 len = 8)
      aie.end
    }
    aiex.dma_start_task(%o1)
    %e0 = aiex.dma_configure_task(%mem, MM2S, 0) {
      aie.dma_bd(%buf : memref<64xi32> offset = 0 len = 8)
      aie.end
    }
    aiex.dma_start_task(%e0)
    %e1 = aiex.dma_configure_task(%mem, MM2S, 0) {
      aie.dma_bd(%buf : memref<64xi32> offset = 0 len = 8)
      aie.end
    }
    aiex.dma_start_task(%e1)
    %e2 = aiex.dma_configure_task(%mem, S2MM, 0) {
      aie.dma_bd(%buf : memref<64xi32> offset = 0 len = 8)
      aie.next_bd ^bd1
    ^bd1:
      aie.dma_bd(%buf : memref<64xi32> offset = 0 len = 8)
      aie.next_bd ^bd2
    ^bd2:
      aie.dma_bd(%buf : memref<64xi32> offset = 0 len = 8)
      aie.next_bd ^bd3
    ^bd3:
      aie.dma_bd(%buf : memref<64xi32> offset = 0 len = 8)
      aie.next_bd ^bd4
    ^bd4:
      aie.dma_bd(%buf : memref<64xi32> offset = 0 len = 8)
      aie.next_bd ^bd5
    ^bd5:
      aie.dma_bd(%buf : memref<64xi32> offset = 0 len = 8)
      aie.next_bd ^bd6
    ^bd6:
      aie.dma_bd(%buf : memref<64xi32> offset = 0 len = 8)
      aie.next_bd ^bd7
    ^bd7:
      aie.dma_bd(%buf : memref<64xi32> offset = 0 len = 8)
      aie.next_bd ^bd8
    ^bd8:
      aie.dma_bd(%buf : memref<64xi32> offset = 0 len = 8)
      aie.next_bd ^bd9
    ^bd9:
      aie.dma_bd(%buf : memref<64xi32> offset = 0 len = 8)
      aie.next_bd ^bd10
    ^bd10:
      aie.dma_bd(%buf : memref<64xi32> offset = 0 len = 8)
      aie.end
    }
    aiex.dma_start_task(%e2)
    %e3 = aiex.dma_configure_task(%mem, S2MM, 2) {
      aie.dma_bd(%buf : memref<64xi32> offset = 0 len = 8)
      aie.next_bd ^bd1
    ^bd1:
      aie.dma_bd(%buf : memref<64xi32> offset = 0 len = 8)
      aie.next_bd ^bd2
    ^bd2:
      aie.dma_bd(%buf : memref<64xi32> offset = 0 len = 8)
      aie.next_bd ^bd3
    ^bd3:
      aie.dma_bd(%buf : memref<64xi32> offset = 0 len = 8)
      aie.next_bd ^bd4
    ^bd4:
      aie.dma_bd(%buf : memref<64xi32> offset = 0 len = 8)
      aie.next_bd ^bd5
    ^bd5:
      aie.dma_bd(%buf : memref<64xi32> offset = 0 len = 8)
      aie.next_bd ^bd6
    ^bd6:
      aie.dma_bd(%buf : memref<64xi32> offset = 0 len = 8)
      aie.next_bd ^bd7
    ^bd7:
      aie.dma_bd(%buf : memref<64xi32> offset = 0 len = 8)
      aie.next_bd ^bd8
    ^bd8:
      aie.dma_bd(%buf : memref<64xi32> offset = 0 len = 8)
      aie.next_bd ^bd9
    ^bd9:
      aie.dma_bd(%buf : memref<64xi32> offset = 0 len = 8)
      aie.next_bd ^bd10
    ^bd10:
      aie.dma_bd(%buf : memref<64xi32> offset = 0 len = 8)
      aie.end
    }
    aiex.dma_start_task(%e3)
    %new = aiex.dma_configure_task(%mem, MM2S, 4) {
      aie.dma_bd(%buf : memref<64xi32> offset = 0 len = 8)
      aie.end
    }
    aiex.dma_start_task(%new)
  }
}
