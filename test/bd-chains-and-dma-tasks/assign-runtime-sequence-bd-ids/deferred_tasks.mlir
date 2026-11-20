// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt %s --aie-assign-runtime-sequence-bd-ids="chain-tasks=true reclaim-bds=true" | FileCheck %s --check-prefixes=CHAIN,RECLAIM
// RUN: aie-opt %s --aie-assign-runtime-sequence-bd-ids=reclaim-bds=true | FileCheck %s --check-prefixes=QUEUE,RECLAIM

module {
  aie.device(npu2) {
    %left = aie.tile(0, 0)
    %right = aie.tile(1, 0)

    // CHAIN-LABEL: @parallel
    // CHAIN: %[[LEFT:.*]] = aiex.dma_configure_task(%shim_noc_tile_0_0, MM2S, 0)
    // CHAIN: offset = 0 len = 4) {bd_id = 0 : i32}
    // CHAIN-NEXT: aie.next_bd
    // CHAIN: offset = 8 len = 4) {bd_id = 1 : i32}
    // CHAIN: aiex.dma_start_task(%[[LEFT]])
    // CHAIN-NEXT: %[[RIGHT:.*]] = aiex.dma_configure_task(%shim_noc_tile_1_0, MM2S, 0)
    // CHAIN: offset = 4 len = 4) {bd_id = 0 : i32}
    // CHAIN: aiex.dma_start_task(%[[RIGHT]])
    // CHAIN-NEXT: aiex.dma_await_task(%[[LEFT]])
    // CHAIN-NEXT: aiex.dma_await_task(%[[RIGHT]])
    // CHAIN-NEXT: {{^ *}}}
    // QUEUE-LABEL: @parallel
    // QUEUE: aiex.dma_start_task
    // QUEUE-NOT: aiex.dma_await_task
    // QUEUE: aiex.dma_start_task
    // QUEUE-NOT: aiex.dma_await_task
    // QUEUE: aiex.dma_start_task
    // QUEUE-COUNT-3: aiex.dma_await_task
    // QUEUE-NEXT: {{^ *}}}
    aie.runtime_sequence @parallel(%buffer: memref<16xi32>) {
      %first = aiex.dma_configure_task(%left, MM2S, 0) {
        aie.dma_bd(%buffer : memref<16xi32> offset = 0 len = 4)
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%first)
      aiex.dma_await_task(%first) {defer_completion}
      %other = aiex.dma_configure_task(%right, MM2S, 0) {
        aie.dma_bd(%buffer : memref<16xi32> offset = 4 len = 4)
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%other)
      aiex.dma_await_task(%other) {defer_completion}
      %second = aiex.dma_configure_task(%left, MM2S, 0) {
        aie.dma_bd(%buffer : memref<16xi32> offset = 8 len = 4)
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%second)
      aiex.dma_await_task(%second)
    }

    // CHAIN-LABEL: @boundary
    // CHAIN: aiex.dma_start_task
    // CHAIN-NEXT: aiex.dma_await_task
    // CHAIN-NEXT: %{{.*}} = aiex.dma_configure_task
    // CHAIN: aiex.dma_start_task
    // CHAIN-NEXT: aiex.dma_await_task
    // QUEUE-LABEL: @boundary
    // QUEUE: aiex.dma_start_task
    // QUEUE-NEXT: aiex.dma_await_task
    // QUEUE-NEXT: %{{.*}} = aiex.dma_configure_task
    aie.runtime_sequence @boundary(%buffer: memref<16xi32>) {
      %first = aiex.dma_configure_task(%left, MM2S, 0) {
        aie.dma_bd(%buffer : memref<16xi32> offset = 0 len = 4)
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%first)
      aiex.dma_await_task(%first)
      %second = aiex.dma_configure_task(%left, MM2S, 0) {
        aie.dma_bd(%buffer : memref<16xi32> offset = 4 len = 4)
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%second)
      aiex.dma_await_task(%second)
    }

    // CHAIN-LABEL: @untokened_predecessor
    // CHAIN: aiex.dma_start_task
    // CHAIN: offset = 4 len = 4) {bd_id = 1 : i32}
    // CHAIN-NEXT: aie.next_bd
    // CHAIN: offset = 8 len = 4) {bd_id = 2 : i32}
    // CHAIN: aiex.dma_start_task
    // CHAIN-NEXT: aiex.dma_await_task
    // CHAIN: offset = 12 len = 4) {bd_id = 0 : i32}
    // QUEUE-LABEL: @untokened_predecessor
    // QUEUE-COUNT-3: aiex.dma_start_task
    // QUEUE-COUNT-2: aiex.dma_await_task
    // QUEUE: offset = 12 len = 4) {bd_id = 0 : i32}
    aie.runtime_sequence @untokened_predecessor(%buffer: memref<16xi32>) {
      %earlier = aiex.dma_configure_task(%left, MM2S, 0) {
        aie.dma_bd(%buffer : memref<16xi32> offset = 0 len = 4)
        aie.end
      }
      aiex.dma_start_task(%earlier)
      %first = aiex.dma_configure_task(%left, MM2S, 0) {
        aie.dma_bd(%buffer : memref<16xi32> offset = 4 len = 4)
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%first)
      aiex.dma_await_task(%first) {defer_completion}
      %second = aiex.dma_configure_task(%left, MM2S, 0) {
        aie.dma_bd(%buffer : memref<16xi32> offset = 8 len = 4)
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%second)
      aiex.dma_await_task(%second)
      aiex.dma_free_task(%earlier)
      %reuse = aiex.dma_configure_task(%left, MM2S, 0) {
        aie.dma_bd(%buffer : memref<16xi32> offset = 12 len = 4)
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%reuse)
      aiex.dma_await_task(%reuse)
    }

    // RECLAIM-LABEL: @reclaim_after_interval
    // RECLAIM: aiex.dma_start_task
    // RECLAIM-NOT: aiex.npu.maskpoll
    // RECLAIM: aiex.dma_await_task
    // RECLAIM: aiex.npu.maskpoll
    // RECLAIM-NEXT: %{{.*}} = aiex.dma_configure_task(%shim_noc_tile_0_0, MM2S, 1)
    aie.runtime_sequence @reclaim_after_interval(%buffer: memref<16xi32>) {
      %waiting = aiex.dma_configure_task(%left, S2MM, 0) {
        aie.dma_bd(%buffer : memref<16xi32> offset = 0 len = 4)
        aie.end
      }
      aiex.dma_start_task(%waiting)
      %first = aiex.dma_configure_task(%left, MM2S, 0) {
        aie.dma_bd(%buffer : memref<16xi32> offset = 4 len = 4)
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%first)
      aiex.dma_await_task(%first) {defer_completion}
      %second = aiex.dma_configure_task(%left, MM2S, 0) {
        aie.dma_bd(%buffer : memref<16xi32> offset = 8 len = 4)
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%second)
      aiex.dma_await_task(%second)
      %full = aiex.dma_configure_task(%left, MM2S, 1) {
        aie.dma_bd(%buffer : memref<16xi32> offset = 0 len = 1)
        aie.next_bd ^bd1
      ^bd1:
        aie.dma_bd(%buffer : memref<16xi32> offset = 1 len = 1)
        aie.next_bd ^bd2
      ^bd2:
        aie.dma_bd(%buffer : memref<16xi32> offset = 2 len = 1)
        aie.next_bd ^bd3
      ^bd3:
        aie.dma_bd(%buffer : memref<16xi32> offset = 3 len = 1)
        aie.next_bd ^bd4
      ^bd4:
        aie.dma_bd(%buffer : memref<16xi32> offset = 4 len = 1)
        aie.next_bd ^bd5
      ^bd5:
        aie.dma_bd(%buffer : memref<16xi32> offset = 5 len = 1)
        aie.next_bd ^bd6
      ^bd6:
        aie.dma_bd(%buffer : memref<16xi32> offset = 6 len = 1)
        aie.next_bd ^bd7
      ^bd7:
        aie.dma_bd(%buffer : memref<16xi32> offset = 7 len = 1)
        aie.next_bd ^bd8
      ^bd8:
        aie.dma_bd(%buffer : memref<16xi32> offset = 8 len = 1)
        aie.next_bd ^bd9
      ^bd9:
        aie.dma_bd(%buffer : memref<16xi32> offset = 9 len = 1)
        aie.next_bd ^bd10
      ^bd10:
        aie.dma_bd(%buffer : memref<16xi32> offset = 10 len = 1)
        aie.next_bd ^bd11
      ^bd11:
        aie.dma_bd(%buffer : memref<16xi32> offset = 11 len = 1)
        aie.next_bd ^bd12
      ^bd12:
        aie.dma_bd(%buffer : memref<16xi32> offset = 12 len = 1)
        aie.next_bd ^bd13
      ^bd13:
        aie.dma_bd(%buffer : memref<16xi32> offset = 13 len = 1)
        aie.next_bd ^bd14
      ^bd14:
        aie.dma_bd(%buffer : memref<16xi32> offset = 14 len = 1)
        aie.next_bd ^bd15
      ^bd15:
        aie.dma_bd(%buffer : memref<16xi32> offset = 15 len = 1)
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%full)
      aiex.dma_await_task(%full)
    }
  }
}