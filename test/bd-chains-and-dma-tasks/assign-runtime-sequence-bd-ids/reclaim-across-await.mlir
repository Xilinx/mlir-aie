//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

// RUN: aie-opt --aie-assign-runtime-sequence-bd-ids='reclaim-bds=true' %s | FileCheck %s

// An await retires pushes a status poll already proved finished, and those
// queued behind its token stay proved. The idle poll after d's start proves d
// alone: a, b and c went before it. With every other id held, f takes d's id
// (0) without a poll of its own.
//
// npu2 shim DMA_MM2S_Status_0 is 0x1D228 (119336); the idle mask is 0x78003C
// (7864380).

// CHECK-LABEL: @proven_across_await
// CHECK:       aiex.npu.maskpoll
// CHECK:       aiex.dma_await_task
// CHECK:       aiex.npu.maskpoll
// CHECK-NOT:   aiex.npu.maskpoll
// CHECK:       aiex.dma_configure_task(%{{.*}}, S2MM, 0)
// CHECK-COUNT-15: {bd_id = {{[1-9]|1[0-5]}} : i32
// CHECK:       {bd_id = 0 : i32}
// CHECK-NOT:   aiex.npu.maskpoll

aie.device(npu2) @shim_dev {
  %shim = aie.tile(0, 0)
  aie.runtime_sequence @proven_across_await(%arg0: memref<64xi32>) {
    %status = arith.constant 119336 : i32
    %zero = arith.constant 0 : i32
    %idle = arith.constant 7864380 : i32
    %a = aiex.dma_configure_task(%shim, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 8)
      aie.end
    }
    aiex.dma_start_task(%a)
    %b = aiex.dma_configure_task(%shim, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 8)
      aie.end
    } {issue_token = true}
    aiex.dma_start_task(%b)
    %c = aiex.dma_configure_task(%shim, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 8)
      aie.end
    }
    aiex.dma_start_task(%c)
    aiex.npu.maskpoll(%status, %zero, %idle) : i32, i32, i32
    aiex.dma_await_task(%b)
    aiex.dma_free_task(%a)
    aiex.dma_free_task(%c)
    %d = aiex.dma_configure_task(%shim, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 8)
      aie.end
    }
    aiex.dma_start_task(%d)
    aiex.npu.maskpoll(%status, %zero, %idle) : i32, i32, i32
    %f = aiex.dma_configure_task(%shim, S2MM, 0) {
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
      aie.next_bd ^bd14
    ^bd14:
      aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 8)
      aie.next_bd ^bd15
    ^bd15:
      aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 8)
      aie.end
    }
    aiex.dma_start_task(%f)
  }
}
