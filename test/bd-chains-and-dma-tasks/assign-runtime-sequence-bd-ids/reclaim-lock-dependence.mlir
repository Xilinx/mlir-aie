//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

// RUN: aie-opt --aie-assign-runtime-sequence-bd-ids='reclaim-bds=true warn-unreleased-transfers=false' \
// RUN:   --verify-diagnostics --split-input-file %s

// A core tile's two channels share its 16 BD ids. A fill chain and a peek
// chain, restarted later, hold all 16, and %final needs one more. The fill acquires
// %prod, which only %final releases: it cannot finish before %final is pushed,
// so a poll for it would never return. Reclaim must not take its ids.

aie.device(npu1) {
  %mt = aie.tile(0, 2)
  %buf = aie.buffer(%mt) : memref<256xi32>
  %prod = aie.lock(%mt, 0) {init = 8 : i32}
  %cons = aie.lock(%mt, 1) {init = 0 : i32}
  %other = aie.lock(%mt, 2) {init = 1 : i32}
  aie.runtime_sequence @waits_on_final(%arg0: memref<64xi32>) {
    %c1 = arith.constant 1 : i32
    %fill = aiex.dma_configure_task(%mt, S2MM, 0) {
      aie.use_lock(%prod, AcquireGreaterEqual, %c1)
      aie.dma_bd(%buf : memref<256xi32> offset = 0 len = 16)
      aie.use_lock(%cons, Release, %c1)
      aie.next_bd ^bd1
    ^bd1:
      aie.use_lock(%prod, AcquireGreaterEqual, %c1)
      aie.dma_bd(%buf : memref<256xi32> offset = 16 len = 16)
      aie.use_lock(%cons, Release, %c1)
      aie.next_bd ^bd2
    ^bd2:
      aie.use_lock(%prod, AcquireGreaterEqual, %c1)
      aie.dma_bd(%buf : memref<256xi32> offset = 32 len = 16)
      aie.use_lock(%cons, Release, %c1)
      aie.next_bd ^bd3
    ^bd3:
      aie.use_lock(%prod, AcquireGreaterEqual, %c1)
      aie.dma_bd(%buf : memref<256xi32> offset = 48 len = 16)
      aie.use_lock(%cons, Release, %c1)
      aie.next_bd ^bd4
    ^bd4:
      aie.use_lock(%prod, AcquireGreaterEqual, %c1)
      aie.dma_bd(%buf : memref<256xi32> offset = 64 len = 16)
      aie.use_lock(%cons, Release, %c1)
      aie.next_bd ^bd5
    ^bd5:
      aie.use_lock(%prod, AcquireGreaterEqual, %c1)
      aie.dma_bd(%buf : memref<256xi32> offset = 80 len = 16)
      aie.use_lock(%cons, Release, %c1)
      aie.next_bd ^bd6
    ^bd6:
      aie.use_lock(%prod, AcquireGreaterEqual, %c1)
      aie.dma_bd(%buf : memref<256xi32> offset = 96 len = 16)
      aie.use_lock(%cons, Release, %c1)
      aie.next_bd ^bd7
    ^bd7:
      aie.use_lock(%prod, AcquireGreaterEqual, %c1)
      aie.dma_bd(%buf : memref<256xi32> offset = 112 len = 16)
      aie.use_lock(%cons, Release, %c1)
      aie.end
    } {repeat_count = 3 : i32}
    aiex.dma_start_task(%fill)
    %peek = aiex.dma_configure_task(%mt, MM2S, 0) {
      aie.use_lock(%cons, AcquireGreaterEqual, %c1)
      aie.dma_bd(%buf : memref<256xi32> offset = 0 len = 16)
      aie.use_lock(%cons, Release, %c1)
      aie.next_bd ^bd1
    ^bd1:
      aie.use_lock(%cons, AcquireGreaterEqual, %c1)
      aie.dma_bd(%buf : memref<256xi32> offset = 16 len = 16)
      aie.use_lock(%cons, Release, %c1)
      aie.next_bd ^bd2
    ^bd2:
      aie.use_lock(%cons, AcquireGreaterEqual, %c1)
      aie.dma_bd(%buf : memref<256xi32> offset = 32 len = 16)
      aie.use_lock(%cons, Release, %c1)
      aie.next_bd ^bd3
    ^bd3:
      aie.use_lock(%cons, AcquireGreaterEqual, %c1)
      aie.dma_bd(%buf : memref<256xi32> offset = 48 len = 16)
      aie.use_lock(%cons, Release, %c1)
      aie.next_bd ^bd4
    ^bd4:
      aie.use_lock(%cons, AcquireGreaterEqual, %c1)
      aie.dma_bd(%buf : memref<256xi32> offset = 64 len = 16)
      aie.use_lock(%cons, Release, %c1)
      aie.next_bd ^bd5
    ^bd5:
      aie.use_lock(%cons, AcquireGreaterEqual, %c1)
      aie.dma_bd(%buf : memref<256xi32> offset = 80 len = 16)
      aie.use_lock(%cons, Release, %c1)
      aie.next_bd ^bd6
    ^bd6:
      aie.use_lock(%cons, AcquireGreaterEqual, %c1)
      aie.dma_bd(%buf : memref<256xi32> offset = 96 len = 16)
      aie.use_lock(%cons, Release, %c1)
      aie.next_bd ^bd7
    ^bd7:
      aie.use_lock(%cons, AcquireGreaterEqual, %c1)
      aie.dma_bd(%buf : memref<256xi32> offset = 112 len = 16)
      aie.use_lock(%cons, Release, %c1)
      aie.end
    }
    aiex.dma_start_task(%peek)
    // expected-error@+1 {{waits on a lock this or a later task releases}}
    %final = aiex.dma_configure_task(%mt, MM2S, 0) {
      aie.use_lock(%cons, AcquireGreaterEqual, %c1)
      aie.dma_bd(%buf : memref<256xi32> offset = 0 len = 16)
      aie.use_lock(%prod, Release, %c1)
      aie.end
    }
    aiex.dma_start_task(%final)
    aiex.dma_start_task(%peek)
  }
}

// -----

// The same, but %final releases a lock the fill never waits on: the fill can
// finish, so reclaim takes its ids behind an idle poll as usual.

aie.device(npu1) {
  %mt = aie.tile(0, 2)
  %buf = aie.buffer(%mt) : memref<256xi32>
  %prod = aie.lock(%mt, 0) {init = 8 : i32}
  %cons = aie.lock(%mt, 1) {init = 0 : i32}
  %other = aie.lock(%mt, 2) {init = 1 : i32}
  aie.runtime_sequence @independent(%arg0: memref<64xi32>) {
    %c1 = arith.constant 1 : i32
    // expected-note@+1 {{the task it waits for}}
    %fill = aiex.dma_configure_task(%mt, S2MM, 0) {
      aie.use_lock(%prod, AcquireGreaterEqual, %c1)
      aie.dma_bd(%buf : memref<256xi32> offset = 0 len = 16)
      aie.use_lock(%cons, Release, %c1)
      aie.next_bd ^bd1
    ^bd1:
      aie.use_lock(%prod, AcquireGreaterEqual, %c1)
      aie.dma_bd(%buf : memref<256xi32> offset = 16 len = 16)
      aie.use_lock(%cons, Release, %c1)
      aie.next_bd ^bd2
    ^bd2:
      aie.use_lock(%prod, AcquireGreaterEqual, %c1)
      aie.dma_bd(%buf : memref<256xi32> offset = 32 len = 16)
      aie.use_lock(%cons, Release, %c1)
      aie.next_bd ^bd3
    ^bd3:
      aie.use_lock(%prod, AcquireGreaterEqual, %c1)
      aie.dma_bd(%buf : memref<256xi32> offset = 48 len = 16)
      aie.use_lock(%cons, Release, %c1)
      aie.next_bd ^bd4
    ^bd4:
      aie.use_lock(%prod, AcquireGreaterEqual, %c1)
      aie.dma_bd(%buf : memref<256xi32> offset = 64 len = 16)
      aie.use_lock(%cons, Release, %c1)
      aie.next_bd ^bd5
    ^bd5:
      aie.use_lock(%prod, AcquireGreaterEqual, %c1)
      aie.dma_bd(%buf : memref<256xi32> offset = 80 len = 16)
      aie.use_lock(%cons, Release, %c1)
      aie.next_bd ^bd6
    ^bd6:
      aie.use_lock(%prod, AcquireGreaterEqual, %c1)
      aie.dma_bd(%buf : memref<256xi32> offset = 96 len = 16)
      aie.use_lock(%cons, Release, %c1)
      aie.next_bd ^bd7
    ^bd7:
      aie.use_lock(%prod, AcquireGreaterEqual, %c1)
      aie.dma_bd(%buf : memref<256xi32> offset = 112 len = 16)
      aie.use_lock(%cons, Release, %c1)
      aie.end
    } {repeat_count = 3 : i32}
    aiex.dma_start_task(%fill)
    %peek = aiex.dma_configure_task(%mt, MM2S, 0) {
      aie.use_lock(%cons, AcquireGreaterEqual, %c1)
      aie.dma_bd(%buf : memref<256xi32> offset = 0 len = 16)
      aie.use_lock(%cons, Release, %c1)
      aie.next_bd ^bd1
    ^bd1:
      aie.use_lock(%cons, AcquireGreaterEqual, %c1)
      aie.dma_bd(%buf : memref<256xi32> offset = 16 len = 16)
      aie.use_lock(%cons, Release, %c1)
      aie.next_bd ^bd2
    ^bd2:
      aie.use_lock(%cons, AcquireGreaterEqual, %c1)
      aie.dma_bd(%buf : memref<256xi32> offset = 32 len = 16)
      aie.use_lock(%cons, Release, %c1)
      aie.next_bd ^bd3
    ^bd3:
      aie.use_lock(%cons, AcquireGreaterEqual, %c1)
      aie.dma_bd(%buf : memref<256xi32> offset = 48 len = 16)
      aie.use_lock(%cons, Release, %c1)
      aie.next_bd ^bd4
    ^bd4:
      aie.use_lock(%cons, AcquireGreaterEqual, %c1)
      aie.dma_bd(%buf : memref<256xi32> offset = 64 len = 16)
      aie.use_lock(%cons, Release, %c1)
      aie.next_bd ^bd5
    ^bd5:
      aie.use_lock(%cons, AcquireGreaterEqual, %c1)
      aie.dma_bd(%buf : memref<256xi32> offset = 80 len = 16)
      aie.use_lock(%cons, Release, %c1)
      aie.next_bd ^bd6
    ^bd6:
      aie.use_lock(%cons, AcquireGreaterEqual, %c1)
      aie.dma_bd(%buf : memref<256xi32> offset = 96 len = 16)
      aie.use_lock(%cons, Release, %c1)
      aie.next_bd ^bd7
    ^bd7:
      aie.use_lock(%cons, AcquireGreaterEqual, %c1)
      aie.dma_bd(%buf : memref<256xi32> offset = 112 len = 16)
      aie.use_lock(%cons, Release, %c1)
      aie.end
    }
    aiex.dma_start_task(%peek)
    // expected-remark@+1 {{channel of an earlier task is idle and takes back its ids 0, 1, 2, 3, 4, 5, 6, 7}}
    %final = aiex.dma_configure_task(%mt, MM2S, 0) {
      aie.use_lock(%cons, AcquireGreaterEqual, %c1)
      aie.dma_bd(%buf : memref<256xi32> offset = 0 len = 16)
      aie.use_lock(%other, Release, %c1)
      aie.end
    }
    aiex.dma_start_task(%final)
    aiex.dma_start_task(%peek)
  }
}
