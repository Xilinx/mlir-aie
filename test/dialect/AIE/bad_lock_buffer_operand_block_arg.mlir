//===- bad_lock_buffer_operand_block_arg.mlir ------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --verify-diagnostics %s

// A lock operand is an index and a buffer operand a memref, so a block argument
// parses where an aie.lock or aie.buffer goes. It has no defining op, and the
// verifiers that look at the operand's defining op reject it instead of
// dereferencing the missing op. The generic form is the way to write a device
// with a block argument.

// A buffer descriptor's buffer.
"aie.device"() ({
^bb0(%buf: memref<16xi32>):
  %t = aie.tile(1, 3)
  %l = aie.lock(%t, 0)
  %c1 = arith.constant 1 : i32
  %m = aie.mem(%t) {
    %d = aie.dma_start(MM2S, 0, ^bb1, ^bb2)
  ^bb1:
    aie.use_lock(%l, AcquireGreaterEqual, %c1)
    // expected-error @+1 {{BDs only support BufferOp or ExternalBufferOp operands.}}
    aie.dma_bd(%buf : memref<16xi32> offset = 0 len = 16)
    aie.use_lock(%l, Release, %c1)
    aie.next_bd ^bb2
  ^bb2:
    aie.end
  }
  "aie.end"() : () -> ()
}) {device = 4 : i32} : () -> ()

// -----

// The lock a DMA block uses.
"aie.device"() ({
^bb0(%lock: index):
  %t = aie.tile(1, 3)
  %buf = aie.buffer(%t) : memref<16xi32>
  %c1 = arith.constant 1 : i32
  %m = aie.mem(%t) {
    %d = aie.dma_start(MM2S, 0, ^bb1, ^bb2)
  ^bb1:
    // expected-error @+1 {{'aie.use_lock' op lock operand is not an aie.lock}}
    aie.use_lock(%lock, AcquireGreaterEqual, %c1)
    aie.dma_bd(%buf : memref<16xi32> offset = 0 len = 16)
    aie.use_lock(%lock, Release, %c1)
    aie.next_bd ^bb2
  ^bb2:
    aie.end
  }
  "aie.end"() : () -> ()
}) {device = 4 : i32} : () -> ()

// -----

// A memtile checks that the BDs reachable from channels 4 and 5 use only its
// own buffers and locks, before the BDs and use_locks are verified.
"aie.device"() ({
^bb0(%buf: memref<16xi32>):
  %t = aie.tile(1, 1)
  %l = aie.lock(%t, 0)
  %c1 = arith.constant 1 : i32
  %m = aie.memtile_dma(%t) {
    %d = aie.dma_start(MM2S, 4, ^bb1, ^bb2)
  ^bb1:
    aie.use_lock(%l, AcquireGreaterEqual, %c1)
    // expected-error @+1 {{BDs only support BufferOp or ExternalBufferOp operands.}}
    aie.dma_bd(%buf : memref<16xi32> offset = 0 len = 16)
    aie.use_lock(%l, Release, %c1)
    aie.next_bd ^bb2
  ^bb2:
    aie.end
  }
  "aie.end"() : () -> ()
}) {device = 4 : i32} : () -> ()

// -----

"aie.device"() ({
^bb0(%lock: index):
  %t = aie.tile(1, 1)
  %buf = aie.buffer(%t) : memref<16xi32>
  %c1 = arith.constant 1 : i32
  %m = aie.memtile_dma(%t) {
    %d = aie.dma_start(MM2S, 4, ^bb1, ^bb2)
  ^bb1:
    // expected-error @+1 {{'aie.use_lock' op lock operand is not an aie.lock}}
    aie.use_lock(%lock, AcquireGreaterEqual, %c1)
    aie.dma_bd(%buf : memref<16xi32> offset = 0 len = 16)
    aie.use_lock(%lock, Release, %c1)
    aie.next_bd ^bb2
  ^bb2:
    aie.end
  }
  "aie.end"() : () -> ()
}) {device = 4 : i32} : () -> ()

// -----

// On AIE1 each use_lock in a DMA block compares its lock with its siblings',
// so a valid use_lock is verified against a sibling's missing lock.
"aie.device"() ({
^bb0(%lock: index):
  %t = aie.tile(1, 3)
  %l = aie.lock(%t, 0)
  %buf = aie.buffer(%t) : memref<16xi32>
  %c1 = arith.constant 1 : i32
  %c0 = arith.constant 0 : i32
  %m = aie.mem(%t) {
    %d = aie.dma_start(MM2S, 0, ^bb1, ^bb2)
  ^bb1:
    aie.use_lock(%l, Acquire, %c1)
    aie.dma_bd(%buf : memref<16xi32> offset = 0 len = 16)
    // expected-error @+1 {{'aie.use_lock' op lock operand is not an aie.lock}}
    aie.use_lock(%lock, Release, %c0)
    aie.next_bd ^bb2
  ^bb2:
    aie.end
  }
  "aie.end"() : () -> ()
}) {device = 1 : i32} : () -> ()

// -----

// The same, with a sibling's lock defined by an op that is not an aie.lock.
module {
  aie.device(xcvc1902) {
    %t = aie.tile(1, 3)
    %l = aie.lock(%t, 0)
    %buf = aie.buffer(%t) : memref<16xi32>
    %c1 = arith.constant 1 : i32
    %c0 = arith.constant 0 : i32
    %m = aie.mem(%t) {
      %notlock = arith.constant 0 : index
      %d = aie.dma_start(MM2S, 0, ^bb1, ^bb2)
    ^bb1:
      aie.use_lock(%l, Acquire, %c1)
      aie.dma_bd(%buf : memref<16xi32> offset = 0 len = 16)
      // expected-error @+1 {{'aie.use_lock' op lock operand is not an aie.lock}}
      aie.use_lock(%notlock, Release, %c0)
      aie.next_bd ^bb2
    ^bb2:
      aie.end
    }
  }
}

// -----

// The lock of a runtime DMA task's buffer descriptor.
"aie.device"() ({
^bb0(%lock: index):
  %t = aie.tile(0, 0)
  aie.shim_dma_allocation @a(%t, MM2S, 0)
  aie.runtime_sequence @s(%arg0: memref<16xi32>) {
    %c1 = arith.constant 1 : i32
    %task = aiex.dma_configure_task_for @a {
      // expected-error @+1 {{'aie.use_lock' op lock operand is not an aie.lock}}
      aie.use_lock(%lock, AcquireGreaterEqual, %c1)
      aie.dma_bd(%arg0 : memref<16xi32> offset = 0 len = 16)
      aie.use_lock(%lock, Release, %c1)
      aie.end
    }
  }
  "aie.end"() : () -> ()
}) {device = 4 : i32} : () -> ()

// -----

// The lock a runtime sequence sets.
"aie.device"() ({
^bb0(%lock: index):
  aie.runtime_sequence() {
    %c1 = arith.constant 1 : i32
    // expected-error @+1 {{'aiex.set_lock' op lock operand is not an aie.lock}}
    aiex.set_lock(%lock, %c1)
  }
  "aie.end"() : () -> ()
}) {device = 4 : i32} : () -> ()

// -----

// A tile DMA cannot reach an external buffer.
module {
  aie.device(npu1) {
    %t = aie.tile(1, 3)
    %l = aie.lock(%t, 0)
    %buf = aie.external_buffer : memref<16xi32>
    %c1 = arith.constant 1 : i32
    %m = aie.mem(%t) {
      %d = aie.dma_start(MM2S, 0, ^bb1, ^bb2)
    ^bb1:
      aie.use_lock(%l, AcquireGreaterEqual, %c1)
      // expected-error @+1 {{Core tile DMAs can only access a buffer in the same tile.}}
      aie.dma_bd(%buf : memref<16xi32> offset = 0 len = 16)
      aie.use_lock(%l, Release, %c1)
      aie.next_bd ^bb2
    ^bb2:
      aie.end
    }
  }
}

// -----

// Nor can a memtile's local-only channels.
module {
  aie.device(npu1) {
    %t = aie.tile(1, 1)
    %l = aie.lock(%t, 0)
    // expected-note @+1 {{buffer}}
    %buf = aie.external_buffer : memref<16xi32>
    %c1 = arith.constant 1 : i32
    %m = aie.memtile_dma(%t) {
      // expected-note @+1 {{channel}}
      %d = aie.dma_start(MM2S, 4, ^bb1, ^bb2)
    ^bb1:
      aie.use_lock(%l, AcquireGreaterEqual, %c1)
      // expected-error @+1 {{is reachable from DMA channel 4 and attempts to access a non-local buffer}}
      aie.dma_bd(%buf : memref<16xi32> offset = 0 len = 16)
      aie.use_lock(%l, Release, %c1)
      aie.next_bd ^bb2
    ^bb2:
      aie.end
    }
  }
}

// -----

// Nor can a memtile's other channels: only a shim DMA reaches an external
// buffer.
module {
  aie.device(npu1) {
    %t = aie.tile(1, 1)
    %l = aie.lock(%t, 0)
    %buf = aie.external_buffer : memref<16xi32>
    %c1 = arith.constant 1 : i32
    %m = aie.memtile_dma(%t) {
      %d = aie.dma_start(MM2S, 0, ^bb1, ^bb2)
    ^bb1:
      aie.use_lock(%l, AcquireGreaterEqual, %c1)
      // expected-error @+1 {{Memtile and core tile DMAs can only access an aie.buffer, not an aie.external_buffer.}}
      aie.dma_bd(%buf : memref<16xi32> offset = 0 len = 16)
      aie.use_lock(%l, Release, %c1)
      aie.next_bd ^bb2
    ^bb2:
      aie.end
    }
  }
}
