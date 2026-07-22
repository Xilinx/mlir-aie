//===- dma_channel_reset_rearm.mlir ----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt -split-input-file -verify-diagnostics --aie-verify-runtime-rearm %s

// A dma_channel_reset whose objectFIFO locks are all re-armed with set_lock in
// the same runtime sequence is accepted (npu2 / AIE2P core tile).
module {
  aie.device(npu2) {
    %t = aie.tile(0, 2)
    %buf = aie.buffer(%t) : memref<16xi32>
    %prod = aie.lock(%t, 0) {init = 1 : i32, sym_name = "fifo_prod_lock_0"}
    %cons = aie.lock(%t, 1) {init = 0 : i32, sym_name = "fifo_cons_lock_0"}
    %mem = aie.mem(%t) {
      %dma = aie.dma_start(S2MM, 0, ^bd0, ^end)
    ^bd0:
      %c1 = arith.constant 1 : i32
      aie.use_lock(%prod, AcquireGreaterEqual, %c1)
      aie.dma_bd(%buf : memref<16xi32> offset = 0 len = 16)
      aie.use_lock(%cons, Release, %c1)
      aie.next_bd ^bd0
    ^end:
      aie.end
    }
    aie.runtime_sequence() {
      aiex.dma_channel_reset(%t, S2MM, 0)
      aiex.set_lock(%prod, 1)
      aiex.set_lock(%cons, 0)
    }
  }
}

// -----

// A reset with no re-arm at all is rejected; every bound lock is a frozen
// counter and gets a note.
module {
  aie.device(npu2) {
    %t = aie.tile(0, 2)
    %buf = aie.buffer(%t) : memref<16xi32>
    // expected-note @+1 {{this lock is bound to the reset channel and is not re-armed}}
    %prod = aie.lock(%t, 0) {init = 1 : i32, sym_name = "fifo_prod_lock_0"}
    // expected-note @+1 {{this lock is bound to the reset channel and is not re-armed}}
    %cons = aie.lock(%t, 1) {init = 0 : i32, sym_name = "fifo_cons_lock_0"}
    %mem = aie.mem(%t) {
      %dma = aie.dma_start(S2MM, 0, ^bd0, ^end)
    ^bd0:
      %c1 = arith.constant 1 : i32
      aie.use_lock(%prod, AcquireGreaterEqual, %c1)
      aie.dma_bd(%buf : memref<16xi32> offset = 0 len = 16)
      aie.use_lock(%cons, Release, %c1)
      aie.next_bd ^bd0
    ^end:
      aie.end
    }
    aie.runtime_sequence() {
      // expected-error @+1 {{resets a DMA channel whose objectFIFO lock is never re-armed}}
      aiex.dma_channel_reset(%t, S2MM, 0)
    }
  }
}

// -----

// A partial re-arm is still a deadlock: prod is re-armed but cons is not, so the
// reset is rejected and only the frozen lock (cons) gets a note.
module {
  aie.device(npu2) {
    %t = aie.tile(0, 2)
    %buf = aie.buffer(%t) : memref<16xi32>
    %prod = aie.lock(%t, 0) {init = 1 : i32, sym_name = "fifo_prod_lock_0"}
    // expected-note @+1 {{this lock is bound to the reset channel and is not re-armed}}
    %cons = aie.lock(%t, 1) {init = 0 : i32, sym_name = "fifo_cons_lock_0"}
    %mem = aie.mem(%t) {
      %dma = aie.dma_start(S2MM, 0, ^bd0, ^end)
    ^bd0:
      %c1 = arith.constant 1 : i32
      aie.use_lock(%prod, AcquireGreaterEqual, %c1)
      aie.dma_bd(%buf : memref<16xi32> offset = 0 len = 16)
      aie.use_lock(%cons, Release, %c1)
      aie.next_bd ^bd0
    ^end:
      aie.end
    }
    aie.runtime_sequence() {
      // expected-error @+1 {{resets a DMA channel whose objectFIFO lock is never re-armed}}
      aiex.dma_channel_reset(%t, S2MM, 0)
      aiex.set_lock(%prod, 1)
    }
  }
}

// -----

// A channel that carries no objectFIFO locks (a raw DMA) has nothing to re-arm,
// so its reset is accepted -- the pass must not false-positive here.
module {
  aie.device(npu2) {
    %t = aie.tile(0, 2)
    %buf = aie.buffer(%t) : memref<16xi32>
    %mem = aie.mem(%t) {
      %dma = aie.dma_start(S2MM, 0, ^bd0, ^end)
    ^bd0:
      aie.dma_bd(%buf : memref<16xi32> offset = 0 len = 16)
      aie.next_bd ^bd0
    ^end:
      aie.end
    }
    aie.runtime_sequence() {
      aiex.dma_channel_reset(%t, S2MM, 0)
    }
  }
}

// -----

// The check is target-model independent: the same design on npu1 (AIE2) is
// accepted when both bound locks are re-armed.
module {
  aie.device(npu1) {
    %t = aie.tile(0, 2)
    %buf = aie.buffer(%t) : memref<16xi32>
    %prod = aie.lock(%t, 0) {init = 1 : i32, sym_name = "fifo_prod_lock_0"}
    %cons = aie.lock(%t, 1) {init = 0 : i32, sym_name = "fifo_cons_lock_0"}
    %mem = aie.mem(%t) {
      %dma = aie.dma_start(S2MM, 0, ^bd0, ^end)
    ^bd0:
      %c1 = arith.constant 1 : i32
      aie.use_lock(%prod, AcquireGreaterEqual, %c1)
      aie.dma_bd(%buf : memref<16xi32> offset = 0 len = 16)
      aie.use_lock(%cons, Release, %c1)
      aie.next_bd ^bd0
    ^end:
      aie.end
    }
    aie.runtime_sequence() {
      aiex.dma_channel_reset(%t, S2MM, 0)
      aiex.set_lock(%prod, 1)
      aiex.set_lock(%cons, 0)
    }
  }
}

// -----

// The binding also resolves through a mem tile (aie.memtile_dma). Here the mem
// tile channel's lock is not re-armed, so the reset is rejected.
module {
  aie.device(npu2) {
    %mt = aie.tile(0, 1)
    %buf = aie.buffer(%mt) : memref<16xi32>
    // expected-note @+1 {{this lock is bound to the reset channel and is not re-armed}}
    %prod = aie.lock(%mt, 0) {init = 1 : i32, sym_name = "mtfifo_prod_lock_0"}
    %cons = aie.lock(%mt, 1) {init = 0 : i32, sym_name = "mtfifo_cons_lock_0"}
    %memtile = aie.memtile_dma(%mt) {
      %dma = aie.dma_start(S2MM, 0, ^bd0, ^end)
    ^bd0:
      %c1 = arith.constant 1 : i32
      aie.use_lock(%prod, AcquireGreaterEqual, %c1)
      aie.dma_bd(%buf : memref<16xi32> offset = 0 len = 16)
      aie.next_bd ^bd0
    ^end:
      aie.end
    }
    aie.runtime_sequence() {
      // expected-error @+1 {{resets a DMA channel whose objectFIFO lock is never re-armed}}
      aiex.dma_channel_reset(%mt, S2MM, 0)
      aiex.set_lock(%cons, 0)
    }
  }
}

// -----

// A re-arm written directly to the lock register (npu.write32 at the lock's
// local address) is recognized, so a lowered-style re-arm is not a false
// positive. Lock ids 0 and 1 on (0, 2) are at local addresses 126976 / 126992.
module {
  aie.device(npu2) {
    %t = aie.tile(0, 2)
    %buf = aie.buffer(%t) : memref<16xi32>
    %prod = aie.lock(%t, 0) {init = 1 : i32}
    %cons = aie.lock(%t, 1) {init = 0 : i32}
    %mem = aie.mem(%t) {
      %dma = aie.dma_start(S2MM, 0, ^bd0, ^end)
    ^bd0:
      %c1 = arith.constant 1 : i32
      aie.use_lock(%prod, AcquireGreaterEqual, %c1)
      aie.dma_bd(%buf : memref<16xi32> offset = 0 len = 16)
      aie.use_lock(%cons, Release, %c1)
      aie.next_bd ^bd0
    ^end:
      aie.end
    }
    aie.runtime_sequence() {
      aiex.dma_channel_reset(%t, S2MM, 0)
      %a0 = arith.constant 126976 : i32
      %a1 = arith.constant 126992 : i32
      %v1 = arith.constant 1 : i32
      %v0 = arith.constant 0 : i32
      aiex.npu.write32(%a0, %v1) {column = 0 : i32, row = 2 : i32} : i32, i32
      aiex.npu.write32(%a1, %v0) {column = 0 : i32, row = 2 : i32} : i32, i32
    }
  }
}

// -----

// Re-arms are collected module-wide: a reset in one runtime sequence whose locks
// are re-armed in another sequence is accepted (no false positive).
module {
  aie.device(npu2) {
    %t = aie.tile(0, 2)
    %buf = aie.buffer(%t) : memref<16xi32>
    %prod = aie.lock(%t, 0) {init = 1 : i32}
    %cons = aie.lock(%t, 1) {init = 0 : i32}
    %mem = aie.mem(%t) {
      %dma = aie.dma_start(S2MM, 0, ^bd0, ^end)
    ^bd0:
      %c1 = arith.constant 1 : i32
      aie.use_lock(%prod, AcquireGreaterEqual, %c1)
      aie.dma_bd(%buf : memref<16xi32> offset = 0 len = 16)
      aie.use_lock(%cons, Release, %c1)
      aie.next_bd ^bd0
    ^end:
      aie.end
    }
    aie.runtime_sequence @reset() {
      aiex.dma_channel_reset(%t, S2MM, 0)
    }
    aie.runtime_sequence @rearm() {
      aiex.set_lock(%prod, 1)
      aiex.set_lock(%cons, 0)
    }
  }
}
