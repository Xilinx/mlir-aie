//===- dma_out_of_order_invalid.mlir ---------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --verify-diagnostics %s

// dma_start: out_of_order is an S2MM-only channel mode.
module {
  aie.device(npu2) {
    %t = aie.tile(0, 2)
    %b = aie.buffer(%t) : memref<8xi32>
    aie.mem(%t) {
        // expected-error@+1 {{out_of_order is only valid on an S2MM channel}}
        aie.dma_start(MM2S, 0, ^bd0, ^end) { out_of_order }
      ^bd0:
        aie.dma_bd(%b : memref<8xi32> offset = 0 len = 4) { bd_id = 0 : i32 }
        aie.next_bd ^end
      ^end:
        aie.end
    }
  }
}

// -----

// dma: out_of_order is an S2MM-only channel mode.
module {
  aie.device(npu2) {
    %t = aie.tile(0, 2)
    %b = aie.buffer(%t) : memref<8xi32>
    %l0 = aie.lock(%t, 0) { init = 1 : i32 }
    %l1 = aie.lock(%t, 1) { init = 0 : i32 }
    aie.mem(%t) {
      %c1 = arith.constant 1 : i32
      // expected-error@+1 {{out_of_order is only valid on an S2MM channel}}
      %0 = aie.dma(MM2S, 0) { out_of_order } [{
        aie.use_lock(%l0, AcquireGreaterEqual, %c1)
        aie.dma_bd(%b : memref<8xi32> offset = 0 len = 4) { bd_id = 0 : i32 }
        aie.use_lock(%l1, Release, %c1)
      }]
      aie.end
    }
  }
}

// -----

// unsupported device.
module {
  aie.device(xcvc1902) {
    %t = aie.tile(2, 2)
    %b = aie.buffer(%t) : memref<8xi32>
    aie.mem(%t) {
        // expected-error@+1 {{out_of_order S2MM DMA is not supported on this device}}
        aie.dma_start(S2MM, 0, ^bd0, ^end) { out_of_order }
      ^bd0:
        aie.dma_bd(%b : memref<8xi32> offset = 0 len = 4) { bd_id = 0 : i32 }
        aie.next_bd ^end
      ^end:
        aie.end
    }
  }
}

// -----

// dma_start: reject a non-packet-enabled receive BD.
module {
  aie.device(npu2) {
    %t = aie.tile(0, 2)
    %b = aie.buffer(%t) : memref<8xi32>
    aie.mem(%t) {
        aie.dma_start(S2MM, 0, ^bd0, ^end) { out_of_order }
      ^bd0:
        // expected-error@+1 {{out-of-order S2MM receive buffer descriptor must be packet-enabled}}
        aie.dma_bd(%b : memref<8xi32> offset = 0 len = 4) { bd_id = 0 : i32 }
        aie.next_bd ^end
      ^end:
        aie.end
    }
  }
}

// -----

// dma: same packet-enabled requirement on the receive BD.
module {
  aie.device(npu2) {
    %t = aie.tile(0, 2)
    %b = aie.buffer(%t) : memref<8xi32>
    %l0 = aie.lock(%t, 0) { init = 1 : i32 }
    %l1 = aie.lock(%t, 1) { init = 0 : i32 }
    aie.mem(%t) {
      %c1 = arith.constant 1 : i32
      %0 = aie.dma(S2MM, 0) { out_of_order } [{
        aie.use_lock(%l0, AcquireGreaterEqual, %c1)
        // expected-error@+1 {{out-of-order S2MM receive buffer descriptor must be packet-enabled}}
        aie.dma_bd(%b : memref<8xi32> offset = 0 len = 4) { bd_id = 0 : i32 }
        aie.use_lock(%l1, Release, %c1)
      }]
      aie.end
    }
  }
}

// -----

// dma_start: reject missing receive BD (stall).
module {
  aie.device(npu2) {
    %t = aie.tile(0, 2)
    aie.mem(%t) {
        // expected-error@+1 {{out-of-order S2MM channel must have at least one receive buffer descriptor}}
        aie.dma_start(S2MM, 0, ^end, ^end) { out_of_order }
      ^end:
        aie.end
    }
  }
}

// -----

// dma_start: out_of_order_id is a sender-side field.
module {
  aie.device(npu2) {
    %t = aie.tile(0, 2)
    %b = aie.buffer(%t) : memref<8xi32>
    aie.mem(%t) {
        aie.dma_start(S2MM, 0, ^bd0, ^end) { out_of_order }
      ^bd0:
        aie.dma_bd_packet(0, 0)
        // expected-error@+1 {{out_of_order_id belongs on the sender BD}}
        aie.dma_bd(%b : memref<8xi32> offset = 0 len = 4) { bd_id = 0 : i32, out_of_order_id = 3 : i32 }
        aie.next_bd ^end
      ^end:
        aie.end
    }
  }
}

// -----

// dma: same rejection of out_of_order_id on a receive BD.
module {
  aie.device(npu2) {
    %t = aie.tile(0, 2)
    %b = aie.buffer(%t) : memref<8xi32>
    %l0 = aie.lock(%t, 0) { init = 1 : i32 }
    %l1 = aie.lock(%t, 1) { init = 0 : i32 }
    aie.mem(%t) {
      %c1 = arith.constant 1 : i32
      %0 = aie.dma(S2MM, 0) { out_of_order } [{
        aie.use_lock(%l0, AcquireGreaterEqual, %c1)
        aie.dma_bd_packet(0, 0)
        // expected-error@+1 {{out_of_order_id belongs on the sender BD}}
        aie.dma_bd(%b : memref<8xi32> offset = 0 len = 4) { bd_id = 0 : i32, out_of_order_id = 3 : i32 }
        aie.use_lock(%l1, Release, %c1)
      }]
      aie.end
    }
  }
}

// -----

// dma_start: inter-descriptor lock dependency deadlocks.
module {
  aie.device(npu2) {
    %t = aie.tile(0, 2)
    %b0 = aie.buffer(%t) : memref<8xi32>
    %b1 = aie.buffer(%t) : memref<8xi32>
    %lA = aie.lock(%t, 0) { init = 0 : i32 }
    aie.mem(%t) {
        aie.dma_start(S2MM, 0, ^bd0, ^end, repeat_count = 2) { out_of_order }
      ^bd0:
        %c1 = arith.constant 1 : i32
        aie.use_lock(%lA, AcquireGreaterEqual, %c1)
        aie.dma_bd_packet(0, 0)
        // expected-error@+1 {{may not acquire a lock released by another receive BD}}
        aie.dma_bd(%b0 : memref<8xi32> offset = 0 len = 4) { bd_id = 0 : i32 }
        aie.next_bd ^bd1
      ^bd1:
        %c2 = arith.constant 1 : i32
        aie.dma_bd_packet(0, 0)
        aie.dma_bd(%b1 : memref<8xi32> offset = 4 len = 4) { bd_id = 1 : i32 }
        aie.use_lock(%lA, Release, %c2)
        aie.next_bd ^end
      ^end:
        aie.end
    }
  }
}
