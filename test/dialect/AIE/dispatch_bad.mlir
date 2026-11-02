//===- dispatch_bad.mlir ---------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --verify-diagnostics %s

module {
  aie.device(npu2) {
    %mem = aie.tile(0, 1)
    aie.objectfifo.pool @y(%mem) {depth = 2 : i32} : memref<16xi32> {
      aie.objectfifo.segment @s0 {offset = 0 : i32, size = 16 : i32}
    }
    // expected-error@+1 {{only a draining endpoint can dispatch}}
    aie.objectfifo.dma_endpoint @y_in(%mem) fills @y {dispatch = [@a]}
  }
}

// -----

module {
  aie.device(npu2) {
    %mem = aie.tile(0, 1)
    aie.objectfifo.pool @y(%mem) {depth = 2 : i32} : memref<16xi32> {
      aie.objectfifo.segment @s0 {offset = 0 : i32, size = 16 : i32}
    }
    // expected-error@+1 {{dispatch names no destination}}
    aie.objectfifo.dma_endpoint @y_out(%mem) drains @y {dispatch = []}
  }
}

// -----

module {
  aie.device(npu2) {
    %mem = aie.tile(0, 1)
    aie.objectfifo.pool @y(%mem) {depth = 2 : i32} : memref<16xi32> {
      aie.objectfifo.segment @s0 {offset = 0 : i32, size = 16 : i32}
    }
    // expected-error@+1 {{a dispatching endpoint carries a header per turn in `dispatchPackets`, not one `packet`}}
    aie.objectfifo.dma_endpoint @y_out(%mem) drains @y {dispatch = [@a], packet = #aie.packet_info<pkt_id = 1>}
  }
}

// -----

module {
  aie.device(npu2) {
    %mem = aie.tile(0, 1)
    aie.objectfifo.pool @y(%mem) {depth = 2 : i32, repeatCount = 2 : i32} : memref<16xi32> {
      aie.objectfifo.segment @s0 {offset = 0 : i32, size = 16 : i32}
    }
    // expected-error@+1 {{cannot dispatch a pool that repeats its objects}}
    aie.objectfifo.dma_endpoint @y_out(%mem) drains @y {dispatch = [@a, @b]}
  }
}

// -----

// Two objects and three turns line up again after three passes, so a bounded
// chain runs a multiple of three.
module {
  aie.device(npu2) {
    %mem = aie.tile(0, 1)
    aie.objectfifo.pool @y(%mem) {depth = 2 : i32} : memref<16xi32> {
      aie.objectfifo.segment @s0 {offset = 0 : i32, size = 16 : i32}
    }
    // expected-error@+1 {{iterCount 4 is not a whole number of dispatch rounds (3 passes over the pool)}}
    aie.objectfifo.dma_endpoint @y_out(%mem) drains @y {dispatch = [@a, @b, @c], iterCount = 4 : i32}
  }
}

// -----

module {
  aie.device(npu2) {
    %mem = aie.tile(0, 1)
    aie.objectfifo.pool @y(%mem) {depth = 2 : i32} : memref<16xi32> {
      aie.objectfifo.segment @s0 {offset = 0 : i32, size = 16 : i32}
    }
    // expected-error@+1 {{dispatchPackets needs one header per dispatch turn}}
    aie.objectfifo.dma_endpoint @y_out(%mem) drains @y {dispatch = [@a, @b], dispatchPackets = [#aie.packet_info<pkt_id = 1>]}
  }
}
