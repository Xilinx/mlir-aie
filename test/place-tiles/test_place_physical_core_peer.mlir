//===- test_place_physical_core_peer.mlir ----------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --verify-diagnostics --pass-pipeline='builtin.module(aie-place-tiles,aie.device(aie-objectFifo-stateful-transform))' %s | FileCheck %s

// A logical core fed by a physical core is its compute peer, as two logical
// cores are: placed where the two share memory, the fifo takes no DMA
// channel on either, and the sequential placer pulls the logical core there
// when either side has no channel to spare.

// %l takes @a and @b from the shim, both of its receiving channels, so @f
// must come by shared memory: %l goes beside %p, which keeps @f's buffers.
// CHECK-LABEL: module @physical_core_peer
// CHECK: %[[P:.*]] = aie.tile(0, 2)
// CHECK: aie.buffer(%[[P]]) {sym_name = "f__buf0"}
// CHECK-NOT: f__cons__buf
// CHECK: aie.core(%tile_0_3)

// %p sends @o1 and @o2 to the shim on both of its sending channels, so @f
// cannot leave it by DMA: %l goes to (0, 3), whose memory %p reaches.
// CHECK-LABEL: module @physical_core_peer_sends_full
// CHECK: %[[L:.*]] = aie.tile(0, 3)
// CHECK: aie.buffer(%[[L]]) {sym_name = "f__buf0"}
// CHECK-NOT: f__cons__buf

// %p's only neighbor sharing its memory holds another physical core, so %l
// has nowhere to take @f by shared memory: the placer says so rather than
// that %l's tile lacks DMA channels.

module @physical_core_peer {
  aie.device(npu2) {
    %p = aie.tile(0, 2)
    %s = aie.logical_tile<ShimNOCTile>(?, ?)
    %l = aie.logical_tile<CoreTile>(?, ?)
    aie.objectfifo @f(%p, {%l}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @a(%s, {%l}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @b(%s, {%l}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.core(%p) {
      %x = aie.objectfifo.acquire @f(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f(Produce, 1)
      aie.end
    }
    aie.core(%l) {
      %x = aie.objectfifo.acquire @f(Consume, 1) : memref<16xi32>
      %y = aie.objectfifo.acquire @a(Consume, 1) : memref<16xi32>
      %z = aie.objectfifo.acquire @b(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f(Consume, 1)
      aie.objectfifo.release @a(Consume, 1)
      aie.objectfifo.release @b(Consume, 1)
      aie.end
    }
  }
}

// -----

module @physical_core_peer_sends_full {
  aie.device(npu2) {
    %p = aie.tile(1, 3)
    %s = aie.logical_tile<ShimNOCTile>(?, ?)
    %l = aie.logical_tile<CoreTile>(?, ?)
    aie.objectfifo @f(%p, {%l}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @o1(%p, {%s}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @o2(%p, {%s}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.core(%p) {
      %x = aie.objectfifo.acquire @f(Produce, 1) : memref<16xi32>
      %y = aie.objectfifo.acquire @o1(Produce, 1) : memref<16xi32>
      %z = aie.objectfifo.acquire @o2(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f(Produce, 1)
      aie.objectfifo.release @o1(Produce, 1)
      aie.objectfifo.release @o2(Produce, 1)
      aie.end
    }
    aie.core(%l) {
      %x = aie.objectfifo.acquire @f(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f(Consume, 1)
      aie.end
    }
  }
}

// -----

module @physical_core_peer_boxed_in {
  aie.device(npu2_1col) {
    // expected-note@+1 {{compute-peer producer peer placed at (0, 2)}}
    %p = aie.tile(0, 2)
    %q = aie.tile(0, 3)
    %s = aie.logical_tile<ShimNOCTile>(?, ?)
    // expected-error@+3 {{no available compute tiles for placement (compute-peer DMA budget unsatisfiable)}}
    // expected-note@+2 {{to fix, pin this LTO to a position that is shared-L1 with the placed compute-peers above}}
    // expected-note@+1 {{the sequential placer places tiles one at a time and never moves a placed peer}}
    %l = aie.logical_tile<CoreTile>(?, ?)
    aie.objectfifo @f(%p, {%l}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @a(%s, {%l}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @b(%s, {%l}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.core(%p) {
      %x = aie.objectfifo.acquire @f(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f(Produce, 1)
      aie.end
    }
    aie.core(%q) {
      aie.end
    }
    aie.core(%l) {
      %x = aie.objectfifo.acquire @f(Consume, 1) : memref<16xi32>
      %y = aie.objectfifo.acquire @a(Consume, 1) : memref<16xi32>
      %z = aie.objectfifo.acquire @b(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f(Consume, 1)
      aie.objectfifo.release @a(Consume, 1)
      aie.objectfifo.release @b(Consume, 1)
      aie.end
    }
  }
}
