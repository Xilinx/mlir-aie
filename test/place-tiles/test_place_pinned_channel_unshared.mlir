//===- test_place_pinned_channel_unshared.mlir -----------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Two circuit-switched fifos one core sends on pinned MM2S 0 cannot share it
// wherever the core goes: both placers report it and fail. The error lands
// on the structurally first of the two and names the other, whichever is
// declared first.

// RUN: aie-opt --split-input-file --verify-diagnostics --aie-place-tiles %s
// RUN: aie-opt --split-input-file --verify-diagnostics --aie-place-tiles='placer=sa_placer sa-seed=1' %s

module @unshared_pinned_send {
  aie.device(npu2) {
    %t = aie.logical_tile<CoreTile>(?, ?)
    %u = aie.logical_tile<CoreTile>(?, ?)
    %v = aie.logical_tile<CoreTile>(?, ?)
    // expected-error@+1 {{shares MM2S channel 0 with @b, but a shared channel carries both at once, which only packet-switched routes tell apart}}
    aie.objectfifo @a(%t, {%u}, 2 : i32) {transport = #aie.transport<dma>, prod_port = #aie.end_port<DMA : 0>} : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @b(%t, {%v}, 2 : i32) {transport = #aie.transport<dma>, prod_port = #aie.end_port<DMA : 0>} : !aie.objectfifo<memref<16xi32>>
    aie.core(%t) {
      %a = aie.objectfifo.acquire @a(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @a(Produce, 1)
      %b = aie.objectfifo.acquire @b(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @b(Produce, 1)
      aie.end
    }
    aie.core(%u) {
      %a = aie.objectfifo.acquire @a(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @a(Consume, 1)
      aie.end
    }
    aie.core(%v) {
      %b = aie.objectfifo.acquire @b(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @b(Consume, 1)
      aie.end
    }
  }
}

// -----

module @unshared_pinned_send_declared_backwards {
  aie.device(npu2) {
    %t = aie.logical_tile<CoreTile>(?, ?)
    %u = aie.logical_tile<CoreTile>(?, ?)
    %v = aie.logical_tile<CoreTile>(?, ?)
    aie.objectfifo @b(%t, {%v}, 2 : i32) {transport = #aie.transport<dma>, prod_port = #aie.end_port<DMA : 0>} : !aie.objectfifo<memref<16xi32>>
    // expected-error@+1 {{shares MM2S channel 0 with @b, but a shared channel carries both at once, which only packet-switched routes tell apart}}
    aie.objectfifo @a(%t, {%u}, 2 : i32) {transport = #aie.transport<dma>, prod_port = #aie.end_port<DMA : 0>} : !aie.objectfifo<memref<16xi32>>
    aie.core(%t) {
      %a = aie.objectfifo.acquire @a(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @a(Produce, 1)
      %b = aie.objectfifo.acquire @b(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @b(Produce, 1)
      aie.end
    }
    aie.core(%u) {
      %a = aie.objectfifo.acquire @a(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @a(Consume, 1)
      aie.end
    }
    aie.core(%v) {
      %b = aie.objectfifo.acquire @b(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @b(Consume, 1)
      aie.end
    }
  }
}
