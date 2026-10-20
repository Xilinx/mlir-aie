//===- out_of_order_test.mlir -----------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// An out-of-order ObjectFifo receives each object into the buffer its sender
// names in the packet header, rather than walking its pool in a fixed chain.
// The receiving channel then carries no order of its own, which is what lets
// several fifos share one S2MM channel.
//
// The sender names the buffer through out_of_order_id, a 6-bit header field of
// its own, so a pool costs no ids out of the 5-bit routing space.

// RUN: aie-opt --split-input-file --verify-diagnostics --aie-objectfifo-split --aie-objectfifo-allocate --aie-objectfifo-lower-cores --aie-objectfifo-lower-dmas --aie-assign-lock-ids --aie-assign-buffer-addresses --aie-assign-bd-ids %s | FileCheck %s

// The header has to reach the receiving DMA, which reads the buffer out of it.
// CHECK-LABEL: @out_of_order
// CHECK:       aie.packet_flow(0)
// CHECK:       {keep_pkt_header = true}
// The sender names a buffer per object ...
// CHECK:       aie.dma_start(MM2S
// CHECK:       aie.dma_bd({{.*}}) {bd_id = 0 : i32, next_bd_id = 1 : i32, out_of_order_id = 0 : i32}
// CHECK:       aie.dma_bd({{.*}}) {bd_id = 1 : i32, next_bd_id = 0 : i32, out_of_order_id = 1 : i32}
// ... and the receiver holds the ids those names select.
// CHECK:       aie.dma_start(S2MM, 0, {{.*}}) {out_of_order}
// CHECK:       aie.dma_bd({{.*}}) {bd_id = 0 : i32
// CHECK-NOT:   out_of_order_id
// CHECK:       aie.dma_bd({{.*}}) {bd_id = 1 : i32
module @out_of_order {
  aie.device(npu2) {
    %tile_0_2 = aie.tile(0, 2)
    %tile_0_5 = aie.tile(0, 5)
    aie.objectfifo @of(%tile_0_2, {%tile_0_5}, 2 : i32) {packet, out_of_order} : !aie.objectfifo<memref<16xi32>>
    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %1 = aie.objectfifo.acquire @of(Produce, 1) : memref<16xi32>
        aie.objectfifo.release @of(Produce, 1)
      }
      aie.end
    }
    %core_0_5 = aie.core(%tile_0_5) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %1 = aie.objectfifo.acquire @of(Consume, 1) : memref<16xi32>
        aie.objectfifo.release @of(Consume, 1)
      }
      aie.end
    }
  }
}

// -----

// Nothing separates the streams on a shared channel without a packet route.
module @needs_packet {
  aie.device(npu2) {
    %tile_0_2 = aie.tile(0, 2)
    %tile_0_5 = aie.tile(0, 5)
    // expected-error@+1 {{out_of_order names each object's buffer in the packet header, so it needs a packet objectfifo}}
    aie.objectfifo @of(%tile_0_2, {%tile_0_5}, 2 : i32) {out_of_order} : !aie.objectfifo<memref<16xi32>>
  }
}

// -----

// A shim's descriptors come from the runtime sequence, which names no buffer.
module @no_shim_sender {
  aie.device(npu2) {
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)
    // expected-error@+1 {{sends from a shim, whose descriptors the runtime sequence writes; out_of_order needs a sender that names the buffer each object lands in}}
    aie.objectfifo @of(%shim_noc_tile_0_0, {%tile_0_2}, 2 : i32) {packet, out_of_order} : !aie.objectfifo<memref<16xi32>>
  }
}
