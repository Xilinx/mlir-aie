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
//
// Taking the order out of the channel takes the chain that would carry it
// round with it, so the channel is armed for a count of objects instead. A
// fifo with an iter_count receives that many passes and no more. One without
// runs on, and is armed for as many objects as one arming holds.

// RUN: aie-opt --split-input-file --verify-diagnostics --aie-objectfifo-split --aie-assign-packet-ids --aie-objectfifo-allocate --aie-objectfifo-lower-cores --aie-objectfifo-lower-dmas --aie-assign-lock-ids --aie-assign-buffer-addresses --aie-assign-bd-ids %s | FileCheck %s

// The header has to reach the receiving DMA, which reads the buffer out of it.
// CHECK-LABEL: @out_of_order
// CHECK:       aie.packet_flow(0)
// CHECK:       {keep_pkt_header = true}
// The sender names a buffer per object ...
// CHECK:       aie.dma_start(MM2S, 0, {{.*}}, repeat_count = 3)
// CHECK:       aie.dma_bd({{.*}}) {bd_id = 0 : i32, next_bd_id = 1 : i32, out_of_order_id = 0 : i32}
// CHECK:       aie.dma_bd({{.*}}) {bd_id = 1 : i32, out_of_order_id = 1 : i32}
// ... and the receiver holds the ids those names select, armed for the whole
// count of objects the four passes bring rather than for a chain.
// CHECK:       aie.dma_start(S2MM, 0, {{.*}}, repeat_count = 7) {out_of_order}
// CHECK:       aie.dma_bd({{.*}}) {bd_id = 0 : i32
// CHECK-NOT:   out_of_order_id
// CHECK:       aie.dma_bd({{.*}}) {bd_id = 1 : i32
module @out_of_order {
  aie.device(npu2) {
    %tile_0_2 = aie.tile(0, 2)
    %tile_0_5 = aie.tile(0, 5)
    aie.objectfifo @of(%tile_0_2, {%tile_0_5}, 2 : i32) { transport = #aie.transport<dma, packet = #aie.packet_info<>, out_of_order = true>, iter_count = 4 : i32} : !aie.objectfifo<memref<16xi32>>
    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4 step %c1 {
        %1 = aie.objectfifo.acquire @of(Produce, 1) : memref<16xi32>
        aie.objectfifo.release @of(Produce, 1)
      }
      aie.end
    }
    %core_0_5 = aie.core(%tile_0_5) {
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4 step %c1 {
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
    // expected-error@+1 {{`out_of_order` requires a packet transport}}
    aie.objectfifo @of(%tile_0_2, {%tile_0_5}, 2 : i32) {transport = #aie.transport<dma, out_of_order = true>} : !aie.objectfifo<memref<16xi32>>
  }
}

// -----

// A fifo that says nothing about how many objects it receives goes on
// receiving them, so its channel is armed for as many as one arming holds.
// CHECK-LABEL: @runs_on
// CHECK:       aie.dma_start(S2MM, 0, {{.*}}, repeat_count = 255) {out_of_order}
module @runs_on {
  aie.device(npu2) {
    %tile_0_2 = aie.tile(0, 2)
    %tile_0_5 = aie.tile(0, 5)
    aie.objectfifo @of(%tile_0_2, {%tile_0_5}, 2 : i32) { transport = #aie.transport<dma, packet = #aie.packet_info<>, out_of_order = true>} : !aie.objectfifo<memref<16xi32>>
  }
}

// -----

// One arming only goes so far: a fifo that names more passes than it holds
// asks for a count the queue cannot take.
module @beyond_one_arming {
  aie.device(npu2) {
    %tile_0_2 = aie.tile(0, 2)
    %tile_0_5 = aie.tile(0, 5)
    // expected-error@+1 {{objects out of order, but a channel arms for at most}}
    aie.objectfifo @of(%tile_0_2, {%tile_0_5}, 2 : i32) { transport = #aie.transport<dma, packet = #aie.packet_info<>, out_of_order = true>, iter_count = 200 : i32} : !aie.objectfifo<memref<16xi32>>
  }
}
