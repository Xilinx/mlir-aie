// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: not aie-opt --split-input-file --aie-objectFifo-stateful-transform="skip-verify=true" %s 2>&1 | FileCheck %s

// CHECK: error: {{.*}}pinned MM2S DMA channel 0 is out of range or already in use on this tile
module @circuits {
  aie.device(npu2) {
    %src = aie.tile(0, 2)
    %a = aie.tile(0, 4)
    %b = aie.tile(0, 5)
    aie.objectfifo @a(%src, {%a}, 2 : i32) {transport = #aie.transport<dma>, prod_port = #aie.end_port<DMA : 0>} : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @b(%src, {%b}, 2 : i32) {transport = #aie.transport<dma>, prod_port = #aie.end_port<DMA : 0>} : !aie.objectfifo<memref<16xi32>>
    %core = aie.core(%src) {
      %a_obj = aie.objectfifo.acquire @a(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @a(Produce, 1)
      %b_obj = aie.objectfifo.acquire @b(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @b(Produce, 1)
      aie.end
    }
  }
}

// -----

// CHECK: error: {{.*}}pinned MM2S DMA channel 0 is out of range or already in use on this tile
module @packet_then_circuit {
  aie.device(npu2) {
    %src = aie.tile(0, 2)
    %a = aie.tile(0, 4)
    %b = aie.tile(0, 5)
    aie.objectfifo @a(%src, {%a}, 2 : i32) {transport = #aie.transport<dma, packet = #aie.packet_info<pkt_id = 3>>, prod_port = #aie.end_port<DMA : 0>} : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @b(%src, {%b}, 2 : i32) {transport = #aie.transport<dma>, prod_port = #aie.end_port<DMA : 0>} : !aie.objectfifo<memref<16xi32>>
    %core = aie.core(%src) {
      %a_obj = aie.objectfifo.acquire @a(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @a(Produce, 1)
      %b_obj = aie.objectfifo.acquire @b(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @b(Produce, 1)
      aie.end
    }
  }
}

// -----

// CHECK: error: {{.*}}pinned MM2S DMA channel 0 is out of range or already in use on this tile
module @circuit_then_packet {
  aie.device(npu2) {
    %src = aie.tile(0, 2)
    %a = aie.tile(0, 4)
    %b = aie.tile(0, 5)
    aie.objectfifo @a(%src, {%a}, 2 : i32) {transport = #aie.transport<dma>, prod_port = #aie.end_port<DMA : 0>} : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @b(%src, {%b}, 2 : i32) {transport = #aie.transport<dma, packet = #aie.packet_info<pkt_id = 3>>, prod_port = #aie.end_port<DMA : 0>} : !aie.objectfifo<memref<16xi32>>
    %core = aie.core(%src) {
      %a_obj = aie.objectfifo.acquire @a(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @a(Produce, 1)
      %b_obj = aie.objectfifo.acquire @b(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @b(Produce, 1)
      aie.end
    }
  }
}

// -----

// CHECK: error: {{.*}}core-pushed senders sharing an MM2S channel require distinct packet IDs
module @same_packet_id {
  aie.device(npu2) {
    %src = aie.tile(0, 2)
    %a = aie.tile(0, 4)
    %b = aie.tile(0, 5)
    aie.objectfifo @a(%src, {%a}, 2 : i32) {transport = #aie.transport<dma, packet = #aie.packet_info<pkt_id = 3>>, prod_port = #aie.end_port<DMA : 0>} : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @b(%src, {%b}, 2 : i32) {transport = #aie.transport<dma, packet = #aie.packet_info<pkt_id = 3>>, prod_port = #aie.end_port<DMA : 0>} : !aie.objectfifo<memref<16xi32>>
    %core = aie.core(%src) {
      %a_obj = aie.objectfifo.acquire @a(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @a(Produce, 1)
      %b_obj = aie.objectfifo.acquire @b(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @b(Produce, 1)
      aie.end
    }
  }
}
