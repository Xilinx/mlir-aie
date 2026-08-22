//===- packet_id_bad.mlir ---------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform="packet-sw-objFifos=true skip-verify=true" -split-input-file --verify-diagnostics %s

// The device offers packet IDs 0 through 31 and the flows below claim every one
// of them, leaving the fifo nothing to route with.

module @packet_ids_exhausted {
 aie.device(xcve2302) {
    %tile00 = aie.tile(0, 0)
    %tile02 = aie.tile(0, 2)
    %tile12 = aie.tile(1, 2)
    %tile33 = aie.tile(3, 3)

    // expected-error@+1 {{'aie.objectfifo.flow' op max number of packet IDs reached}}
    aie.objectfifo @of1 (%tile12, {%tile33}, 2 : i32) : !aie.objectfifo<memref<16xi32>>

    aie.packet_flow(0) {
      aie.packet_source<%tile02, Trace : 0>
      aie.packet_dest<%tile00, DMA : 1>
    } {keep_pkt_header = true}

    aie.packet_flow(1) {
      aie.packet_source<%tile02, Trace : 0>
      aie.packet_dest<%tile00, DMA : 1>
    } {keep_pkt_header = true}

    aie.packet_flow(2) {
      aie.packet_source<%tile02, Trace : 0>
      aie.packet_dest<%tile00, DMA : 1>
    } {keep_pkt_header = true}

    aie.packet_flow(3) {
      aie.packet_source<%tile02, Trace : 0>
      aie.packet_dest<%tile00, DMA : 1>
    } {keep_pkt_header = true}

    aie.packet_flow(4) {
      aie.packet_source<%tile02, Trace : 0>
      aie.packet_dest<%tile00, DMA : 1>
    } {keep_pkt_header = true}

    aie.packet_flow(5) {
      aie.packet_source<%tile02, Trace : 0>
      aie.packet_dest<%tile00, DMA : 1>
    } {keep_pkt_header = true}

    aie.packet_flow(6) {
      aie.packet_source<%tile02, Trace : 0>
      aie.packet_dest<%tile00, DMA : 1>
    } {keep_pkt_header = true}

    aie.packet_flow(7) {
      aie.packet_source<%tile02, Trace : 0>
      aie.packet_dest<%tile00, DMA : 1>
    } {keep_pkt_header = true}

    aie.packet_flow(8) {
      aie.packet_source<%tile02, Trace : 0>
      aie.packet_dest<%tile00, DMA : 1>
    } {keep_pkt_header = true}

    aie.packet_flow(9) {
      aie.packet_source<%tile02, Trace : 0>
      aie.packet_dest<%tile00, DMA : 1>
    } {keep_pkt_header = true}

    aie.packet_flow(10) {
      aie.packet_source<%tile02, Trace : 0>
      aie.packet_dest<%tile00, DMA : 1>
    } {keep_pkt_header = true}

    aie.packet_flow(11) {
      aie.packet_source<%tile02, Trace : 0>
      aie.packet_dest<%tile00, DMA : 1>
    } {keep_pkt_header = true}

    aie.packet_flow(12) {
      aie.packet_source<%tile02, Trace : 0>
      aie.packet_dest<%tile00, DMA : 1>
    } {keep_pkt_header = true}

    aie.packet_flow(13) {
      aie.packet_source<%tile02, Trace : 0>
      aie.packet_dest<%tile00, DMA : 1>
    } {keep_pkt_header = true}

    aie.packet_flow(14) {
      aie.packet_source<%tile02, Trace : 0>
      aie.packet_dest<%tile00, DMA : 1>
    } {keep_pkt_header = true}

    aie.packet_flow(15) {
      aie.packet_source<%tile02, Trace : 0>
      aie.packet_dest<%tile00, DMA : 1>
    } {keep_pkt_header = true}

    aie.packet_flow(16) {
      aie.packet_source<%tile02, Trace : 0>
      aie.packet_dest<%tile00, DMA : 1>
    } {keep_pkt_header = true}

    aie.packet_flow(17) {
      aie.packet_source<%tile02, Trace : 0>
      aie.packet_dest<%tile00, DMA : 1>
    } {keep_pkt_header = true}

    aie.packet_flow(18) {
      aie.packet_source<%tile02, Trace : 0>
      aie.packet_dest<%tile00, DMA : 1>
    } {keep_pkt_header = true}

    aie.packet_flow(19) {
      aie.packet_source<%tile02, Trace : 0>
      aie.packet_dest<%tile00, DMA : 1>
    } {keep_pkt_header = true}

    aie.packet_flow(20) {
      aie.packet_source<%tile02, Trace : 0>
      aie.packet_dest<%tile00, DMA : 1>
    } {keep_pkt_header = true}

    aie.packet_flow(21) {
      aie.packet_source<%tile02, Trace : 0>
      aie.packet_dest<%tile00, DMA : 1>
    } {keep_pkt_header = true}

    aie.packet_flow(22) {
      aie.packet_source<%tile02, Trace : 0>
      aie.packet_dest<%tile00, DMA : 1>
    } {keep_pkt_header = true}

    aie.packet_flow(23) {
      aie.packet_source<%tile02, Trace : 0>
      aie.packet_dest<%tile00, DMA : 1>
    } {keep_pkt_header = true}

    aie.packet_flow(24) {
      aie.packet_source<%tile02, Trace : 0>
      aie.packet_dest<%tile00, DMA : 1>
    } {keep_pkt_header = true}

    aie.packet_flow(25) {
      aie.packet_source<%tile02, Trace : 0>
      aie.packet_dest<%tile00, DMA : 1>
    } {keep_pkt_header = true}

    aie.packet_flow(26) {
      aie.packet_source<%tile02, Trace : 0>
      aie.packet_dest<%tile00, DMA : 1>
    } {keep_pkt_header = true}

    aie.packet_flow(27) {
      aie.packet_source<%tile02, Trace : 0>
      aie.packet_dest<%tile00, DMA : 1>
    } {keep_pkt_header = true}

    aie.packet_flow(28) {
      aie.packet_source<%tile02, Trace : 0>
      aie.packet_dest<%tile00, DMA : 1>
    } {keep_pkt_header = true}

    aie.packet_flow(29) {
      aie.packet_source<%tile02, Trace : 0>
      aie.packet_dest<%tile00, DMA : 1>
    } {keep_pkt_header = true}

    aie.packet_flow(30) {
      aie.packet_source<%tile02, Trace : 0>
      aie.packet_dest<%tile00, DMA : 1>
    } {keep_pkt_header = true}

    aie.packet_flow(31) {
      aie.packet_source<%tile02, Trace : 0>
      aie.packet_dest<%tile00, DMA : 1>
    } {keep_pkt_header = true}
 }
}

// -----

// A pinned ID has to name one the device actually has.

module @packet_id_out_of_range {
 aie.device(xcve2302) {
    %tile12 = aie.tile(1, 2)
    %tile33 = aie.tile(3, 3)

    // expected-error@+1 {{'aie.objectfifo.flow' op packet_id 32 is out of range (max 31)}}
    aie.objectfifo @of1 (%tile12, {%tile33}, 2 : i32) {packet, packet_id = 32 : i8} : !aie.objectfifo<memref<16xi32>>
 }
}
