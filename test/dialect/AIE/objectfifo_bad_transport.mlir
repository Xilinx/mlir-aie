//===- objectfifo_bad_transport.mlir ----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --verify-diagnostics %s

// The switches this replaced would otherwise be carried along as unknown
// attributes and silently ignored, so each is named and refused.
module {
 aie.device(xcve2302) {
    %tile12 = aie.tile(1, 2)
    %tile13 = aie.tile(1, 3)
    // expected-error@+1 {{`via_DMA` has been replaced; write transport = #aie.transport<dma>}}
    aie.objectfifo @of (%tile12, {%tile13}, 2 : i32) {via_DMA = true} : !aie.objectfifo<memref<16xi32>>
 }
}

// -----

module {
 aie.device(xcve2302) {
    %tile12 = aie.tile(1, 2)
    %tile13 = aie.tile(1, 3)
    // expected-error@+1 {{`packet` has been replaced; write transport = #aie.transport<dma, packet = #aie.packet_info<>>}}
    aie.objectfifo @of (%tile12, {%tile13}, 2 : i32) {packet} : !aie.objectfifo<memref<16xi32>>
 }
}

// -----

module {
 aie.device(xcve2302) {
    %tile12 = aie.tile(1, 2)
    %tile13 = aie.tile(1, 3)
    // expected-error@+1 {{`packet_id` has been replaced; write transport = #aie.transport<dma, packet = #aie.packet_info<pkt_id = 3>>}}
    aie.objectfifo @of (%tile12, {%tile13}, 2 : i32) {packet_id = 3 : i8} : !aie.objectfifo<memref<16xi32>>
 }
}

// -----

module {
 aie.device(xcve2302) {
    %tile12 = aie.tile(1, 2)
    %tile13 = aie.tile(1, 3)
    // expected-error@+1 {{`aie_stream` has been replaced; write prod_port = #aie.end_port<Core : 0>}}
    aie.objectfifo @of (%tile12, {%tile13}, 2 : i32) {aie_stream = 0 : i32} : !aie.objectfifo<memref<16xi32>>
 }
}

// -----

module {
 aie.device(xcve2302) {
    %tile12 = aie.tile(1, 2)
    %tile13 = aie.tile(1, 3)
    // expected-error@+1 {{`aie_stream_port` has been replaced; write prod_port = #aie.end_port<Core : 0>}}
    aie.objectfifo @of (%tile12, {%tile13}, 2 : i32) {aie_stream_port = 1 : i32} : !aie.objectfifo<memref<16xi32>>
 }
}

// -----

module {
 aie.device(xcve2302) {
    %tile12 = aie.tile(1, 2)
    %tile13 = aie.tile(1, 3)
    // expected-error@+1 {{`prod_dma_channel` has been replaced; write prod_port = #aie.end_port<DMA : 1>}}
    aie.objectfifo @of (%tile12, {%tile13}, 2 : i32) {prod_dma_channel = 1 : i32} : !aie.objectfifo<memref<16xi32>>
 }
}

// -----

module {
 aie.device(xcve2302) {
    %tile12 = aie.tile(1, 2)
    %tile13 = aie.tile(1, 3)
    // expected-error@+1 {{`cons_dma_channels` has been replaced; write cons_ports = [#aie.end_port<DMA : 1>]}}
    aie.objectfifo @of (%tile12, {%tile13}, 2 : i32) {cons_dma_channels = array<i32: 1>} : !aie.objectfifo<memref<16xi32>>
 }
}

// -----

// A stream reaches exactly one consumer, because a stream port is a wire.
module {
 aie.device(xcve2302) {
    %tile12 = aie.tile(1, 2)
    %tile13 = aie.tile(1, 3)
    %tile23 = aie.tile(2, 3)
    // expected-error@+1 {{a Core stream port end can only be used in 1-to-1 object FIFOs}}
    aie.objectfifo @of (%tile12, {%tile13, %tile23}, 2 : i32) {prod_port = #aie.end_port<Core : 0>} : !aie.objectfifo<memref<16xi32>>
 }
}

// -----

// Only compute tiles have stream ports.
module {
 aie.device(xcve2302) {
    %tile11 = aie.tile(1, 1)
    %tile13 = aie.tile(1, 3)
    // expected-error@+1 {{a Core stream port end is not available for shim and mem tiles}}
    aie.objectfifo @of (%tile11, {%tile13}, 2 : i32) {prod_port = #aie.end_port<Core : 0>} : !aie.objectfifo<memref<16xi32>>
 }
}

// -----

module {
 aie.device(xcve2302) {
    %tile12 = aie.tile(1, 2)
    %tile13 = aie.tile(1, 3)
    // expected-error@+1 {{a Core stream port end sends over the stream, not through shared memory}}
    aie.objectfifo @of (%tile12, {%tile13}, 2 : i32) {transport = #aie.transport<shared_mem>, cons_ports = [#aie.end_port<Core : 0>]} : !aie.objectfifo<memref<16xi32>>
 }
}

// -----

// A packet header rides the DMA path's stream connection.
module {
 aie.device(xcve2302) {
    %tile12 = aie.tile(1, 2)
    %tile13 = aie.tile(1, 3)
    // expected-error@+1 {{`packet` belongs to a dma or auto transport, not shared_mem}}
    aie.objectfifo @of (%tile12, {%tile13}, 2 : i32) {transport = #aie.transport<shared_mem, packet = #aie.packet_info<pkt_id = 1>>} : !aie.objectfifo<memref<16xi32>>
 }
}

// -----

module {
 aie.device(xcve2302) {
    %tile12 = aie.tile(1, 2)
    %tile13 = aie.tile(1, 3)
    // expected-error@+1 {{an objectFifo end is driven by a DMA channel or a Core stream port, not North}}
    aie.objectfifo @of (%tile12, {%tile13}, 2 : i32) {prod_port = #aie.end_port<North : 0>} : !aie.objectfifo<memref<16xi32>>
 }
}

// -----

module {
 aie.device(xcve2302) {
    %tile12 = aie.tile(1, 2)
    %tile13 = aie.tile(1, 3)
    // expected-error@+1 {{a Core end needs its stream port}}
    aie.objectfifo @of (%tile12, {%tile13}, 2 : i32) {prod_port = #aie.end_port<Core>} : !aie.objectfifo<memref<16xi32>>
 }
}

// -----

module {
 aie.device(xcve2302) {
    %tile12 = aie.tile(1, 2)
    %tile13 = aie.tile(1, 3)
    // expected-error@+1 {{`cons_ports` has 2 entries for 1 consumers}}
    aie.objectfifo @of (%tile12, {%tile13}, 2 : i32) {cons_ports = [#aie.end_port<DMA>, #aie.end_port<DMA : 1>]} : !aie.objectfifo<memref<16xi32>>
 }
}
