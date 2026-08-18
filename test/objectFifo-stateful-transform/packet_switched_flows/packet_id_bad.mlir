//===- packet_id_bad.mlir ---------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform="skip-verify=true" -split-input-file --verify-diagnostics %s

// A packet header carries five bits of id, so a pinned one above 31 cannot be
// expressed on the wire.

module @pinned_out_of_range {
 aie.device(xcve2302) {
    %tile12 = aie.tile(1, 2)
    %tile33 = aie.tile(3, 3)

    // expected-error@+1 {{packet_id 32 is out of range (max 31)}}
    aie.objectfifo @of1 (%tile12, {%tile33}, 2 : i32) {packet, packet_id = 32 : i8} : !aie.objectfifo<memref<16xi32>>
 }
}

// -----

// A pinned id says nothing unless the flow is packet-switched.

module @pinned_without_packet {
 aie.device(xcve2302) {
    %tile12 = aie.tile(1, 2)
    %tile33 = aie.tile(3, 3)

    // expected-error@+1 {{packet_id is only meaningful on a packet objectfifo}}
    aie.objectfifo @of1 (%tile12, {%tile33}, 2 : i32) {packet_id = 3 : i8} : !aie.objectfifo<memref<16xi32>>
 }
}
