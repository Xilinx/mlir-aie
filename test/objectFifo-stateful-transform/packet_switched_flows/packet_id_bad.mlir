//===- packet_id_bad.mlir ---------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform="packet-sw-objFifos" -split-input-file --verify-diagnostics %s

module @packet_id {
 // expected-error@+1 {{'aie.device' op max number of packet IDs reached}}
 aie.device(xcve2302) {
    %tile00 = aie.tile(0, 0)
    %tile02 = aie.tile(0, 2)
    %tile12 = aie.tile(1, 2)
    %tile33 = aie.tile(3, 3)

    aie.objectfifo @of1 (%tile12, {%tile33}, 2 : i32) : !aie.objectfifo<memref<16xi32>>

    aie.packet_flow(31) {
      aie.packet_source<%tile02, Trace : 0>
      aie.packet_dest<%tile00, DMA : 1>
    } {keep_pkt_header = true}
 }
}
