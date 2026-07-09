//===- unrelated_cycles_no_crosstalk.mlir ---------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectfifo-liveness --verify-diagnostics %s
//
// Per-group-scoping regression. Two INDEPENDENT cyclic objectFIFO structures
// that share no tiles:
//   SCC A: a wide coupled multicast (fan-out 8) but SINGLE-TRIP (no
//          repeat_count, T == 1) -> safe on its own (the replay guard).
//   SCC B: a point-to-point two-core cycle replayed 8x (repeat_count = 8,
//          fan-out 1) -> safe on its own (no multicast).
// A device-GLOBAL model would take T = max repeat_count = 8 from SCC B and
// array_fan = 8 from SCC A and report a bogus deadlock on the SCC-A broadcast.
// Scoped per coupled-multicast group, SCC A's T is drawn only from SCC A
// (== 1), so the pass MUST stay silent.
module {
  aie.device(npu2) {
    // SCC A: single-trip wide broadcast; @cA closes the cycle back to the mem tile.
    %memA = aie.logical_tile<MemTile>(?, ?)
    %a0 = aie.logical_tile<CoreTile>(?, ?)
    %a1 = aie.logical_tile<CoreTile>(?, ?)
    %a2 = aie.logical_tile<CoreTile>(?, ?)
    %a3 = aie.logical_tile<CoreTile>(?, ?)
    %a4 = aie.logical_tile<CoreTile>(?, ?)
    %a5 = aie.logical_tile<CoreTile>(?, ?)
    %a6 = aie.logical_tile<CoreTile>(?, ?)
    %a7 = aie.logical_tile<CoreTile>(?, ?)
    aie.objectfifo @wA(%memA, {%a0, %a1, %a2, %a3, %a4, %a5, %a6, %a7}, 4 : i32) : !aie.objectfifo<memref<32x32xbf16>>
    aie.objectfifo @cA(%a0, {%memA}, 4 : i32) : !aie.objectfifo<memref<16x32xbf16>>
    // SCC B: independent tiles, high replay, point-to-point (no multicast).
    %b0 = aie.logical_tile<CoreTile>(?, ?)
    %b1 = aie.logical_tile<CoreTile>(?, ?)
    aie.objectfifo @fwdB(%b0, {%b1}, 2 : i32) {repeat_count = 8 : i32} : !aie.objectfifo<memref<16x32xbf16>>
    aie.objectfifo @bwdB(%b1, {%b0}, 2 : i32) {repeat_count = 8 : i32} : !aie.objectfifo<memref<16x32xbf16>>
  }
}
