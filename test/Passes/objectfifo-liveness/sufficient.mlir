//===- sufficient.mlir -----------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectfifo-liveness --verify-diagnostics %s
//
// True-negative / boundary case. The same coupled weight broadcast as
// under_buffered.mlir, but with only ONE half present (memW1_0): array_fan = 4,
// and the back-pressure cycle is multi-trip (repeat_count = 2 -> T = 2), so
// demand = array_fan * T = 8 outstanding tokens exactly equals the round-trip
// slack 2 * depth = 8. demand > slack is false, so the fifo is sufficiently
// buffered and the pass MUST stay silent. Adding the second half (memW1_1) is
// what tips demand to 16 > 8 in under_buffered.mlir -- the two files differ only
// by the coupled-group fan-out summation, so this pins that the check neither
// under- nor over-counts at the boundary.
//
// Minimal skeleton: the pass walks ObjectFifoCreateOp only, so cores/kernels/DMA
// are elided.
module {
  aie.device(npu2) {
    %mem0 = aie.logical_tile<MemTile>(?, ?)
    %c0 = aie.logical_tile<CoreTile>(?, ?)
    %c1 = aie.logical_tile<CoreTile>(?, ?)
    %c2 = aie.logical_tile<CoreTile>(?, ?)
    %c3 = aie.logical_tile<CoreTile>(?, ?)
    // Single-half weight broadcast: fan = 4, depth = 4.
    aie.objectfifo @memW1_0(%mem0, {%c0, %c1, %c2, %c3}, 4 : i32) : !aie.objectfifo<memref<32x32xbf16>>
    // Output closes the back-pressure cycle and carries the multi-trip count
    // (repeat_count = 2 -> T = 2). demand = 4 * 2 = 8 == slack 2 * 4 = 8.
    aie.objectfifo @memC0(%c0, {%mem0}, 4 : i32) {repeat_count = 2 : i32} : !aie.objectfifo<memref<16x32xf32>>
  }
}
