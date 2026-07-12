//===- under_buffered.mlir -------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectfifo-liveness --verify-diagnostics %s
//
// True-positive. A single coupled weight broadcast is split into two halves that
// live on two different MemTiles (memW1_0, memW1_1) and drive two disjoint
// back-pressure cycles. The halves are one logical tensor, coupled only by their
// shared name base "memW1", so the SDF demand SUMS their fan-out: array_fan =
// 4 + 4 = 8. The back-pressure cycles carry a multi-trip load (repeat_count = 2
// -> T = 2), so demand = array_fan * T = 16 outstanding tokens, while the
// allocated round-trip slack is only 2 * depth = 8. 16 > 8 is a static
// under-buffering deadlock: the IR compiles clean and hangs the array at
// runtime. The pass must flag it and name the required depth.
//
// This is the minimal skeleton the pass actually walks (ObjectFifoCreateOp only);
// the compute cores, kernels and DMA program are elided. Contrast sufficient.mlir,
// which keeps only ONE half (fan = 4, demand = 8 == slack 8) and stays silent --
// the delta between the two files is exactly the coupled-group summation.
module {
  aie.device(npu2) {
    %mem0 = aie.logical_tile<MemTile>(?, ?)
    %mem1 = aie.logical_tile<MemTile>(?, ?)
    %c0 = aie.logical_tile<CoreTile>(?, ?)
    %c1 = aie.logical_tile<CoreTile>(?, ?)
    %c2 = aie.logical_tile<CoreTile>(?, ?)
    %c3 = aie.logical_tile<CoreTile>(?, ?)
    %c4 = aie.logical_tile<CoreTile>(?, ?)
    %c5 = aie.logical_tile<CoreTile>(?, ?)
    %c6 = aie.logical_tile<CoreTile>(?, ?)
    %c7 = aie.logical_tile<CoreTile>(?, ?)
    // Coupled weight broadcast, split across two MemTiles: fan 4 + 4 = 8, depth 4.
    // expected-error @below {{objectFIFO @memW1 in a cyclic dependency requires depth >= 8 for deadlock-free execution; allocated depth = 4}}
    aie.objectfifo @memW1_0(%mem0, {%c0, %c1, %c2, %c3}, 4 : i32) : !aie.objectfifo<memref<32x32xbf16>>
    aie.objectfifo @memW1_1(%mem1, {%c4, %c5, %c6, %c7}, 4 : i32) : !aie.objectfifo<memref<32x32xbf16>>
    // One output per half closes that half's back-pressure cycle to its MemTile,
    // and carries the multi-trip count (repeat_count = 2 -> T = 2).
    aie.objectfifo @memC0(%c0, {%mem0}, 4 : i32) {repeat_count = 2 : i32} : !aie.objectfifo<memref<16x32xf32>>
    aie.objectfifo @memC1(%c4, {%mem1}, 4 : i32) {repeat_count = 2 : i32} : !aie.objectfifo<memref<16x32xf32>>
  }
}
