//===- namebase_gate.mlir ---------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectfifo-liveness --verify-diagnostics %s
//
// Coupling gate. The pass groups a coupled broadcast by the shared name base of
// its halves (memX_0, memX_1 -> "memX"), but a name is a weak signal, so the
// grouping is gated on structural agreement before the fan-out is summed. Here
// two cyclic multicast fifos merely COLLIDE on the base name "memX" -- they are
// unrelated broadcasts with different element types (32x32xbf16 vs 16x16xbf16).
// The gate must keep them in separate groups, so each is checked on its own:
// fan = 2, T = 2 -> demand = 4 == slack 2 * depth = 4, no error. If the name
// alone were trusted (no gate) they would be folded into one group with
// fan = 4 -> demand = 8 > 4 and the pass would emit a FALSE positive. So this
// file must stay silent; it pins that name-base grouping is structurally gated.
module {
  aie.device(npu2) {
    %mem0 = aie.logical_tile<MemTile>(?, ?)
    %mem1 = aie.logical_tile<MemTile>(?, ?)
    %c0 = aie.logical_tile<CoreTile>(?, ?)
    %c1 = aie.logical_tile<CoreTile>(?, ?)
    %c2 = aie.logical_tile<CoreTile>(?, ?)
    %c3 = aie.logical_tile<CoreTile>(?, ?)
    // Same base name "memX", but different element types -> not one tensor.
    aie.objectfifo @memX_0(%mem0, {%c0, %c1}, 2 : i32) : !aie.objectfifo<memref<32x32xbf16>>
    aie.objectfifo @memX_1(%mem1, {%c2, %c3}, 2 : i32) : !aie.objectfifo<memref<16x16xbf16>>
    // A back-pressure output per half closes its cycle and sets T = 2.
    aie.objectfifo @memC0(%c0, {%mem0}, 2 : i32) {repeat_count = 2 : i32} : !aie.objectfifo<memref<8x8xf32>>
    aie.objectfifo @memC1(%c2, {%mem1}, 2 : i32) {repeat_count = 2 : i32} : !aie.objectfifo<memref<8x8xf32>>
  }
}
