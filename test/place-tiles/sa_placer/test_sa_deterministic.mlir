//===- test_sa_deterministic.mlir -------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// Verify SA placer produces identical output with the same seed.
// Run twice with the same seed and diff the outputs.

// RUN: aie-opt --aie-place-tiles='placer=sa_placer sa-seed=42' %s > %t.run1.mlir 2>/dev/null
// RUN: aie-opt --aie-place-tiles='placer=sa_placer sa-seed=42' %s > %t.run2.mlir 2>/dev/null
// RUN: diff %t.run1.mlir %t.run2.mlir

// Also verify the output is valid placed IR
// RUN: aie-opt --aie-place-tiles='placer=sa_placer sa-seed=42' %s 2>/dev/null | FileCheck %s

// CHECK-NOT: aie.logical_tile

module @deterministic_4core {
  aie.device(npu2) {
    %shim1 = aie.logical_tile<ShimNOCTile>(?, ?)
    %shim2 = aie.logical_tile<ShimNOCTile>(?, ?)
    %mem1 = aie.logical_tile<MemTile>(?, ?)
    %mem2 = aie.logical_tile<MemTile>(?, ?)
    %c0 = aie.logical_tile<CoreTile>(?, ?)
    %c1 = aie.logical_tile<CoreTile>(?, ?)
    %c2 = aie.logical_tile<CoreTile>(?, ?)
    %c3 = aie.logical_tile<CoreTile>(?, ?)

    // Cascade flow between c0 and c1 (tests cascade determinism too)
    aie.cascade_flow(%c0, %c1)

    // L3 -> L2
    aie.objectfifo @inA(%shim1, {%mem1}, 2 : i32) : !aie.objectfifo<memref<1024xi32>>
    aie.objectfifo @inB(%shim2, {%mem2}, 2 : i32) : !aie.objectfifo<memref<1024xi32>>

    // L2 -> L1 (distribute)
    aie.objectfifo @memA0(%mem1, {%c0}, 2 : i32) : !aie.objectfifo<memref<512xi32>>
    aie.objectfifo @memA1(%mem1, {%c1}, 2 : i32) : !aie.objectfifo<memref<512xi32>>
    aie.objectfifo.link [@inA] -> [@memA0, @memA1]([] [0, 512])

    aie.objectfifo @memB0(%mem2, {%c2}, 2 : i32) : !aie.objectfifo<memref<512xi32>>
    aie.objectfifo @memB1(%mem2, {%c3}, 2 : i32) : !aie.objectfifo<memref<512xi32>>
    aie.objectfifo.link [@inB] -> [@memB0, @memB1]([] [0, 512])

    // L1 -> L2 (join)
    aie.objectfifo @outC0(%c0, {%mem1}, 2 : i32) : !aie.objectfifo<memref<512xi32>>
    aie.objectfifo @outC1(%c1, {%mem1}, 2 : i32) : !aie.objectfifo<memref<512xi32>>
    aie.objectfifo @outC(%mem1, {%shim1}, 2 : i32) : !aie.objectfifo<memref<1024xi32>>
    aie.objectfifo.link [@outC0, @outC1] -> [@outC]([0, 512] [])

    aie.core(%c0) { aie.end }
    aie.core(%c1) { aie.end }
    aie.core(%c2) { aie.end }
    aie.core(%c3) { aie.end }
    aie.end
  }
}
