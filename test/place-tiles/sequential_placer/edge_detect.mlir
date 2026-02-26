//===- edge_detect.mlir ----------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-place-tiles %s | FileCheck %s

// Test based on programming_examples/vision/edge_detect
//
// This verifies that the C++ SequentialPlacer produces the same placement
// behavior as the original Python SequentialPlacer:
//   - Compute tiles: Sequential placement
//   - Mem/shim tiles: Merged and placed near common column
//   - Final result: 1 shim + 1 mem + 4 cores (all in same column)

// CHECK-LABEL: @edge_detect
module @edge_detect {
  aie.device(npu2) {
    %shim_in = aie.logical_tile<ShimNOCTile>(?, ?)
    %shim_out = aie.logical_tile<ShimNOCTile>(?, ?)

    // Separate logical mems from separate forward() calls
    %mem_in = aie.logical_tile<MemTile>(?, ?)
    %mem_out = aie.logical_tile<MemTile>(?, ?)

    %core2 = aie.logical_tile<CoreTile>(?, ?)
    %core3 = aie.logical_tile<CoreTile>(?, ?)
    %core4 = aie.logical_tile<CoreTile>(?, ?)
    %core5 = aie.logical_tile<CoreTile>(?, ?)

    // Compute tiles: column-major sequential placement (fill column vertically)
    // CHECK-DAG: %[[T2:.*]] = aie.tile(0, 2)
    // CHECK-DAG: %[[T3:.*]] = aie.tile(0, 3)
    // CHECK-DAG: %[[T4:.*]] = aie.tile(0, 4)
    // CHECK-DAG: %[[T5:.*]] = aie.tile(0, 5)

    // Common column = 0 (all cores in column 0)
    // Both shims and both mems should merge to column 0

    // CHECK-DAG: %[[SHIM:shim_noc_tile_0_0]] = aie.tile(0, 0)
    // Both shim_in and shim_out should merge to ONE physical shim

    // CHECK-DAG: %[[MEM:mem_tile_0_1]] = aie.tile(0, 1)
    // Both mem_in and mem_out should merge to ONE physical mem

    // Input path: Shim -> {Core, Mem} (consumer: mem_in)
    // CHECK: aie.objectfifo @inOF_L3L2(%[[SHIM]], {%[[T2]], %[[MEM]]}, 2 : i32)
    aie.objectfifo @inOF_L3L2 (%shim_in, {%core2, %mem_in}, 2 : i32) : !aie.objectfifo<memref<1920xi8>>

    // Skip connection
    // CHECK: aie.objectfifo @inOF_L2L1(%[[MEM]], {%[[T5]]}, 2 : i32)
    aie.objectfifo @inOF_L2L1 (%mem_in, {%core5}, 2 : i32) : !aie.objectfifo<memref<1920xi8>>

    // CHECK: aie.objectfifo.link [@inOF_L3L2] -> [@inOF_L2L1]
    aie.objectfifo.link [@inOF_L3L2] -> [@inOF_L2L1] ([] [])

    // Output path: Core -> Mem (consumer: mem_out)
    // CHECK: aie.objectfifo @outOF_L1L2(%[[T5]], {%[[MEM]]}, 2 : i32)
    aie.objectfifo @outOF_L1L2 (%core5, {%mem_out}, 2 : i32) : !aie.objectfifo<memref<1920xi8>>

    // Mem -> Shim (producer: mem_out, same physical tile as mem_in)
    // CHECK: aie.objectfifo @outOF_L2L3(%[[MEM]], {%[[SHIM]]}, 2 : i32)
    aie.objectfifo @outOF_L2L3 (%mem_out, {%shim_out}, 2 : i32) : !aie.objectfifo<memref<1920xi8>>

    // CHECK: aie.objectfifo.link [@outOF_L1L2] -> [@outOF_L2L3]
    aie.objectfifo.link [@outOF_L1L2] -> [@outOF_L2L3] ([] [])

    // CHECK: aie.objectfifo @OF_2to3(%[[T2]], {%[[T3]]}, 2 : i32)
    aie.objectfifo @OF_2to3 (%core2, {%core3}, 2 : i32) : !aie.objectfifo<memref<1920xi8>>

    // CHECK: aie.objectfifo @OF_3to4(%[[T3]], {%[[T4]]}, 2 : i32)
    aie.objectfifo @OF_3to4 (%core3, {%core4}, 2 : i32) : !aie.objectfifo<memref<1920xi8>>

    // CHECK: aie.objectfifo @OF_4to5(%[[T4]], {%[[T5]]}, 2 : i32)
    aie.objectfifo @OF_4to5 (%core4, {%core5}, 2 : i32) : !aie.objectfifo<memref<1920xi8>>

    // CHECK: aie.core(%[[T2]])
    %c2 = aie.core(%core2) { aie.end }

    // CHECK: aie.core(%[[T3]])
    %c3 = aie.core(%core3) { aie.end }

    // CHECK: aie.core(%[[T4]])
    %c4 = aie.core(%core4) { aie.end }

    // CHECK: aie.core(%[[T5]])
    %c5 = aie.core(%core5) { aie.end }

    // CHECK-NOT: aie.logical_tile
  }
}
