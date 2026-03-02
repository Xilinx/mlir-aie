//===- test_objectfifo_link.mlir --------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-place-tiles %s | FileCheck %s

// Test: ObjectFifo.link should group linked fifos to same column
// This models the scale_shift pattern: shim -> mem -> {core0, core1}
// CHECK-LABEL: @objectfifo_link_placement
module @objectfifo_link_placement {
  aie.device(npu1) {
    // Physical tiles should be in column 0
    // CHECK-DAG: %[[SHIM:.*]] = aie.tile(0, 0)
    // CHECK-DAG: %[[MEM:.*]] = aie.tile(0, 1)
    // CHECK-DAG: %[[CORE0:.*]] = aie.tile(0, 2)
    // CHECK-DAG: %[[CORE1:.*]] = aie.tile(0, 3)

    %shim = aie.logical_tile<ShimNOCTile>(?, ?)
    %mem = aie.logical_tile<MemTile>(?, ?)
    %core0 = aie.logical_tile<CoreTile>(?, ?)
    %core1 = aie.logical_tile<CoreTile>(?, ?)

    // shim -> mem (1 input, will split to 2 outputs)
    aie.objectfifo @inA (%shim, {%mem}, 2 : i32) : !aie.objectfifo<memref<1024xbf16>>

    // mem -> core0 (output from split)
    aie.objectfifo @memA0 (%mem, {%core0}, 2 : i32) : !aie.objectfifo<memref<1024xbf16>>

    // mem -> core1 (output from split)
    aie.objectfifo @memA1 (%mem, {%core1}, 2 : i32) : !aie.objectfifo<memref<1024xbf16>>

    // Link: @inA splits to @memA0 and @memA1
    // Mem tile should have:
    // - 1 input channel (from shim via @inA)
    // - 2 output channels (to core0 via @memA0, to core1 via @memA1)
    // Total: 1 input + 2 output = within MemTile capacity (2 input / 2 output)
    aie.objectfifo.link [@inA] -> [@memA0, @memA1] ([] [0, 1024])

    // CHECK-NOT: aie.logical_tile
  }
}
