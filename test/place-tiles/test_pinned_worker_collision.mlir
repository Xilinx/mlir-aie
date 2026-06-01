//===- test_pinned_worker_collision.mlir -----------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// Four Workers, one pinned at Tile(0,4) and three unpinned, all sharing
// ObjectFifos in a bottleneck-shape topology. --aie-place-tiles must assign
// four distinct CoreTile coordinates so that no two `aie.core` ops share a
// tile, and the downstream AIEObjectFifoStatefulTransform pass must succeed
// without exhausting per-tile DMA channels.
//
// Captured from programming_examples/ml/bottleneck/bottleneck.py after
// stripping the residual `tile=AnyMemTile` and three `tile=Tile(0,N)`
// arguments left on the IRON Worker/ObjectFifo constructors.

// RUN: aie-opt --aie-place-tiles --aie-objectFifo-stateful-transform %s 2>&1 | FileCheck %s

// CHECK-NOT: error
// CHECK-NOT: DMA channel exceeded

module @pinned_worker_collision {
  aie.device(npu2) {
    %t_w1 = aie.logical_tile<CoreTile>(?, ?)
    %t_w2 = aie.logical_tile<CoreTile>(?, ?)
    %t_w3 = aie.logical_tile<CoreTile>(?, ?)
    %t_w4 = aie.logical_tile<CoreTile>(0, 4)
    %shim = aie.logical_tile<ShimNOCTile>(?, ?)
    %mem  = aie.logical_tile<MemTile>(?, ?)
    aie.objectfifo @in1  (%shim, {%t_w1}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @act23(%t_w1, {%t_w2, %t_w3}, [2 : i32, 4 : i32, 4 : i32]) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @act34(%t_w2, {%t_w4}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @act54(%t_w3, {%t_w4}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @skip (%mem,  {%t_w4}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @wts  (%mem,  {%t_w4}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.core(%t_w1) { aie.end }
    aie.core(%t_w2) { aie.end }
    aie.core(%t_w3) { aie.end }
    aie.core(%t_w4) { aie.end }
  }
}
