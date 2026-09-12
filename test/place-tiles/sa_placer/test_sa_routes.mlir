//===- test_sa_routes.mlir ------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --aie-place-tiles='placer=sa_placer sa-seed=42' %s | FileCheck %s

// A route is a net for the annealer too: the shim its route reaches ends up
// under the pinned core rather than wherever the initial placement put it.

// CHECK-LABEL: @sa_route_shim_near_core
module @sa_route_shim_near_core {
  aie.device(npu1) {
    // CHECK-DAG: %[[CORE:.*]] = aie.tile(3, 2)
    %core = aie.logical_tile<CoreTile>(3, 2)
    // CHECK-DAG: %[[SHIM:.*]] = aie.tile(3, 0)
    %shim = aie.logical_tile<ShimNOCTile>(?, ?)
    aie.route_endpoint @in(%shim) DMA
    aie.objectfifo.pool @p(%core) {depth = 2 : i32} : memref<16xi32> {
      aie.objectfifo.segment @s0 {offset = 0 : i32, size = 16 : i32}
    }
    aie.objectfifo.dma_endpoint @p_dma(%core) fills @p
    aie.objectfifo.core_endpoint @p_core(%core) drains @p
    aie.route from @in to [@p_dma]
    // CHECK-NOT: aie.logical_tile
  }
}
