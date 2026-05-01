//===- test_place_flows.mlir ----------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// Tests that --aie-place-tiles is connectivity-aware on lowered IR using
// `aie.flow` and `aie.packet_flow` (no `aie.objectfifo` involved).

// RUN: aie-opt --split-input-file --aie-place-tiles %s | FileCheck %s

// MemTile placed near connected core via aie.flow (mirrors the objectfifo
// `memtile_near_core` test in test_place_basic.mlir).
// CHECK-LABEL: @flow_memtile_near_core
module @flow_memtile_near_core {
  aie.device(npu1) {
    // CHECK-DAG: %[[CORE:.*]] = aie.tile(0, 2)
    %core = aie.logical_tile<CoreTile>(?, ?)
    // MemTile placed near core's column (0).
    // CHECK-DAG: %[[MEM:.*]] = aie.tile(0, 1)
    %mem = aie.logical_tile<MemTile>(?, ?)
    // CHECK: aie.flow(%[[MEM]], DMA : 0, %[[CORE]], DMA : 0)
    aie.flow(%mem, DMA : 0, %core, DMA : 0)
    // CHECK-NOT: aie.logical_tile
  }
}

// -----

// Shim --DMA flow--> Mem --DMA flow--> Core: connected component places mem
// and shim near the core's column (column 0 by default).
// CHECK-LABEL: @flow_chain_shim_mem_core
module @flow_chain_shim_mem_core {
  aie.device(npu1) {
    // CHECK-DAG: %[[CORE:.*]] = aie.tile(0, 2)
    %core = aie.logical_tile<CoreTile>(?, ?)
    // CHECK-DAG: %[[MEM:.*]] = aie.tile(0, 1)
    %mem  = aie.logical_tile<MemTile>(?, ?)
    // CHECK-DAG: %[[SHIM:.*]] = aie.tile(0, 0)
    %shim = aie.logical_tile<ShimNOCTile>(?, ?)

    // CHECK-DAG: aie.flow(%[[SHIM]], DMA : 0, %[[MEM]], DMA : 0)
    aie.flow(%shim, DMA : 0, %mem,  DMA : 0)
    // CHECK-DAG: aie.flow(%[[MEM]], DMA : 1, %[[CORE]], DMA : 0)
    aie.flow(%mem,  DMA : 1, %core, DMA : 0)
    // CHECK-NOT: aie.logical_tile
  }
}

// -----

// Two unconstrained cores in different columns share a memtile via flows.
// The memtile's common column is the rounded average of the two cores'
// placed columns. With sequential column-major placement on npu1, the two
// cores land in (0,2) and (0,3) — both column 0 — so the memtile lands in
// column 0 too.
// CHECK-LABEL: @flow_memtile_two_cores
module @flow_memtile_two_cores {
  aie.device(npu1) {
    // CHECK-DAG: %[[C1:.*]] = aie.tile(0, 2)
    %c1 = aie.logical_tile<CoreTile>(?, ?)
    // CHECK-DAG: %[[C2:.*]] = aie.tile(0, 3)
    %c2 = aie.logical_tile<CoreTile>(?, ?)
    // CHECK-DAG: %[[MEM:.*]] = aie.tile(0, 1)
    %mem = aie.logical_tile<MemTile>(?, ?)

    aie.flow(%mem, DMA : 0, %c1, DMA : 0)
    aie.flow(%mem, DMA : 1, %c2, DMA : 0)
    aie.core(%c1) { aie.end }
    aie.core(%c2) { aie.end }
    // CHECK-NOT: aie.logical_tile
  }
}

// -----

// Channel capacity is enforced from flows: a constrained shim with too many
// MM2S flows out of it errors out.
// CHECK-LABEL: @flow_shim_channel_capacity
module @flow_shim_channel_capacity {
  aie.device(npu1) {
    // npu1 ShimMux has 2 source DMA channels per shim. Two flows out from a
    // pinned shim should succeed.
    // CHECK-DAG: %[[SHIM:.*]] = aie.tile(0, 0)
    %shim = aie.logical_tile<ShimNOCTile>(0, 0)
    // CHECK-DAG: %[[C1:.*]] = aie.tile(0, 2)
    %c1 = aie.logical_tile<CoreTile>(?, ?)
    // CHECK-DAG: %[[C2:.*]] = aie.tile(0, 3)
    %c2 = aie.logical_tile<CoreTile>(?, ?)

    aie.flow(%shim, DMA : 0, %c1, DMA : 0)
    aie.flow(%shim, DMA : 1, %c2, DMA : 0)
    // CHECK-NOT: aie.logical_tile
  }
}

// -----

// Packet flow sources/dests are walked too.
// CHECK-LABEL: @packet_flow_memtile_near_core
module @packet_flow_memtile_near_core {
  aie.device(npu1) {
    // CHECK-DAG: %[[CORE:.*]] = aie.tile(0, 2)
    %core = aie.logical_tile<CoreTile>(?, ?)
    // CHECK-DAG: %[[MEM:.*]] = aie.tile(0, 1)
    %mem = aie.logical_tile<MemTile>(?, ?)
    aie.packet_flow(0x1) {
      aie.packet_source<%mem, DMA : 0>
      aie.packet_dest<%core, DMA : 0>
    }
    // CHECK-NOT: aie.logical_tile
  }
}

// -----

// Non-DMA bundles do NOT contribute to channel demand. A core-to-core flow
// over the Core bundle places without consuming any DMA channels — capacity
// check should not fire even though both endpoints are on the same column.
// CHECK-LABEL: @flow_core_bundle_no_dma
module @flow_core_bundle_no_dma {
  aie.device(npu1) {
    // CHECK-DAG: %[[C1:.*]] = aie.tile(0, 2)
    %c1 = aie.logical_tile<CoreTile>(?, ?)
    // CHECK-DAG: %[[C2:.*]] = aie.tile(0, 3)
    %c2 = aie.logical_tile<CoreTile>(?, ?)
    aie.flow(%c1, Core : 0, %c2, Core : 0)
    // CHECK-NOT: aie.logical_tile
  }
}
