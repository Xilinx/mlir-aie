//===- test_place_flows.mlir ----------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --aie-place-tiles %s | FileCheck %s

// MemTile placed near connected core via aie.flow.
// CHECK-LABEL: @flow_memtile_near_core
module @flow_memtile_near_core {
  aie.device(npu1) {
    // CHECK-DAG: %[[CORE:.*]] = aie.tile(0, 2)
    %core = aie.logical_tile<CoreTile>(?, ?)
    // CHECK-DAG: %[[MEM:.*]] = aie.tile(0, 1)
    %mem = aie.logical_tile<MemTile>(?, ?)
    // CHECK: aie.flow(%[[MEM]], DMA : 0, %[[CORE]], DMA : 0)
    aie.flow(%mem, DMA : 0, %core, DMA : 0)
    // CHECK-NOT: aie.logical_tile
  }
}

// -----

// Shim->Mem->Core: connected component places mem and shim in the core's column.
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

// Memtile shared by two cores: common column is the rounded average.
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

// Cores pinned to columns 0 and 2: memtile lands at the rounded average (col 1).
// CHECK-LABEL: @flow_memtile_averaging
module @flow_memtile_averaging {
  aie.device(npu1) {
    // CHECK-DAG: %[[C0:.*]] = aie.tile(0, 2)
    %c0 = aie.logical_tile<CoreTile>(0, 2)
    // CHECK-DAG: %[[C2:.*]] = aie.tile(2, 2)
    %c2 = aie.logical_tile<CoreTile>(2, 2)
    // CHECK-DAG: %[[MEM:.*]] = aie.tile(1, 1)
    %mem = aie.logical_tile<MemTile>(?, ?)

    aie.flow(%mem, DMA : 0, %c0, DMA : 0)
    aie.flow(%mem, DMA : 1, %c2, DMA : 0)
    aie.core(%c0) { aie.end }
    aie.core(%c2) { aie.end }
    // CHECK-NOT: aie.logical_tile
  }
}

// -----

// Two MM2S flows fit within a npu1 shim's 2 source DMA channels.
// CHECK-LABEL: @flow_shim_channel_capacity
module @flow_shim_channel_capacity {
  aie.device(npu1) {
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

// Non-DMA bundles do not contribute to channel demand.
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

// -----

// Objectfifo and flow on the same tiles contribute additively.
// CHECK-LABEL: @mixed_objfifo_and_flow
module @mixed_objfifo_and_flow {
  aie.device(npu1) {
    // CHECK-DAG: %[[CORE:.*]] = aie.tile(0, 2)
    %core = aie.logical_tile<CoreTile>(?, ?)
    // CHECK-DAG: %[[MEM:.*]] = aie.tile(0, 1)
    %mem  = aie.logical_tile<MemTile>(?, ?)
    // CHECK-DAG: %[[SHIM:.*]] = aie.tile(0, 0)
    %shim = aie.logical_tile<ShimNOCTile>(?, ?)

    aie.objectfifo @of1 (%shim, {%mem}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.flow(%mem, DMA : 0, %core, DMA : 0)
    // CHECK-NOT: aie.logical_tile
  }
}

// -----

// Packet flow with one source and two destinations: every src x dst pair
// becomes a connectivity edge, and both destinations land in the source's
// connected component.
// CHECK-LABEL: @packet_flow_one_to_many
module @packet_flow_one_to_many {
  aie.device(npu1) {
    // CHECK-DAG: %[[C1:.*]] = aie.tile(0, 2)
    %c1 = aie.logical_tile<CoreTile>(?, ?)
    // CHECK-DAG: %[[C2:.*]] = aie.tile(0, 3)
    %c2 = aie.logical_tile<CoreTile>(?, ?)
    // CHECK-DAG: %[[MEM:.*]] = aie.tile(0, 1)
    %mem = aie.logical_tile<MemTile>(?, ?)
    aie.packet_flow(0x1) {
      aie.packet_source<%mem, DMA : 0>
      aie.packet_dest<%c1,  DMA : 0>
      aie.packet_dest<%c2,  DMA : 0>
    }
    // CHECK-NOT: aie.logical_tile
  }
}

// -----

// A flow with one already-resolved TileOp endpoint and one LogicalTileOp
// endpoint: the TileOp side contributes nothing (no placer-visible budget),
// the LogicalTileOp side still placement-validates and gets placed.
// CHECK-LABEL: @flow_mixed_logical_and_tile
module @flow_mixed_logical_and_tile {
  aie.device(npu1) {
    // CHECK-DAG: %[[SHIM:.*]] = aie.tile(0, 0)
    %shim = aie.tile(0, 0)
    // CHECK-DAG: %[[CORE:.*]] = aie.tile(0, 2)
    %core = aie.logical_tile<CoreTile>(?, ?)
    aie.flow(%shim, DMA : 0, %core, DMA : 0)
    // CHECK-NOT: aie.logical_tile
  }
}

// -----

// Broadcast on the source side: three flows fan out from the same shim DMA
// channel 0. Hardware-wise this is one MM2S channel, so the shim's MM2S
// budget should account for 1 (not 3). Without dedup, the shim would be
// counted as needing 3 MM2S channels and exceed npu1's 2-channel limit.
// CHECK-LABEL: @flow_source_broadcast_dedup
module @flow_source_broadcast_dedup {
  aie.device(npu1) {
    // CHECK-DAG: %[[SHIM:.*]] = aie.tile(0, 0)
    %shim = aie.logical_tile<ShimNOCTile>(0, 0)
    // CHECK-DAG: %[[C1:.*]] = aie.tile(0, 2)
    %c1 = aie.logical_tile<CoreTile>(?, ?)
    // CHECK-DAG: %[[C2:.*]] = aie.tile(0, 3)
    %c2 = aie.logical_tile<CoreTile>(?, ?)
    // CHECK-DAG: %[[C3:.*]] = aie.tile(0, 4)
    %c3 = aie.logical_tile<CoreTile>(?, ?)

    // Three flows share source DMA channel 0 -> 1 MM2S consumed on shim.
    aie.flow(%shim, DMA : 0, %c1, DMA : 0)
    aie.flow(%shim, DMA : 0, %c2, DMA : 0)
    aie.flow(%shim, DMA : 0, %c3, DMA : 0)
    // CHECK-NOT: aie.logical_tile
  }
}

// -----

// Merge on the destination side: two flows from distinct sources land on the
// same memtile S2MM channel. Hardware-wise this is one S2MM channel, so the
// memtile's S2MM budget should account for 1 (not 2).
// CHECK-LABEL: @flow_dest_merge_dedup
module @flow_dest_merge_dedup {
  aie.device(npu1) {
    // CHECK-DAG: %[[MEM:.*]] = aie.tile(0, 1)
    %mem = aie.logical_tile<MemTile>(?, ?)
    // CHECK-DAG: %[[C1:.*]] = aie.tile(0, 2)
    %c1 = aie.logical_tile<CoreTile>(?, ?)
    // CHECK-DAG: %[[C2:.*]] = aie.tile(0, 3)
    %c2 = aie.logical_tile<CoreTile>(?, ?)

    // Both flows merge onto memtile S2MM channel 0 -> 1 S2MM consumed.
    aie.flow(%c1, DMA : 0, %mem, DMA : 0)
    aie.flow(%c2, DMA : 0, %mem, DMA : 0)
    // CHECK-NOT: aie.logical_tile
  }
}

// -----

// Packet flow broadcast to many destinations on the same dest DMA channel:
// the source side counts once per (tile, channel) and each PacketDestOp's
// (tile, channel) pair is also deduplicated.
// CHECK-LABEL: @packet_flow_dedup
module @packet_flow_dedup {
  aie.device(npu1) {
    // CHECK-DAG: %[[MEM:.*]] = aie.tile(0, 1)
    %mem = aie.logical_tile<MemTile>(?, ?)
    // CHECK-DAG: %[[C1:.*]] = aie.tile(0, 2)
    %c1 = aie.logical_tile<CoreTile>(?, ?)
    // CHECK-DAG: %[[C2:.*]] = aie.tile(0, 3)
    %c2 = aie.logical_tile<CoreTile>(?, ?)
    // CHECK-DAG: %[[C3:.*]] = aie.tile(0, 4)
    %c3 = aie.logical_tile<CoreTile>(?, ?)
    // Two packet flows share the same memtile MM2S channel 0 -> count once.
    aie.packet_flow(0x1) {
      aie.packet_source<%mem, DMA : 0>
      aie.packet_dest<%c1, DMA : 0>
      aie.packet_dest<%c2, DMA : 0>
    }
    aie.packet_flow(0x2) {
      aie.packet_source<%mem, DMA : 0>
      aie.packet_dest<%c3, DMA : 0>
    }
    // CHECK-NOT: aie.logical_tile
  }
}
