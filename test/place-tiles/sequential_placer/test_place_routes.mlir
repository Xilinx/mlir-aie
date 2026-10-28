//===- test_place_routes.mlir ---------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --aie-place-tiles %s | FileCheck %s

// An aie.route ties its endpoints' tiles together the way an aie.flow does,
// before allocation has picked any channel: the MemTile follows its core to
// column 2 rather than taking column 0.
// CHECK-LABEL: @route_memtile_near_core
module @route_memtile_near_core {
  aie.device(npu2) {
    // CHECK-DAG: %[[CORE:.*]] = aie.tile(2, 2)
    %core = aie.logical_tile<CoreTile>(2, 2)
    // CHECK-DAG: %[[MEM:.*]] = aie.tile(2, 1)
    %mem = aie.logical_tile<MemTile>(?, ?)
    // CHECK-DAG: aie.route_endpoint @mem_out(%[[MEM]]) DMA
    aie.route_endpoint @mem_out(%mem) DMA
    aie.route_endpoint @core_in(%core) DMA
    aie.route from @mem_out to [@core_in]
    // CHECK-NOT: aie.logical_tile
  }
}

// -----

// A broadcast's destinations span columns 1-3, and the shim feeding the
// MemTile lands under it.
// CHECK-LABEL: @route_broadcast_chain
module @route_broadcast_chain {
  aie.device(npu2) {
    %c1 = aie.logical_tile<CoreTile>(1, 2)
    %c3 = aie.logical_tile<CoreTile>(3, 2)
    // CHECK-DAG: %[[MEM:.*]] = aie.tile(2, 1)
    %mem = aie.logical_tile<MemTile>(?, ?)
    // CHECK-DAG: %[[SHIM:.*]] = aie.tile(2, 0)
    %shim = aie.logical_tile<ShimNOCTile>(?, ?)
    aie.route_endpoint @host(%shim) DMA {fifoName = "host"}
    aie.route_endpoint @mem_in(%mem) DMA
    aie.route_endpoint @mem_out(%mem) DMA
    aie.route_endpoint @d1(%c1) DMA
    aie.route_endpoint @d3(%c3) DMA
    aie.route from @host to [@mem_in]
    aie.route from @mem_out to [@d1, @d3]
    // CHECK-NOT: aie.logical_tile
  }
}

// -----

// Each unassigned route end holds a DMA channel: two MemTiles taking four
// inputs each exceed one MemTile's six, so they cannot share a tile. The one
// with more channels to hold (%mb) is placed first.
// CHECK-LABEL: @route_channel_demand
module @route_channel_demand {
  aie.device(npu2) {
    %core = aie.logical_tile<CoreTile>(0, 2)
    // CHECK-DAG: %[[MA:.*]] = aie.tile(1, 1)
    %ma = aie.logical_tile<MemTile>(?, ?)
    // CHECK-DAG: %[[MB:.*]] = aie.tile(0, 1)
    %mb = aie.logical_tile<MemTile>(?, ?)
    // CHECK-DAG: aie.route_endpoint @a0(%[[MA]]) DMA
    aie.route_endpoint @a0(%ma) DMA
    aie.route_endpoint @a1(%ma) DMA
    aie.route_endpoint @a2(%ma) DMA
    aie.route_endpoint @a3(%ma) DMA
    // CHECK-DAG: aie.route_endpoint @b0(%[[MB]]) DMA
    aie.route_endpoint @b0(%mb) DMA
    aie.route_endpoint @b1(%mb) DMA
    aie.route_endpoint @b2(%mb) DMA
    aie.route_endpoint @b3(%mb) DMA
    aie.route_endpoint @s0(%core) DMA
    aie.route_endpoint @s1(%core) DMA
    aie.route_endpoint @s2(%ma) DMA
    aie.route_endpoint @s3(%ma) DMA
    aie.route_endpoint @s4(%mb) DMA
    aie.route_endpoint @s5(%mb) DMA
    aie.route_endpoint @s6(%mb) DMA
    aie.route_endpoint @s7(%mb) DMA
    aie.route from @s0 to [@a0]
    aie.route from @s1 to [@a1]
    aie.route from @s2 to [@a2]
    aie.route from @s3 to [@a3]
    aie.route from @s4 to [@b0]
    aie.route from @s5 to [@b1]
    aie.route from @s6 to [@b2]
    aie.route from @s7 to [@b3]
    // CHECK-NOT: aie.logical_tile
  }
}

// -----

// Packet routes merging into one endpoint share its channel, so a core taking
// two merged flows plus one more input needs its two S2MM channels, not three.
// CHECK-LABEL: @route_shared_dest_unplaced
// CHECK-DAG: %[[CORE:.*]] = aie.tile(0, 2)
// CHECK-DAG: aie.route_endpoint @merge(%[[CORE]]) DMA
// CHECK-DAG: aie.route_endpoint @own(%[[CORE]]) DMA
module @route_shared_dest_unplaced {
  aie.device(npu2) {
    %mt = aie.logical_tile<MemTile>(0, 1)
    %core = aie.logical_tile<CoreTile>(?, ?)
    aie.core(%core) { aie.end }
    aie.route_endpoint @x(%mt) DMA
    aie.route_endpoint @y(%mt) DMA
    aie.route_endpoint @z(%mt) DMA
    aie.route_endpoint @merge(%core) DMA
    aie.route_endpoint @own(%core) DMA
    aie.route from @x to [@merge] {packet, packet_id = 1 : i8}
    aie.route from @y to [@merge] {packet, packet_id = 2 : i8}
    aie.route from @z to [@own]
    // CHECK-NOT: aie.logical_tile
  }
}

// -----

// The same on a pinned core, fed from three shims.
// CHECK-LABEL: @route_shared_dest_pinned
// CHECK-DAG: %[[CORE:.*]] = aie.tile(0, 2)
// CHECK-DAG: aie.route_endpoint @merge(%[[CORE]]) DMA
// CHECK-DAG: aie.route_endpoint @own(%[[CORE]]) DMA
// CHECK-DAG: aie.route_endpoint @out(%[[CORE]]) DMA
module @route_shared_dest_pinned {
  aie.device(npu2) {
    %core = aie.logical_tile<CoreTile>(0, 2)
    %s0 = aie.logical_tile<ShimNOCTile>(?, ?)
    %s1 = aie.logical_tile<ShimNOCTile>(?, ?)
    %s2 = aie.logical_tile<ShimNOCTile>(?, ?)
    %s3 = aie.logical_tile<ShimNOCTile>(?, ?)
    aie.core(%core) { aie.end }
    aie.route_endpoint @in0(%s0) DMA {fifoName = "in0"}
    aie.route_endpoint @in1(%s1) DMA {fifoName = "in1"}
    aie.route_endpoint @in2(%s2) DMA {fifoName = "in2"}
    aie.route_endpoint @merge(%core) DMA
    aie.route_endpoint @own(%core) DMA
    aie.route from @in0 to [@merge] {packet, packet_id = 1 : i8}
    aie.route from @in1 to [@merge] {packet, packet_id = 2 : i8}
    aie.route from @in2 to [@own]
    aie.route_endpoint @out(%core) DMA
    aie.route_endpoint @back(%s3) DMA {fifoName = "back"}
    aie.route from @out to [@back]
    // CHECK-NOT: aie.logical_tile
  }
}

// -----

// One endpoint sending two packet routes holds one MM2S channel, so a core
// sourcing it and one more flow fits its two.
// CHECK-LABEL: @route_shared_source_unplaced
// CHECK-DAG: %[[CORE:.*]] = aie.tile(0, 2)
// CHECK-DAG: aie.route_endpoint @split(%[[CORE]]) DMA
// CHECK-DAG: aie.route_endpoint @other(%[[CORE]]) DMA
module @route_shared_source_unplaced {
  aie.device(npu2) {
    %mt = aie.logical_tile<MemTile>(0, 1)
    %core = aie.logical_tile<CoreTile>(?, ?)
    aie.core(%core) { aie.end }
    aie.route_endpoint @split(%core) DMA
    aie.route_endpoint @other(%core) DMA
    aie.route_endpoint @a(%mt) DMA
    aie.route_endpoint @b(%mt) DMA
    aie.route_endpoint @c(%mt) DMA
    aie.route from @split to [@a] {packet, packet_id = 1 : i8}
    aie.route from @split to [@b] {packet, packet_id = 2 : i8}
    aie.route from @other to [@c]
    // CHECK-NOT: aie.logical_tile
  }
}
