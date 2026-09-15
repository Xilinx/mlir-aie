//===- duplicate_flow.mlir -------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// DeviceOp::verify rejects a flow that is declared twice. For aie.flow the key
// is the (tile, bundle, channel) pair of endpoints; for aie.packet_flow it is
// the ID together with the sets of sources and destinations. Endpoints on an
// aie.logical_tile that --aie-place-tiles has not placed are skipped, because
// they cannot be compared until placement assigns coordinates -- at which
// point this same check runs again on the resulting aie.tile coordinates.

// RUN: not aie-opt -split-input-file %s 2>&1 | FileCheck %s

// The same circuit-switched flow declared twice.
// CHECK: error{{.*}}'aie.flow' op duplicates an earlier flow; (0, 0) DMA : 0 -> (0, 2) DMA : 0 is already declared
// CHECK: note:{{.*}}the other flow is here
module @flow_dup {
  aie.device(npu2) {
    %shim = aie.tile(0, 0)
    %core = aie.tile(0, 2)
    aie.flow(%shim, DMA : 0, %core, DMA : 0)
    aie.flow(%shim, DMA : 0, %core, DMA : 0)
  }
}

// -----

// Duplicate endpoints reached through two distinct aie.tile ops at the same
// coordinate, as a frontend that did not merge tiles by coordinate emits.
// CHECK: error{{.*}}'aie.flow' op duplicates an earlier flow; (0, 0) DMA : 1 -> (0, 2) DMA : 1 is already declared
module @flow_dup_distinct_tiles {
  aie.device(npu2) {
    %shim0 = aie.tile(0, 0)
    %shim1 = aie.tile(0, 0)
    %core0 = aie.tile(0, 2)
    %core1 = aie.tile(0, 2)
    aie.flow(%shim0, DMA : 1, %core0, DMA : 1)
    aie.flow(%shim1, DMA : 1, %core1, DMA : 1)
  }
}

// -----

// Two pinned logical tiles at the same coordinate, caught before placement.
// CHECK: error{{.*}}'aie.flow' op duplicates an earlier flow; (0, 2) DMA : 0 -> (0, 3) DMA : 0 is already declared
module @flow_dup_logical {
  aie.device(npu2) {
    %c0 = aie.logical_tile<CoreTile>(0, 2)
    %c1 = aie.logical_tile<CoreTile>(0, 3)
    aie.flow(%c0, DMA : 0, %c1, DMA : 0)
    aie.flow(%c0, DMA : 0, %c1, DMA : 0)
  }
}

// -----

// The same packet flow declared twice, as reported in issue #3706.
// CHECK: error{{.*}}'aie.packet_flow' op duplicates an earlier packet flow; ID 0 under mask 0x1F is already declared between the same sources and destinations
// CHECK: note:{{.*}}the other packet flow is here
module @packet_flow_dup {
  aie.device(npu2) {
    %shim = aie.tile(2, 0)
    %core = aie.tile(2, 2)
    aie.packet_flow(0) {
      aie.packet_source<%shim, DMA : 0>
      aie.packet_dest<%core, DMA : 0>
    } {keep_pkt_header = false}
    aie.packet_flow(0) {
      aie.packet_source<%shim, DMA : 0>
      aie.packet_dest<%core, DMA : 0>
    } {keep_pkt_header = false}
  }
}

// -----

// Sources and destinations are unordered within a packet flow, so listing the
// same endpoints in a different order is still the same flow. Disagreeing
// keep_pkt_header attributes make the redeclaration contradictory, not merely
// redundant, so the attributes are deliberately not part of the key.
// CHECK: error{{.*}}'aie.packet_flow' op duplicates an earlier packet flow; ID 3 under mask 0x1F is already declared between the same sources and destinations
module @packet_flow_dup_reordered {
  aie.device(npu2) {
    %shim = aie.tile(2, 0)
    %c2 = aie.tile(2, 2)
    %c3 = aie.tile(2, 3)
    aie.packet_flow(3) {
      aie.packet_source<%shim, DMA : 0>
      aie.packet_dest<%c2, DMA : 0>
      aie.packet_dest<%c3, DMA : 0>
    } {keep_pkt_header = false}
    aie.packet_flow(3) {
      aie.packet_source<%shim, DMA : 0>
      aie.packet_dest<%c3, DMA : 0>
      aie.packet_dest<%c2, DMA : 0>
    } {keep_pkt_header = true}
  }
}

// -----

// Distinct flows that only look similar must all be accepted: a shared source
// broadcasting to different destinations, a shared destination fed from
// different sources, the same endpoints on a different channel, and the same
// endpoints in a different device.
// CHECK-NOT: error
module @flow_no_false_positives {
  aie.device(npu2) @main {
    %shim = aie.tile(0, 0)
    %c2 = aie.tile(0, 2)
    %c3 = aie.tile(0, 3)
    aie.flow(%shim, DMA : 0, %c2, DMA : 0)
    aie.flow(%shim, DMA : 0, %c3, DMA : 0)
    aie.flow(%c2, DMA : 0, %c3, DMA : 1)
    aie.flow(%shim, DMA : 1, %c3, DMA : 1)
    aie.flow(%shim, DMA : 1, %c2, DMA : 1)
  }
  aie.device(npu2) @second {
    %shim = aie.tile(0, 0)
    %c2 = aie.tile(0, 2)
    aie.flow(%shim, DMA : 0, %c2, DMA : 0)
  }
}

// -----

// Packet flows sharing an ID are legal as long as they connect different
// endpoints, and unplaced logical tiles are skipped until placement runs.
// CHECK-NOT: error
module @packet_flow_no_false_positives {
  aie.device(npu2) @main {
    %shim = aie.tile(2, 0)
    %c2 = aie.tile(2, 2)
    %c3 = aie.tile(2, 3)
    aie.packet_flow(0) {
      aie.packet_source<%shim, DMA : 0>
      aie.packet_dest<%c2, DMA : 0>
    }
    aie.packet_flow(0) {
      aie.packet_source<%shim, DMA : 0>
      aie.packet_dest<%c3, DMA : 0>
    }
    aie.packet_flow(1) {
      aie.packet_source<%shim, DMA : 0>
      aie.packet_dest<%c2, DMA : 0>
    }
  }
  aie.device(npu2) @unplaced {
    %u0 = aie.logical_tile<CoreTile>(?, ?)
    %u1 = aie.logical_tile<CoreTile>(?, ?)
    aie.packet_flow(0) {
      aie.packet_source<%u0, DMA : 0>
      aie.packet_dest<%u1, DMA : 0>
    }
    aie.packet_flow(0) {
      aie.packet_source<%u0, DMA : 0>
      aie.packet_dest<%u1, DMA : 0>
    }
  }
}

// -----

// Two flows carrying one ID between one pair of endpoints under different
// masks claim different sets of packets, so neither duplicates the other.
// The first claims 0x0 through 0x3, the second 0x0 alone.
// CHECK-LABEL: @packet_flow_distinct_masks
module @packet_flow_distinct_masks {
  aie.device(npu2) {
    %shim = aie.tile(2, 0)
    %core = aie.tile(2, 2)
    aie.packet_flow(0 mask 28) {
      aie.packet_source<%shim, DMA : 0>
      aie.packet_dest<%core, DMA : 0>
    }
    aie.packet_flow(0) {
      aie.packet_source<%shim, DMA : 0>
      aie.packet_dest<%core, DMA : 0>
    }
  }
}
