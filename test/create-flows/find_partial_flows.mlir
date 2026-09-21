//===- find_partial_flows.mlir ----------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// --aie-find-flows lifts every configured connection into a flow, including
// partial flows whose source or destination is a directional switchbox port
// rather than a core or DMA (array edges, off-fabric consumers, PLIO/edge
// entries, or packet routes a running design steers by packet id).  With
// keep-partial-flows=false only flows that begin and end on a core/DMA are
// recovered.

// RUN: aie-opt --aie-find-flows --split-input-file %s | FileCheck %s
// RUN: aie-opt --aie-find-flows=keep-partial-flows=false --split-input-file %s | FileCheck %s --check-prefix=NONE

// A circuit-switched connection driven out an edge port (North:0 has no wire).
// NONE-NOT: aie.flow
// CHECK-LABEL: aie.device
// CHECK: %[[T02:.*]] = aie.tile(0, 2)
// CHECK: aie.flow(%[[T02]], DMA : 0, %[[T02]], North : 0)
module {
  aie.device(xcvc1902) {
    %t02 = aie.tile(0, 2)
    %sb02 = aie.switchbox(%t02) {
      aie.connect<DMA : 0, North : 0>
    }
    aie.wire(%t02 : DMA, %sb02 : DMA)
  }
}

// -----

// Time-multiplexed packet routing: one source fans out to different edge ports
// by packet id.  Each id recovers a partial packet flow to its output port.
// NONE-NOT: aie.packet_flow
// CHECK-LABEL: aie.device
// CHECK: %[[T12:.*]] = aie.tile(0, 2)
// CHECK: aie.packet_flow(0)
// CHECK:   aie.packet_source<%[[T12]], DMA : 0>
// CHECK:   aie.packet_dest<%[[T12]], North : 0>
// CHECK: aie.packet_flow(1)
// CHECK:   aie.packet_source<%[[T12]], DMA : 0>
// CHECK:   aie.packet_dest<%[[T12]], East : 0>
module {
  aie.device(xcvc1902) {
    %t02 = aie.tile(0, 2)
    %sb02 = aie.switchbox(%t02) {
      %a0 = aie.amsel<0> (0)
      %a1 = aie.amsel<0> (1)
      aie.masterset(North : 0, %a0)
      aie.masterset(East : 0, %a1)
      aie.packet_rules(DMA : 0) {
        aie.rule(31, 0, %a0)
        aie.rule(31, 1, %a1)
      }
    }
    aie.wire(%t02 : DMA, %sb02 : DMA)
  }
}

// -----

// Pure transit: a switchbox forwards West:0 -> East:0 with neither endpoint on
// a core or DMA.  The source side seeds from the (undriven) West:0 input and
// the destination side from the (unwired) East:0 output, lifting the whole
// segment into one partial flow so the switchbox can be dropped.
// NONE-NOT: aie.flow
// CHECK-LABEL: aie.device
// CHECK: %[[T22:.*]] = aie.tile(2, 2)
// CHECK: aie.flow(%[[T22]], West : 0, %[[T22]], East : 0)
module {
  aie.device(xcvc1902) {
    %t22 = aie.tile(2, 2)
    %sb22 = aie.switchbox(%t22) {
      aie.connect<West : 0, East : 0>
    }
  }
}
