//===- generate_column_control_overlay.mlir --------------------*- MLIR -*-===//
//
// Copyright (C) 2024 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s -aie-generate-column-control-overlay --split-input-file | FileCheck %s
// RUN: aie-opt %s -aie-generate-column-control-overlay="route-shim-to-tct=all-tiles" --split-input-file | FileCheck %s --check-prefix=TCTALLTILES
// RUN: aie-opt %s -aie-generate-column-control-overlay="route-shim-to-tile-ctrl=true" --split-input-file | FileCheck %s --check-prefix=CTRLPKT

// assign controller ids to aie.tile_op, for control packets

// CHECK-LABEL: module {
// CHECK: %[[tile_0_0:.*]] = aie.tile(0, 0)
// CHECK: %[[tile_0_1:.*]] = aie.tile(0, 1)
// CHECK: aie.packet_flow(15) {
// CHECK:   aie.packet_source<%[[tile_0_0]], TileControl : 0>
// CHECK:   aie.packet_dest<%[[tile_0_0]], South : 0>
// CHECK: }{{.*}}keep_pkt_header = true{{.*}}priority_route = true
// TCTALLTILES-LABEL: module {
// TCTALLTILES: %[[tile_0_0:.*]] = aie.tile(0, 0)
// TCTALLTILES: %[[tile_0_1:.*]] = aie.tile(0, 1)
// TCTALLTILES: aie.packet_flow(15) {
// TCTALLTILES:   aie.packet_source<%[[tile_0_0]], TileControl : 0>
// TCTALLTILES:   aie.packet_dest<%[[tile_0_0]], South : 0>
// TCTALLTILES: }{{.*}}keep_pkt_header = true{{.*}}priority_route = true
// TCTALLTILES: aie.packet_flow(26) {
// TCTALLTILES:   aie.packet_source<%[[tile_0_1]], TileControl : 0>
// TCTALLTILES:   aie.packet_dest<%[[tile_0_0]], South : 0>
// TCTALLTILES: }{{.*}}keep_pkt_header = true{{.*}}priority_route = true
// CTRLPKT-LABEL: module {
// CTRLPKT: %[[tile_0_0:.*]] = aie.tile(0, 0)
// CTRLPKT: %[[tile_0_1:.*]] = aie.tile(0, 1)
// CTRLPKT: aie.packet_flow(15) {
// CTRLPKT:   aie.packet_source<%[[tile_0_0]], DMA : 0>
// CTRLPKT:   aie.packet_dest<%[[tile_0_0]], TileControl : 0>
// CTRLPKT: }
// CTRLPKT: aie.shim_dma_allocation @ctrlpkt_col0_mm2s_chan0(%[[tile_0_0]], MM2S, 0)
// CTRLPKT: aie.packet_flow(26) {
// CTRLPKT:   aie.packet_source<%[[tile_0_0]], DMA : 0>
// CTRLPKT:   aie.packet_dest<%[[tile_0_1]], TileControl : 0>
// CTRLPKT: }

aie.device(npu1_1col) {
  %tile_0_0 = aie.tile(0, 0)
  %tile_0_1 = aie.tile(0, 1)
}

// -----

// two columns

// CHECK-LABEL: module {
// CHECK: %[[tile_0_0:.*]] = aie.tile(0, 0)
// CHECK: %[[tile_0_1:.*]] = aie.tile(0, 1)
// CHECK: %[[tile_1_0:.*]] = aie.tile(1, 0)
// CHECK: %[[tile_1_1:.*]] = aie.tile(1, 1)
// CHECK: aie.packet_flow(15) {
// CHECK:   aie.packet_source<%[[tile_0_0]], TileControl : 0>
// CHECK:   aie.packet_dest<%[[tile_0_0]], South : 0>
// CHECK: }{{.*}}keep_pkt_header = true{{.*}}priority_route = true
// CHECK: aie.packet_flow(15) {
// CHECK:   aie.packet_source<%[[tile_1_0]], TileControl : 0>
// CHECK:   aie.packet_dest<%[[tile_1_0]], South : 0>
// CHECK: }{{.*}}keep_pkt_header = true{{.*}}priority_route = true
// TCTALLTILES-LABEL: module {
// TCTALLTILES: %[[tile_0_0:.*]] = aie.tile(0, 0)
// TCTALLTILES: %[[tile_0_1:.*]] = aie.tile(0, 1)
// TCTALLTILES: %[[tile_1_0:.*]] = aie.tile(1, 0)
// TCTALLTILES: %[[tile_1_1:.*]] = aie.tile(1, 1)
// TCTALLTILES: aie.packet_flow(15) {
// TCTALLTILES:   aie.packet_source<%[[tile_0_0]], TileControl : 0>
// TCTALLTILES:   aie.packet_dest<%[[tile_0_0]], South : 0>
// TCTALLTILES: }{{.*}}keep_pkt_header = true{{.*}}priority_route = true
// TCTALLTILES: aie.packet_flow(26) {
// TCTALLTILES:   aie.packet_source<%[[tile_0_1]], TileControl : 0>
// TCTALLTILES:   aie.packet_dest<%[[tile_0_0]], South : 0>
// TCTALLTILES: }{{.*}}keep_pkt_header = true{{.*}}priority_route = true
// TCTALLTILES: aie.packet_flow(15) {
// TCTALLTILES:   aie.packet_source<%[[tile_1_0]], TileControl : 0>
// TCTALLTILES:   aie.packet_dest<%[[tile_1_0]], South : 0>
// TCTALLTILES: }{{.*}}keep_pkt_header = true{{.*}}priority_route = true
// TCTALLTILES: aie.packet_flow(26) {
// TCTALLTILES:   aie.packet_source<%[[tile_1_1]], TileControl : 0>
// TCTALLTILES:   aie.packet_dest<%[[tile_1_0]], South : 0>
// TCTALLTILES: }{{.*}}keep_pkt_header = true{{.*}}priority_route = true
// CTRLPKT-LABEL: module {
// CTRLPKT: %[[tile_0_0:.*]] = aie.tile(0, 0)
// CTRLPKT: %[[tile_0_1:.*]] = aie.tile(0, 1)
// CTRLPKT: %[[tile_1_0:.*]] = aie.tile(1, 0)
// CTRLPKT: %[[tile_1_1:.*]] = aie.tile(1, 1)
// CTRLPKT: aie.packet_flow(15) {
// CTRLPKT:   aie.packet_source<%[[tile_0_0]], DMA : 0>
// CTRLPKT:   aie.packet_dest<%[[tile_0_0]], TileControl : 0>
// CTRLPKT: }
// CTRLPKT: aie.shim_dma_allocation @ctrlpkt_col0_mm2s_chan0(%[[tile_0_0]], MM2S, 0)
// CTRLPKT: aie.packet_flow(26) {
// CTRLPKT:   aie.packet_source<%[[tile_0_0]], DMA : 0>
// CTRLPKT:   aie.packet_dest<%[[tile_0_1]], TileControl : 0>
// CTRLPKT: }
// CTRLPKT: aie.packet_flow(15) {
// CTRLPKT:   aie.packet_source<%[[tile_1_0]], DMA : 0>
// CTRLPKT:   aie.packet_dest<%[[tile_1_0]], TileControl : 0>
// CTRLPKT: }
// CTRLPKT: aie.shim_dma_allocation @ctrlpkt_col1_mm2s_chan0(%[[tile_1_0]], MM2S, 0)
// CTRLPKT: aie.packet_flow(26) {
// CTRLPKT:   aie.packet_source<%[[tile_1_0]], DMA : 0>
// CTRLPKT:   aie.packet_dest<%[[tile_1_1]], TileControl : 0>
// CTRLPKT: }

aie.device(npu1_2col) {
  %tile_0_0 = aie.tile(0, 0)
  %tile_0_1 = aie.tile(0, 1)
  %tile_1_0 = aie.tile(1, 0)
  %tile_1_1 = aie.tile(1, 1)
}

// -----

// controller_id attribute overriding packet header assignment in aie.packet_flow; 
// round-robin shim dma channel assignment to cover all 5 tiles in a column

// CHECK-LABEL: module {
// CHECK: %[[tile_0_0:.*]] = aie.tile(0, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 4>}
// CHECK: %[[tile_0_1:.*]] = aie.tile(0, 1) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 3>}
// CHECK: %[[tile_0_2:.*]] = aie.tile(0, 2) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 5>}
// CHECK: %[[tile_0_3:.*]] = aie.tile(0, 3) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 1>}
// CHECK: %[[tile_0_4:.*]] = aie.tile(0, 4) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 6>}
// CHECK: %[[tile_0_5:.*]] = aie.tile(0, 5) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 2>}
// CHECK: %[[tile_1_0:.*]] = aie.tile(1, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 5>}
// CHECK: %[[tile_1_1:.*]] = aie.tile(1, 1) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 7>}
// CHECK: aie.packet_flow(4) {
// CHECK:   aie.packet_source<%[[tile_0_0]], TileControl : 0>
// CHECK:   aie.packet_dest<%[[tile_0_0]], South : 0>
// CHECK: }{{.*}}keep_pkt_header = true{{.*}}priority_route = true
// CHECK: aie.packet_flow(5) {
// CHECK:   aie.packet_source<%[[tile_1_0]], TileControl : 0>
// CHECK:   aie.packet_dest<%[[tile_1_0]], South : 0>
// CHECK: }{{.*}}keep_pkt_header = true{{.*}}priority_route = true
// TCTALLTILES-LABEL: module {
// TCTALLTILES: %[[tile_0_0:.*]] = aie.tile(0, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 4>}
// TCTALLTILES: %[[tile_0_1:.*]] = aie.tile(0, 1) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 3>}
// TCTALLTILES: %[[tile_0_2:.*]] = aie.tile(0, 2) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 5>}
// TCTALLTILES: %[[tile_0_3:.*]] = aie.tile(0, 3) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 1>}
// TCTALLTILES: %[[tile_0_4:.*]] = aie.tile(0, 4) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 6>}
// TCTALLTILES: %[[tile_0_5:.*]] = aie.tile(0, 5) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 2>}
// TCTALLTILES: %[[tile_1_0:.*]] = aie.tile(1, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 5>}
// TCTALLTILES: %[[tile_1_1:.*]] = aie.tile(1, 1) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 7>}
// TCTALLTILES: aie.packet_flow(4) {
// TCTALLTILES:   aie.packet_source<%[[tile_0_0]], TileControl : 0>
// TCTALLTILES:   aie.packet_dest<%[[tile_0_0]], South : 0>
// TCTALLTILES: }{{.*}}keep_pkt_header = true{{.*}}priority_route = true
// TCTALLTILES: aie.packet_flow(3) {
// TCTALLTILES:   aie.packet_source<%[[tile_0_1]], TileControl : 0>
// TCTALLTILES:   aie.packet_dest<%[[tile_0_0]], South : 0>
// TCTALLTILES: }{{.*}}keep_pkt_header = true{{.*}}priority_route = true
// TCTALLTILES: aie.packet_flow(5) {
// TCTALLTILES:   aie.packet_source<%[[tile_0_2]], TileControl : 0>
// TCTALLTILES:   aie.packet_dest<%[[tile_0_0]], South : 0>
// TCTALLTILES: }{{.*}}keep_pkt_header = true{{.*}}priority_route = true
// TCTALLTILES: aie.packet_flow(1) {
// TCTALLTILES:   aie.packet_source<%[[tile_0_3]], TileControl : 0>
// TCTALLTILES:   aie.packet_dest<%[[tile_0_0]], South : 0>
// TCTALLTILES: }{{.*}}keep_pkt_header = true{{.*}}priority_route = true
// TCTALLTILES: aie.packet_flow(6) {
// TCTALLTILES:   aie.packet_source<%[[tile_0_4]], TileControl : 0>
// TCTALLTILES:   aie.packet_dest<%[[tile_0_0]], South : 0>
// TCTALLTILES: }{{.*}}keep_pkt_header = true{{.*}}priority_route = true
// TCTALLTILES: aie.packet_flow(2) {
// TCTALLTILES:   aie.packet_source<%[[tile_0_5]], TileControl : 0>
// TCTALLTILES:   aie.packet_dest<%[[tile_0_0]], South : 0>
// TCTALLTILES: }{{.*}}keep_pkt_header = true{{.*}}priority_route = true
// TCTALLTILES: aie.packet_flow(5) {
// TCTALLTILES:   aie.packet_source<%[[tile_1_0]], TileControl : 0>
// TCTALLTILES:   aie.packet_dest<%[[tile_1_0]], South : 0>
// TCTALLTILES: }{{.*}}keep_pkt_header = true{{.*}}priority_route = true
// TCTALLTILES: aie.packet_flow(7) {
// TCTALLTILES:   aie.packet_source<%[[tile_1_1]], TileControl : 0>
// TCTALLTILES:   aie.packet_dest<%[[tile_1_0]], South : 0>
// TCTALLTILES: }{{.*}}keep_pkt_header = true{{.*}}priority_route = true
// CTRLPKT-LABEL: module {
// CTRLPKT: %[[tile_0_0:.*]] = aie.tile(0, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 4>}
// CTRLPKT: %[[tile_0_1:.*]] = aie.tile(0, 1) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 3>}
// CTRLPKT: %[[tile_0_2:.*]] = aie.tile(0, 2) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 5>}
// CTRLPKT: %[[tile_0_3:.*]] = aie.tile(0, 3) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 1>}
// CTRLPKT: %[[tile_0_4:.*]] = aie.tile(0, 4) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 6>}
// CTRLPKT: %[[tile_0_5:.*]] = aie.tile(0, 5) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 2>}
// CTRLPKT: %[[tile_1_0:.*]] = aie.tile(1, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 5>}
// CTRLPKT: %[[tile_1_1:.*]] = aie.tile(1, 1) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 7>}
// CTRLPKT: aie.packet_flow(4) {
// CTRLPKT:   aie.packet_source<%[[tile_0_0]], DMA : 0>
// CTRLPKT:   aie.packet_dest<%[[tile_0_0]], TileControl : 0>
// CTRLPKT: }
// CTRLPKT: aie.shim_dma_allocation @ctrlpkt_col0_mm2s_chan0(%[[tile_0_0]], MM2S, 0)
// CTRLPKT: aie.packet_flow(3) {
// CTRLPKT:   aie.packet_source<%[[tile_0_0]], DMA : 0>
// CTRLPKT:   aie.packet_dest<%[[tile_0_1]], TileControl : 0>
// CTRLPKT: }
// CTRLPKT: aie.packet_flow(5) {
// CTRLPKT:   aie.packet_source<%[[tile_0_0]], DMA : 0>
// CTRLPKT:   aie.packet_dest<%[[tile_0_2]], TileControl : 0>
// CTRLPKT: }
// CTRLPKT: aie.packet_flow(1) {
// CTRLPKT:   aie.packet_source<%[[tile_0_0]], DMA : 1>
// CTRLPKT:   aie.packet_dest<%[[tile_0_3]], TileControl : 0>
// CTRLPKT: }
// CTRLPKT: aie.shim_dma_allocation @ctrlpkt_col0_mm2s_chan1(%[[tile_0_0]], MM2S, 1)
// CTRLPKT: aie.packet_flow(6) {
// CTRLPKT:   aie.packet_source<%[[tile_0_0]], DMA : 1>
// CTRLPKT:   aie.packet_dest<%[[tile_0_4]], TileControl : 0>
// CTRLPKT: }
// CTRLPKT: aie.packet_flow(2) {
// CTRLPKT:   aie.packet_source<%[[tile_0_0]], DMA : 1>
// CTRLPKT:   aie.packet_dest<%[[tile_0_5]], TileControl : 0>
// CTRLPKT: }
// CTRLPKT: aie.packet_flow(5) {
// CTRLPKT:   aie.packet_source<%[[tile_1_0]], DMA : 0>
// CTRLPKT:   aie.packet_dest<%[[tile_1_0]], TileControl : 0>
// CTRLPKT: }
// CTRLPKT: aie.shim_dma_allocation @ctrlpkt_col1_mm2s_chan0(%[[tile_1_0]], MM2S, 0)
// CTRLPKT: aie.packet_flow(7) {
// CTRLPKT:   aie.packet_source<%[[tile_1_0]], DMA : 0>
// CTRLPKT:   aie.packet_dest<%[[tile_1_1]], TileControl : 0>
// CTRLPKT: }

aie.device(npu1_2col) {
  %tile_0_0 = aie.tile(0, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 4>}
  %tile_0_1 = aie.tile(0, 1) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 3>}
  %tile_0_2 = aie.tile(0, 2) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 5>}
  %tile_0_3 = aie.tile(0, 3) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 1>}
  %tile_0_4 = aie.tile(0, 4) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 6>}
  %tile_0_5 = aie.tile(0, 5) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 2>}
  %tile_1_0 = aie.tile(1, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 5>}
  %tile_1_1 = aie.tile(1, 1) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 7>}
}

// -----

// two occupied columns with a gap: flows between column 0 and column 2 route
// through column 1's stream switches, so column 1's switchboxes get configured
// by control packets and need a shim dma allocation of their own. Column 2
// reaches row 5, which maps to the second shim channel, so column 1 has to
// cover both channels and not just the one its own shim row maps to.

// Only the control-packet path covers the gap. Without
// route-shim-to-tile-ctrl the pass runs on every aiecc invocation for every
// target, so it must not reach into a column the design never declared: doing
// so gave such columns routes they previously had none of and left designs
// that used to route with no legal routing at all.
// CHECK-LABEL: module {
// CHECK-NOT: aie.tile(1, {{[0-9]+}})
// CTRLPKT-LABEL: module {
// CTRLPKT-DAG: aie.shim_dma_allocation @ctrlpkt_col0_mm2s_chan0
// CTRLPKT-DAG: aie.shim_dma_allocation @ctrlpkt_col1_mm2s_chan0
// CTRLPKT-DAG: aie.shim_dma_allocation @ctrlpkt_col1_mm2s_chan1
// CTRLPKT-DAG: aie.shim_dma_allocation @ctrlpkt_col2_mm2s_chan0
// CTRLPKT-DAG: aie.shim_dma_allocation @ctrlpkt_col2_mm2s_chan1

aie.device(npu2) {
  %tile_0_0 = aie.tile(0, 0)
  %tile_2_1 = aie.tile(2, 1)
  %tile_2_5 = aie.tile(2, 5)
}

// -----

// shim-only still routes TCTs from a mem or core tile whose runtime task
// issues a token, since an await on that token otherwise never returns.

// CHECK-LABEL: module {
// CHECK-DAG: %[[tile_0_0:.*]] = aie.tile(0, 0)
// CHECK-DAG: %[[tile_0_1:.*]] = aie.tile(0, 1)
// CHECK-DAG: %[[tile_0_2:.*]] = aie.tile(0, 2)
// CHECK-DAG: %[[tile_0_4:.*]] = aie.tile(0, 4)
// CHECK: aie.packet_flow(15) {
// CHECK:   aie.packet_source<%[[tile_0_0]], TileControl : 0>
// CHECK:   aie.packet_dest<%[[tile_0_0]], South : 0>
// CHECK: aie.packet_flow(26) {
// CHECK:   aie.packet_source<%[[tile_0_1]], TileControl : 0>
// CHECK:   aie.packet_dest<%[[tile_0_0]], South : 0>
// CHECK: aie.packet_flow(27) {
// CHECK:   aie.packet_source<%[[tile_0_2]], TileControl : 0>
// CHECK:   aie.packet_dest<%[[tile_0_0]], South : 0>
// CHECK-NOT: aie.packet_source<%{{.*}}tile_0_3{{.*}}, TileControl
// CHECK: aie.packet_flow(30) {
// CHECK:   aie.packet_source<%[[tile_0_4]], TileControl : 0>
// CHECK:   aie.packet_dest<%[[tile_0_0]], South : 0>

aie.device(npu2_1col) {
  %tile_0_0 = aie.tile(0, 0)
  %tile_0_1 = aie.tile(0, 1)
  %tile_0_2 = aie.tile(0, 2)
  %tile_0_3 = aie.tile(0, 3)
  %tile_0_4 = aie.tile(0, 4)
  %buf = aie.buffer(%tile_0_1) : memref<16xi32>
  %buf3 = aie.buffer(%tile_0_3) : memref<16xi32>
  %buf4 = aie.buffer(%tile_0_4) : memref<16xi32>
  aie.route_endpoint @into4(%tile_0_4) DMA
  aie.route_endpoint @from0(%tile_0_0) DMA
  aie.route from @from0 to [@into4]
  aie.runtime_sequence(%arg0: memref<16xi32>) {
    %t = aiex.dma_configure_task(%tile_0_1, MM2S, 0) {
      aie.dma_bd(%buf : memref<16xi32> offset = 0 len = 16)
      aie.end
    } {issue_token = true}
    aiex.dma_start_task(%t)
    aiex.dma_await_task(%t)
    %u = aiex.dma_configure_task(%tile_0_3, MM2S, 0) {
      aie.dma_bd(%buf3 : memref<16xi32> offset = 0 len = 16)
      aie.end
    }
    aiex.dma_start_task(%u)
    %v = aiex.dma_configure_task_for @into4 {
      aie.dma_bd(%buf4 : memref<16xi32> offset = 0 len = 16)
      aie.end
    } {issue_token = true}
    aiex.dma_start_task(%v)
    aiex.dma_await_task(%v)
    %c0 = arith.constant 0 : i32
    aiex.npu.push_queue (0, 2, MM2S:0) bd_id %c0 repeat %c0 {issue_token = true} : i32, i32
  }
}
