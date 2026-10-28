//===- generate_column_control_overlay_standalone_tokens.mlir --*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s -aie-generate-column-control-overlay="emit-standalone-overlay=true" | FileCheck %s

// The standalone overlay has no runtime sequence of its own, so it routes the
// token of every tile that any participating device awaits, and each device
// gets that same set so all of them share one overlay shape.

// CHECK-LABEL: aie.device(npu2_1col) @first
// CHECK-DAG: aie.packet_source<%{{.*}}tile_0_1{{.*}}, TileControl : 0>
// CHECK-DAG: aie.packet_source<%{{.*}}tile_0_2{{.*}}, TileControl : 0>
// CHECK-LABEL: aie.device(npu2_1col) @second
// CHECK-DAG: aie.packet_source<%{{.*}}tile_0_1{{.*}}, TileControl : 0>
// CHECK-DAG: aie.packet_source<%{{.*}}tile_0_2{{.*}}, TileControl : 0>
// CHECK-LABEL: aie.device(npu2_1col) @ctrl_pkt_overlay
// CHECK-DAG: aie.packet_source<%{{.*}}tile_0_1{{.*}}, TileControl : 0>
// CHECK-DAG: aie.packet_source<%{{.*}}tile_0_2{{.*}}, TileControl : 0>

module {
  aie.device(npu2_1col) @first {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_1 = aie.tile(0, 1)
    %tile_0_2 = aie.tile(0, 2)
    %buf = aie.buffer(%tile_0_1) : memref<16xi32>
    aie.runtime_sequence(%arg0: memref<16xi32>) {
      %t = aiex.dma_configure_task(%tile_0_1, MM2S, 0) {
        aie.dma_bd(%buf : memref<16xi32> offset = 0 len = 16)
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t)
      aiex.dma_await_task(%t)
    }
  }
  aie.device(npu2_1col) @second {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_1 = aie.tile(0, 1)
    %tile_0_2 = aie.tile(0, 2)
    %buf = aie.buffer(%tile_0_2) : memref<16xi32>
    aie.runtime_sequence(%arg0: memref<16xi32>) {
      %t = aiex.dma_configure_task(%tile_0_2, MM2S, 0) {
        aie.dma_bd(%buf : memref<16xi32> offset = 0 len = 16)
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t)
      aiex.dma_await_task(%t)
    }
  }
}
