// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt --aie-import-overlay %s -o %t
// RUN: FileCheck %s < %t
// RUN: aie-opt --aie-import-overlay %t | diff %t -

// CHECK-LABEL: aie.device(npu2_1col) @application
// CHECK-COUNT-1: aie.tile(0, 2)
// CHECK-COUNT-1: aie.buffer
// CHECK-COUNT-1: aie.packet_flow(3)
// CHECK: {keep_pkt_header = true, priority_route = true}
// CHECK-COUNT-1: aie.lock
// CHECK-COUNT-1: aie.runtime_sequence @transfer
// CHECK: aie.dma_bd(%{{.*}}
// CHECK: overlay = @resources
// CHECK-LABEL: aie.device(npu2_1col) @resources

module {
  aie.device(npu2_1col) @application {
    %tile = aie.tile(0, 2)
    %buf = aie.buffer(%tile) {sym_name = "buf"} : memref<16xi32>
    aie.packet_flow(3) {
      aie.packet_source<%tile, DMA : 0>
      aie.packet_dest<%tile, Core : 0>
    }
  } {overlay = @resources}
  aie.device(npu2_1col) @resources {
    %tile = aie.tile(0, 2)
    %buf = aie.buffer(%tile) {sym_name = "buf"} : memref<16xi32>
    %lock = aie.lock(%tile, 0) {init = 1 : i32}
    aie.packet_flow(3) {
      aie.packet_source<%tile, DMA : 0>
      aie.packet_dest<%tile, Core : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.runtime_sequence @transfer() {
      %task = aiex.dma_configure_task(%tile, MM2S, 0) {
        aie.dma_bd(%buf : memref<16xi32> offset = 0 len = 16)
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%task)
      aiex.dma_await_task(%task)
    }
  }
}
