//===- configured_static_bds.mlir ------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s --split-input-file -aie-substitute-shim-dma-allocations -aie-assign-runtime-sequence-bd-ids="warn-unreleased-transfers=false" -verify-diagnostics | FileCheck %s

// @cfg's static DMA takes BD 0, which a running transfer still holds when its
// control packets start. Once that transfer is awaited, BD 0 stays @cfg's.

// CHECK-LABEL: @held
// CHECK:       aiex.dma_configure_task({{.*}}, MM2S, 1)
// CHECK-NEXT:    aie.dma_bd({{.*}}) {bd_id = 0 : i32
// CHECK:       aiex.dma_configure_task({{.*}}, MM2S, 0)
// CHECK-NEXT:    aie.dma_bd({{.*}}) {bd_id = 1 : i32
// CHECK:       aiex.dma_configure_task({{.*}}, MM2S, 1)
// CHECK-NEXT:    aie.dma_bd({{.*}}) {bd_id = 1 : i32
module {
  aie.device(npu2_1col) @cfg {
    %t00 = aie.tile(0, 0)
    %buf = aie.external_buffer {sym_name = "buf"} : memref<16xi32>
    %dma = aie.shim_dma(%t00) {
      %0 = aie.dma_start(S2MM, 1, ^bd0, ^end)
      ^bd0:
        aie.dma_bd(%buf : memref<16xi32> offset = 0 len = 16) {bd_id = 0 : i32, next_bd_id = 0 : i32}
        aie.next_bd ^bd0
      ^end:
        aie.end
    }
  }
  aie.device(npu2_1col) {
    %t00 = aie.tile(0, 0)
    aie.shim_dma_allocation @ctrl(%t00, MM2S, 0)
    aie.shim_dma_allocation @data(%t00, MM2S, 1)
    aie.runtime_sequence @held(%buf : memref<16xi32>) {
      %d = aiex.dma_configure_task_for @data {
        aie.dma_bd(%buf : memref<16xi32> offset = 0 len = 16)
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%d)
      // expected-note @-5 {{the transfer is configured here}}
      // expected-warning @+1 {{configures @cfg, whose static DMAs take buffer descriptor ID 0 on tile (0,0), which a transfer not known to be finished holds}}
      %c = aiex.dma_configure_task_for @ctrl {
        aie.dma_bd(%buf : memref<16xi32> offset = 0 len = 16)
        aie.end
      } {issue_token = true, aiex.ctrl_pkt_transfer = @cfg}
      aiex.dma_start_task(%c)
      aiex.dma_await_task(%c)
      aiex.dma_await_task(%d)
      %e = aiex.dma_configure_task_for @data {
        aie.dma_bd(%buf : memref<16xi32> offset = 0 len = 16)
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%e)
      aiex.dma_await_task(%e)
    }
  }
}

// -----

// @b's configuration follows @a's teardown, so only @b's static BDs are taken.

// CHECK-LABEL: @replaced
// CHECK:       aiex.dma_configure_task({{.*}}, MM2S, 0)
// CHECK-NEXT:    aie.dma_bd({{.*}}) {bd_id = 2 : i32
// CHECK:       aiex.dma_configure_task({{.*}}, MM2S, 0)
// CHECK-NEXT:    aie.dma_bd({{.*}}) {bd_id = 2 : i32
// CHECK:       aiex.dma_configure_task({{.*}}, MM2S, 0)
// CHECK-NEXT:    aie.dma_bd({{.*}}) {bd_id = 0 : i32
module {
  aie.device(npu2_1col) @a {
    %t00 = aie.tile(0, 0)
    %buf = aie.external_buffer {sym_name = "buf"} : memref<16xi32>
    %dma = aie.shim_dma(%t00) {
      %0 = aie.dma_start(S2MM, 1, ^bd0, ^end)
      ^bd0:
        aie.dma_bd(%buf : memref<16xi32> offset = 0 len = 16) {bd_id = 0 : i32, next_bd_id = 1 : i32}
        aie.next_bd ^bd1
      ^bd1:
        aie.dma_bd(%buf : memref<16xi32> offset = 0 len = 16) {bd_id = 1 : i32, next_bd_id = 0 : i32}
        aie.next_bd ^bd0
      ^end:
        aie.end
    }
  }
  aie.device(npu2_1col) @b {
    %t00 = aie.tile(0, 0)
    %buf = aie.external_buffer {sym_name = "buf"} : memref<16xi32>
    %dma = aie.shim_dma(%t00) {
      %0 = aie.dma_start(S2MM, 1, ^bd2, ^end)
      ^bd2:
        aie.dma_bd(%buf : memref<16xi32> offset = 0 len = 16) {bd_id = 2 : i32, next_bd_id = 2 : i32}
        aie.next_bd ^bd2
      ^end:
        aie.end
    }
  }
  aie.device(npu2_1col) {
    %t00 = aie.tile(0, 0)
    aie.shim_dma_allocation @ctrl(%t00, MM2S, 0)
    aie.runtime_sequence @replaced(%buf : memref<16xi32>) {
      %ca = aiex.dma_configure_task_for @ctrl {
        aie.dma_bd(%buf : memref<16xi32> offset = 0 len = 16)
        aie.end
      } {issue_token = true, aiex.ctrl_pkt_transfer = @a}
      aiex.dma_start_task(%ca)
      aiex.dma_await_task(%ca)
      %ta = aiex.dma_configure_task_for @ctrl {
        aie.dma_bd(%buf : memref<16xi32> offset = 0 len = 16)
        aie.end
      } {issue_token = true, aiex.ctrl_pkt_transfer = @a}
      aiex.dma_start_task(%ta)
      aiex.dma_await_task(%ta)
      %cb = aiex.dma_configure_task_for @ctrl {
        aie.dma_bd(%buf : memref<16xi32> offset = 0 len = 16)
        aie.end
      } {issue_token = true, aiex.ctrl_pkt_transfer = @b}
      aiex.dma_start_task(%cb)
      aiex.dma_await_task(%cb)
    }
  }
}
