// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt %s --split-input-file -aie-substitute-shim-dma-allocations -aie-assign-runtime-sequence-bd-ids -verify-diagnostics | FileCheck %s --implicit-check-not=aiex.set_device

// The control task and the live data copy occupy different BDs.
// CHECK-LABEL: @live
// CHECK: aiex.dma_configure_task({{.*}}, MM2S, 0)
// CHECK-NEXT: aie.dma_bd({{.*}}) {bd_id = 1 : i32
module {
  aie.device(npu2_1col) @cfg {
  }
  aie.device(npu2_1col) {
    %t00 = aie.tile(0, 0)
    aie.shim_dma_allocation @ctrl(%t00, MM2S, 0)
    aie.shim_dma_allocation @data(%t00, MM2S, 1)
    aie.runtime_sequence @live(%buf : memref<16xi32>) {
      %c0 = arith.constant 0 : i32
      %c1 = arith.constant 1 : i32
      // expected-note @+1 {{unfinished transfer}}
      aiex.npu.dma_memcpy_nd(%buf[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0, 1]) {metadata = @data, id = 0 : i64, issue_token = true} : memref<16xi32>
      // expected-warning @+1 {{configures a device while runtime DMA transfers may still be in flight}}
      aiex.set_device @cfg
      %t = aiex.dma_configure_task_for @ctrl {
        aie.dma_bd(%buf : memref<16xi32> offset = 0 len = 16)
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t)
      aiex.dma_await_task(%t)
      aiex.npu.sync(%c0, %c0, %c1, %c1, %c1, %c1) : i32, i32, i32, i32, i32, i32
    }
  }
}

// -----

// The second control task reuses the first control task's BD.
// CHECK-LABEL: @twice
// CHECK: aiex.dma_configure_task({{.*}}, MM2S, 0)
// CHECK-NEXT: aie.dma_bd({{.*}}) {bd_id = 1 : i32
// CHECK: aiex.dma_configure_task({{.*}}, MM2S, 0)
// CHECK-NEXT: aie.dma_bd({{.*}}) {bd_id = 1 : i32
module {
  aie.device(npu2_1col) @cfg {
  }
  aie.device(npu2_1col) {
    %t00 = aie.tile(0, 0)
    aie.shim_dma_allocation @ctrl(%t00, MM2S, 0)
    aie.shim_dma_allocation @data(%t00, MM2S, 1)
    aie.runtime_sequence @twice(%buf : memref<16xi32>) {
      %c0 = arith.constant 0 : i32
      %c1 = arith.constant 1 : i32
      // expected-note @+1 {{unfinished transfer}}
      aiex.npu.dma_memcpy_nd(%buf[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0, 1]) {metadata = @data, id = 0 : i64, issue_token = true} : memref<16xi32>
      // expected-warning @+1 {{configures a device while runtime DMA transfers may still be in flight}}
      aiex.set_device @cfg
      %t = aiex.dma_configure_task_for @ctrl {
        aie.dma_bd(%buf : memref<16xi32> offset = 0 len = 16)
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t)
      aiex.dma_await_task(%t)
      %u = aiex.dma_configure_task_for @ctrl {
        aie.dma_bd(%buf : memref<16xi32> offset = 0 len = 16)
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%u)
      aiex.dma_await_task(%u)
      aiex.npu.sync(%c0, %c0, %c1, %c1, %c1, %c1) : i32, i32, i32, i32, i32, i32
    }
  }
}

// -----

// The sync releases the data copy's BD before the control task.
// CHECK-LABEL: @synced
// CHECK: aiex.dma_configure_task({{.*}}, MM2S, 0)
// CHECK-NEXT: aie.dma_bd({{.*}}) {bd_id = 0 : i32
module {
  aie.device(npu2_1col) @cfg {
  }
  aie.device(npu2_1col) {
    %t00 = aie.tile(0, 0)
    aie.shim_dma_allocation @ctrl(%t00, MM2S, 0)
    aie.shim_dma_allocation @data(%t00, MM2S, 1)
    aie.runtime_sequence @synced(%buf : memref<16xi32>) {
      %c0 = arith.constant 0 : i32
      %c1 = arith.constant 1 : i32
      aiex.npu.dma_memcpy_nd(%buf[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0, 1]) {metadata = @data, id = 0 : i64, issue_token = true} : memref<16xi32>
      aiex.npu.sync(%c0, %c0, %c1, %c1, %c1, %c1) : i32, i32, i32, i32, i32, i32
      aiex.set_device @cfg
      %t = aiex.dma_configure_task_for @ctrl {
        aie.dma_bd(%buf : memref<16xi32> offset = 0 len = 16)
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t)
      aiex.dma_await_task(%t)
    }
    aie.runtime_sequence @task_live(%buf : memref<16xi32>) {
      aiex.set_device @cfg
      // expected-note @+1 {{unfinished transfer}}
      %task = aiex.dma_configure_task_for @data {
        aie.dma_bd(%buf : memref<16xi32> offset = 0 len = 16)
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%task)
      // expected-warning @+1 {{configures a device while runtime DMA transfers may still be in flight}}
      aiex.set_device @cfg
      aiex.dma_await_task(%task)
      aiex.set_device @cfg
    }
  }
}

// -----

// A load_pdi between them resets the array.
// CHECK-LABEL: @reset
// CHECK: aiex.dma_configure_task({{.*}}, MM2S, 0)
// CHECK-NEXT: aie.dma_bd({{.*}}) {bd_id = 0 : i32
module {
  aie.device(npu2_1col) @cfg {
  }
  aie.device(npu2_1col) @other {
    %t00 = aie.tile(0, 0)
  }
  aie.device(npu2_1col) {
    %t00 = aie.tile(0, 0)
    aie.shim_dma_allocation @ctrl(%t00, MM2S, 0)
    aie.shim_dma_allocation @data(%t00, MM2S, 1)
    aie.runtime_sequence @reset(%buf : memref<16xi32>) {
      %c0 = arith.constant 0 : i32
      %c1 = arith.constant 1 : i32
      aiex.npu.dma_memcpy_nd(%buf[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0, 1]) {metadata = @data, id = 0 : i64, issue_token = true} : memref<16xi32>
      aiex.npu.load_pdi {device_ref = @other}
      aiex.set_device @cfg
      %t = aiex.dma_configure_task_for @ctrl {
        aie.dma_bd(%buf : memref<16xi32> offset = 0 len = 16)
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t)
      aiex.dma_await_task(%t)
      aiex.npu.sync(%c0, %c0, %c1, %c1, %c1, %c1) : i32, i32, i32, i32, i32, i32
    }
  }
}
