//===- good-out-of-order-channel.mlir --------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --aie-dma-tasks-to-npu %s | FileCheck %s

module {
  aie.device(npu2) {
    %tile_0_0 = aie.tile(0, 0)
    aie.runtime_sequence(%arg0: memref<4xi32>) {
      // Shim S2MM channel 0 CTRL register is at local offset 0x1D200
      // (= 119296); the Enable_Out_of_Order field is bit 3 (value 8, mask 8).
      // CHECK-DAG: %[[ADDR:.*]] = arith.constant 119296 : i32
      // CHECK-DAG: %[[VAL:.*]] = arith.constant 8 : i32
      // CHECK-DAG: %[[MASK:.*]] = arith.constant 8 : i32
      // CHECK: aiex.npu.maskwrite32(%[[ADDR]], %[[VAL]], %[[MASK]])
      %t = aiex.dma_configure_task(%tile_0_0, S2MM, 0, <pkt_type = 0, pkt_id = 0>) {
        aie.dma_bd(%arg0 : memref<4xi32> offset = 0 len = 4) {bd_id = 0 : i32}
        aie.end
      } {out_of_order}
      aiex.dma_start_task(%t)
    }
  }
}

// -----

module {
  aie.device(npu2) {
    %tile_0_1 = aie.tile(0, 1)
    %buf = aie.buffer(%tile_0_1) { address = 0xBEEF : i32 } : memref<32xi8>
    aie.runtime_sequence(%arg0: memref<4xi32>) {
      // On memtile S2MM receiver.
      // CHECK-DAG: %[[ADDR:.*]] = arith.constant 656896 : i32
      // CHECK-DAG: %[[VAL:.*]] = arith.constant 8 : i32
      // CHECK-DAG: %[[MASK:.*]] = arith.constant 8 : i32
      // CHECK: aiex.npu.maskwrite32(%[[ADDR]], %[[VAL]], %[[MASK]])
      %t = aiex.dma_configure_task(%tile_0_1, S2MM, 0, <pkt_type = 0, pkt_id = 0>) {
        aie.dma_bd(%buf : memref<32xi8> offset = 0 len = 4) {bd_id = 0 : i32}
        aie.end
      } {out_of_order}
      aiex.dma_start_task(%t)
    }
  }
}

// -----

// A release-only lock on an out-of-order receive BD.
// CHECK-LABEL: @test_release_only_ooo
module @test_release_only_ooo {
  aie.device(npu2) {
    %tile_0_0 = aie.tile(0, 0)
    %lock = aie.lock(%tile_0_0, 0) {init = 0 : i32}
    aie.runtime_sequence(%arg0: memref<4xi32>) {
      // CHECK: aiex.npu.writebd
      // CHECK-SAME: lock_acq_enable = 0
      // CHECK-SAME: lock_acq_id = 0
      // CHECK-SAME: lock_acq_val = 0
      // CHECK-SAME: lock_rel_id = 0
      // CHECK-SAME: lock_rel_val = 1
      %t = aiex.dma_configure_task(%tile_0_0, S2MM, 0, <pkt_type = 0, pkt_id = 0>) {
        %c1 = arith.constant 1 : i32
        aie.dma_bd(%arg0 : memref<4xi32> offset = 0 len = 4) {bd_id = 0 : i32}
        aie.use_lock(%lock, Release, %c1)
        aie.end
      } {out_of_order}
      aiex.dma_start_task(%t)
    }
  }
}
