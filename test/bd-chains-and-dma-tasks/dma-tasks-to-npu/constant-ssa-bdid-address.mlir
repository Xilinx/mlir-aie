// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt --aie-dma-tasks-to-npu --canonicalize %s | FileCheck %s

// A constant SSA id must select the same BD for configuration and address
// patching, even though the dma_bd has no bd_id attribute.
module {
  aie.device(npu2) {
    %mem = aie.tile(0, 1)
    %core = aie.tile(0, 2)
    %shim = aie.tile(0, 0)
    %mem_buf = aie.buffer(%mem) {address = 4096 : i32} : memref<256xi32>
    %core_buf = aie.buffer(%core) {address = 4096 : i32} : memref<256xi32>

    // CHECK-LABEL: @mem_id
    // CHECK-DAG: %[[MEM_BASE:.*]] = arith.constant 1704032 : i32
    // CHECK-DAG: %[[MEM_ADDR:.*]] = arith.constant 1704036 : i32
    // CHECK: aiex.npu.blockwrite_values(%[[MEM_BASE]] : i32)
    // CHECK: aiex.npu.maskwrite32(%[[MEM_ADDR]],
    aie.runtime_sequence @mem_id() {
      %id = arith.constant 3 : i32
      %task = aiex.dma_configure_task(%mem, MM2S, 0) {
        aie.dma_bd(%mem_buf : memref<256xi32> offset = 0 len = 256) bd_id_val %id : i32
        aie.end
      }
    }

    // CHECK-LABEL: @core_id
    // CHECK: %[[CORE_ADDR:.*]] = arith.constant 2216032 : i32
    // CHECK: aiex.npu.blockwrite_values(%[[CORE_ADDR]] : i32)
    // CHECK: aiex.npu.maskwrite32(%[[CORE_ADDR]],
    aie.runtime_sequence @core_id() {
      %id = arith.constant 3 : i32
      %task = aiex.dma_configure_task(%core, MM2S, 0) {
        aie.dma_bd(%core_buf : memref<256xi32> offset = 0 len = 256) bd_id_val %id : i32
        aie.end
      }
    }

    // CHECK-LABEL: @shim_id
    // CHECK: %[[SHIM_BASE:.*]] = arith.constant 118880 : i32
    // CHECK: aiex.npu.blockwrite_values(%[[SHIM_BASE]] : i32)
    // CHECK: aiex.npu.address_patch
    // CHECK-SAME: addr = 118884
    aie.runtime_sequence @shim_id(%buf: memref<256xi32>) {
      %id = arith.constant 3 : i32
      %task = aiex.dma_configure_task(%shim, MM2S, 0) {
        aie.dma_bd(%buf : memref<256xi32> offset = 0 len = 256) bd_id_val %id : i32
        aie.end
      }
    }
  }
}
