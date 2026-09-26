// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// A runtime repeat_count is packed into the queue push's 8-bit field with a
// 0xFF mask, so without a guard 256 would push a task that runs once. The
// builder must refuse it instead, on a mem tile and on a shim tile alike.

// REQUIRES: peano
// RUN: aie-opt --aie-prepare-buffers --aie-assign-buffer-addresses \
// RUN:   --aie-dma-tasks-to-npu --aie-dma-to-npu %s -o %t.mlir
// RUN: FileCheck %s < %t.mlir
// RUN: aie-translate --aie-npu-to-cpp %t.mlir > %t.h
// RUN: %host_clang -std=c++17 -I%S/../../../../include -DGEN_HDR='"%t.h"' \
// RUN:   %S/Inputs/runtime_repeat_guard.cpp %host_link_flags -o %t.exe
// RUN: %t.exe

module {
  aie.device(npu2) {
    %shim = aie.tile(0, 0)
    %mem = aie.tile(0, 1)
    %buf = aie.buffer(%mem) : memref<1024xi32>

    // CHECK-LABEL: @mem_repeat
    // CHECK: aiex.npu.assert_bd_field(%arg0) {max = 255 : i32}
    // CHECK: aiex.npu.write32
    aie.runtime_sequence @mem_repeat(%repeat: i32) {
      %t = aiex.dma_configure_task(%mem, MM2S, 0) repeat %repeat : i32 {
        aie.dma_bd(%buf : memref<1024xi32> offset = 0 len = 1024) {bd_id = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t)
    }

    // CHECK-LABEL: @shim_repeat
    // CHECK: aiex.npu.assert_bd_field(%arg1) {max = 255 : i32}
    // CHECK: aiex.npu.write32
    aie.runtime_sequence @shim_repeat(%in: memref<1024xi32>, %repeat: i32) {
      %t = aiex.dma_configure_task(%shim, MM2S, 0) repeat %repeat : i32 {
        aie.dma_bd(%in : memref<1024xi32> offset = 0 len = 1024) {bd_id = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t)
    }

    // A constant repeat_count is verifier-checked, so it gets no guard.
    // CHECK-LABEL: @const_repeat
    // CHECK-NOT: aiex.npu.assert_bd_field
    // CHECK: aiex.npu.write32(%{{.*}}, %c16711680_i32)
    aie.runtime_sequence @const_repeat() {
      %t = aiex.dma_configure_task(%mem, MM2S, 0) {
        aie.dma_bd(%buf : memref<1024xi32> offset = 0 len = 1024) {bd_id = 0 : i32}
        aie.end
      } {repeat_count = 255 : i32}
      aiex.dma_start_task(%t)
    }
  }
}
