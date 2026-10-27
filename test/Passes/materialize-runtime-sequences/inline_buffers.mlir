//===- inline_buffers.mlir --------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-materialize-runtime-sequences --split-input-file %s | FileCheck %s
// RUN: aie-opt --aie-materialize-runtime-sequences --split-input-file %s | FileCheck %s --check-prefix=NAMES

// A DMA task that names an aie.buffer inlines a clone of that buffer, keeping
// its address. Each definition is cloned once however often it is run, and a
// clone whose name the caller already holds is renamed.

// CHECK-LABEL: aie.device(npu2) {
// CHECK: %[[MT:.*]] = aie.tile(0, 1)
// CHECK-DAG: aie.buffer @[[BUFFER_A:b_mt(_[0-9]+)?]](%[[MT]]) {address = 0 : i32} : memref<256xi32>
// CHECK-DAG: aie.lock @[[LOCK_A:b_lock(_[0-9]+)?]](%[[MT]], 0) {init = 1 : i32}
// CHECK-DAG: aie.buffer @[[BUFFER_B:b_mt(_[0-9]+)?]](%[[MT]]) {address = 1024 : i32} : memref<128xi32>
// CHECK-DAG: aie.lock @[[LOCK_B:b_lock(_[0-9]+)?]](%[[MT]], 1) {init = 1 : i32}
// CHECK-NOT: aie.buffer
// CHECK-NOT: aie.lock
// CHECK: aie.runtime_sequence
// CHECK: aiex.npu.load_pdi {device_ref = @a}
// CHECK: aie.use_lock(@[[LOCK_A]], AcquireGreaterEqual
// CHECK-NEXT: %[[ACCESS_A:.*]] = aie.get_buffer @[[BUFFER_A]] : memref<256xi32>
// CHECK-NEXT: aie.dma_bd(%[[ACCESS_A]] : memref<256xi32>
// CHECK: aie.use_lock(@[[LOCK_A]], AcquireGreaterEqual
// CHECK-NEXT: %[[ACCESS_A_AGAIN:.*]] = aie.get_buffer @[[BUFFER_A]] : memref<256xi32>
// CHECK-NEXT: aie.dma_bd(%[[ACCESS_A_AGAIN]] : memref<256xi32>
// CHECK: aiex.npu.load_pdi {device_ref = @b}
// CHECK: aie.use_lock(@[[LOCK_B]], AcquireGreaterEqual
// CHECK-NEXT: %[[ACCESS_B:.*]] = aie.get_buffer @[[BUFFER_B]] : memref<128xi32>
// CHECK-NEXT: aie.dma_bd(%[[ACCESS_B]] : memref<128xi32>
// CHECK-LABEL: aie.device(npu2) @a

// NAMES-LABEL: aie.device(npu2) {
// NAMES-DAG: aie.buffer @b_mt{{.*}}}
// NAMES-DAG: aie.buffer @b_mt_{{[0-9]+}}{{.*}}}
// NAMES-DAG: aie.lock @b_lock{{.*}}}
// NAMES-DAG: aie.lock @b_lock_{{[0-9]+}}{{.*}}}
// NAMES-LABEL: aie.device(npu2) @a
module {
  aie.device(npu2) {
    aie.runtime_sequence(%arg0: memref<64xi32>) {
      aiex.configure @a {
        aiex.run @sequence(%arg0) : (memref<64xi32>)
        aiex.run @sequence(%arg0) : (memref<64xi32>)
      }
      aiex.configure @b {
        aiex.run @sequence(%arg0) : (memref<64xi32>)
      }
    }
  }

  aie.device(npu2) @a {
    %mt = aie.tile(0, 1)
    aie.buffer @b_mt(%mt) {address = 0 : i32} : memref<256xi32>
    aie.lock @b_lock(%mt, 0) {init = 1 : i32}
    aie.runtime_sequence(%arg0: memref<64xi32>) {
      %t = aiex.dma_configure_task(%mt, MM2S, 0) {
        %c1 = arith.constant 1 : i32
        aie.use_lock(@b_lock, AcquireGreaterEqual, %c1)
        %buf = aie.get_buffer @b_mt : memref<256xi32>
        aie.dma_bd(%buf : memref<256xi32> len = 256)
        aie.end
      }
      aiex.dma_start_task(%t)
    }
  }

  aie.device(npu2) @b {
    %mt = aie.tile(0, 1)
    aie.buffer @b_mt(%mt) {address = 1024 : i32} : memref<128xi32>
    aie.lock @b_lock(%mt, 1) {init = 1 : i32}
    aie.runtime_sequence(%arg0: memref<64xi32>) {
      %t = aiex.dma_configure_task(%mt, MM2S, 0) {
        %c1 = arith.constant 1 : i32
        aie.use_lock(@b_lock, AcquireGreaterEqual, %c1)
        %buf = aie.get_buffer @b_mt : memref<128xi32>
        aie.dma_bd(%buf : memref<128xi32> len = 128)
        aie.end
      }
      aiex.dma_start_task(%t)
    }
  }
}

// -----

// A buffer named both by symbol and by SSA is cloned once.

// CHECK-LABEL: aie.device(npu2) {
// CHECK: %[[MT:.*]] = aie.tile(0, 1)
// CHECK-NEXT: aie.buffer @[[BUF:b_mt]](%[[MT]]) {address = 0 : i32} : memref<256xi32>
// CHECK-NOT: aie.buffer
// CHECK: aiex.npu.rtp_write(@b_mt, 0,
// CHECK: %[[BUF_ACCESS_0:.*]] = aie.get_buffer @[[BUF]] : memref<256xi32>
// CHECK: aie.dma_bd(%[[BUF_ACCESS_0]] : memref<256xi32>
module {
  aie.device(npu2) {
    aie.runtime_sequence(%arg0: memref<64xi32>) {
      aiex.configure @a {
        aiex.run @sequence(%arg0) : (memref<64xi32>)
      }
    }
  }

  aie.device(npu2) @a {
    %mt = aie.tile(0, 1)
    aie.buffer @b_mt(%mt) {address = 0 : i32} : memref<256xi32>
    aie.runtime_sequence(%arg0: memref<64xi32>) {
      %c7 = arith.constant 7 : i32
      aiex.npu.rtp_write(@b_mt, 0, %c7) : i32
      %t = aiex.dma_configure_task(%mt, MM2S, 0) {
        %buf = aie.get_buffer @b_mt : memref<256xi32>
        aie.dma_bd(%buf : memref<256xi32> len = 256)
        aie.end
      }
      aiex.dma_start_task(%t)
    }
  }
}
