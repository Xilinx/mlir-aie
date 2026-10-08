//===- inline_buffers.mlir --------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-materialize-runtime-sequences %s | FileCheck %s
// RUN: aie-opt --aie-materialize-runtime-sequences %s | FileCheck %s --check-prefix=NAMES

// A DMA task that names an aie.buffer inlines a clone of that buffer, keeping
// its address. Each definition is cloned once however often it is run, and a
// clone whose name the caller already holds is renamed.

// CHECK-LABEL: aie.device(npu2) {
// CHECK: %[[MT:.*]] = aie.tile(0, 1)
// CHECK-DAG: %[[BUF_A:.*]] = aie.buffer(%[[MT]]) {address = 0 : i32, sym_name = "{{b_mt(_0)?}}"} : memref<256xi32>
// CHECK-DAG: %[[LOCK_A:.*]] = aie.lock(%[[MT]], 0) {init = 1 : i32, sym_name = "{{b_lock(_0)?}}"}
// CHECK-DAG: %[[BUF_B:.*]] = aie.buffer(%[[MT]]) {address = 1024 : i32, sym_name = "{{b_mt(_0)?}}"} : memref<128xi32>
// CHECK-DAG: %[[LOCK_B:.*]] = aie.lock(%[[MT]], 1) {init = 1 : i32, sym_name = "{{b_lock(_0)?}}"}
// CHECK-NOT: aie.buffer
// CHECK-NOT: aie.lock
// CHECK: aie.runtime_sequence
// CHECK: aiex.npu.load_pdi {device_ref = @a}
// CHECK: aie.use_lock(%[[LOCK_A]], AcquireGreaterEqual
// CHECK-NEXT: aie.dma_bd(%[[BUF_A]] : memref<256xi32>
// CHECK: aie.use_lock(%[[LOCK_A]], AcquireGreaterEqual
// CHECK-NEXT: aie.dma_bd(%[[BUF_A]] : memref<256xi32>
// CHECK: aiex.npu.load_pdi {device_ref = @b}
// CHECK: aie.use_lock(%[[LOCK_B]], AcquireGreaterEqual
// CHECK-NEXT: aie.dma_bd(%[[BUF_B]] : memref<128xi32>
// CHECK-LABEL: aie.device(npu2) @a

// NAMES-LABEL: aie.device(npu2) {
// NAMES-DAG: sym_name = "b_mt"}
// NAMES-DAG: sym_name = "b_mt_0"}
// NAMES-DAG: sym_name = "b_lock"}
// NAMES-DAG: sym_name = "b_lock_0"}
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
    %buf = aie.buffer(%mt) {address = 0 : i32, sym_name = "b_mt"} : memref<256xi32>
    %lock = aie.lock(%mt, 0) {init = 1 : i32, sym_name = "b_lock"}
    aie.runtime_sequence(%arg0: memref<64xi32>) {
      %t = aiex.dma_configure_task(%mt, MM2S, 0) {
        %c1 = arith.constant 1 : i32
        aie.use_lock(%lock, AcquireGreaterEqual, %c1)
        aie.dma_bd(%buf : memref<256xi32> len = 256)
        aie.end
      }
      aiex.dma_start_task(%t)
    }
  }

  aie.device(npu2) @b {
    %mt = aie.tile(0, 1)
    %buf = aie.buffer(%mt) {address = 1024 : i32, sym_name = "b_mt"} : memref<128xi32>
    %lock = aie.lock(%mt, 1) {init = 1 : i32, sym_name = "b_lock"}
    aie.runtime_sequence(%arg0: memref<64xi32>) {
      %t = aiex.dma_configure_task(%mt, MM2S, 0) {
        %c1 = arith.constant 1 : i32
        aie.use_lock(%lock, AcquireGreaterEqual, %c1)
        aie.dma_bd(%buf : memref<128xi32> len = 128)
        aie.end
      }
      aiex.dma_start_task(%t)
    }
  }
}
