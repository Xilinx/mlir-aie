// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt --aie-materialize-runtime-sequences %s | FileCheck %s

// Cloning a sequence into a device with colliding declarations must rename both
// the buffer accessor's symbol and the lock symbol, not just SSA operands.
module {
  // CHECK-LABEL: aie.device(npu2) @caller
  aie.device(npu2) @caller {
    %tile = aie.tile(0, 2)
    // CHECK: aie.buffer @buffer(
    aie.buffer @buffer(%tile) {address = 0 : i32} : memref<16xi32>
    // CHECK: aie.lock @lock(
    aie.lock @lock(%tile, 0)
    // CHECK-DAG: aie.buffer @[[BUFFER:buffer_[0-9]+]]({{.*}}) {address = 1024 : i32}
    // CHECK-DAG: aie.lock @[[LOCK:lock_[0-9]+]]({{.*}}, 1)
    // CHECK-LABEL: aie.runtime_sequence @main_seq
    aie.runtime_sequence @main_seq(%arg0: memref<1xi32>) {
      // CHECK: aiex.npu.load_pdi {device_ref = @configuration}
      // CHECK: aiex.dma_configure_task(
      // CHECK: %[[ACCESS:.*]] = aie.get_buffer @[[BUFFER]] : memref<16xi32>
      // CHECK: aie.use_lock(@[[LOCK]], AcquireGreaterEqual, %{{.*}})
      // CHECK: aie.dma_bd(%[[ACCESS]] : memref<16xi32>
      // CHECK: aie.use_lock(@[[LOCK]], Release, %{{.*}})
      aiex.configure @configuration {
        aiex.run @sequence(%arg0) : (memref<1xi32>)
      }
    }
  }

  aie.device(npu2) @configuration {
    %tile = aie.tile(0, 2)
    aie.buffer @buffer(%tile) {address = 1024 : i32} : memref<16xi32>
    aie.lock @lock(%tile, 1)
    aie.runtime_sequence @sequence(%arg0: memref<1xi32>) {
      %task = aiex.dma_configure_task(%tile, MM2S, 0) {
        %buffer = aie.get_buffer @buffer : memref<16xi32>
        %value = arith.constant 1 : i32
        aie.use_lock(@lock, AcquireGreaterEqual, %value)
        aie.dma_bd(%buffer : memref<16xi32> offset = 0 len = 16)
        aie.use_lock(@lock, Release, %value)
        aie.end
      }
      aiex.dma_start_task(%task)
    }
  }
}
