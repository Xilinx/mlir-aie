// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: aie-opt --aie-lower-dynamic-bd-pool --verify-diagnostics %s | FileCheck %s

// AIE1 has no queued-task model (depth zero). Accumulating pushes in the loop
// simulation would grow its state forever, since no overflow caps its size.
// CHECK-LABEL: @no_queue_model
// CHECK: scf.for
// CHECK-NOT: aiex.npu.maskpoll
// CHECK: aiex.dma_start_task
// CHECK-NOT: aiex.npu.maskpoll
aie.device(xcvc1902) {
  %tile = aie.tile(2, 0)
  aie.runtime_sequence @no_queue_model(%arg0: memref<256xi32>, %n: index) {
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    scf.for %i = %zero to %n step %one {
      %task = aiex.dma_configure_task(%tile, MM2S, 0) {
        aie.dma_bd(%arg0 : memref<256xi32> offset = 0 len = 256)
        aie.end
      }
      aiex.dma_start_task(%task)
    }
  }
}
