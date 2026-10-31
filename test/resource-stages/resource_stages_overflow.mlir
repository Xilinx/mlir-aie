//===- resource_stages_overflow.mlir ----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// A design whose buffers do not fit a core tile's memory fails
// aie-resource-stages, as it fails aiecc.

// RUN: not aie-opt --pass-pipeline='builtin.module(aie-resource-stages)' %s 2>&1 | FileCheck %s

// CHECK: error: 'aie.buffer' op could not be placed: buffer "in__cons__buf{{[01]}}" needs 32768 bytes and this tile has no room left for it

module {
  aie.device(npu1_1col) {
    %shim = aie.logical_tile<ShimNOCTile>(?, ?)
    %core = aie.logical_tile<CoreTile>(?, ?)
    aie.objectfifo @in(%shim, {%core}, 2 : i32) : !aie.objectfifo<memref<8192xi32>>
    aie.core(%core) {
      scf.while : () -> () {
        %true = arith.constant true
        scf.condition(%true)
      } do {
        %o = aie.objectfifo.acquire @in (Consume, 1) : memref<8192xi32>
        aie.objectfifo.release @in (Consume, 1)
        scf.yield
      }
      aie.end
    }
    aie.runtime_sequence(%arg0: memref<8192xi32>) {
      %t = aiex.dma_configure_task_for @in {
        aie.dma_bd(%arg0 : memref<8192xi32> offset = 0 len = 8192)
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t)
      aiex.dma_await_task(%t)
    }
  }
}
