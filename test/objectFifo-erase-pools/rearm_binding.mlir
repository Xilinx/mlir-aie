//===- rearm_binding.mlir ---------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// An aie.objectfifo_rearm_binding holds the tiles, channels and locks a
// resident re-arm needs by value, so erasing the pools leaves it whole.

// RUN: aie-opt --aie-objectFifo-stateful-transform="skip-verify=true" --aie-objectfifo-erase-pools %s | FileCheck %s

module {
  aie.device(npu1_1col) {
    %shim = aie.tile(0, 0)
    %tile02 = aie.tile(0, 2)
    aie.objectfifo @of(%tile02, {%shim}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.runtime_sequence(%arg0: memref<16xi32>) {
      aiex.dma_channel_reset_for(@of)
    }
  }
}

// CHECK-NOT: aie.objectfifo.pool
// CHECK:     aiex.dma_channel_reset_for(@of_rearm)
// CHECK:     aie.objectfifo_rearm_binding @of_rearm channels(%{{.*}} : index) locks(%{{.*}}, %{{.*}} : index, index)
