//===- shared_send_channel_selected.mlir -----------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --aie-objectFifo-stateful-transform="skip-verify=true" %s | FileCheck %s

// Two fifos pin the same MM2S channel on the same producer tile. A channel
// cycling on one fifo's buffers never reaches the other's, so pinning them
// together asks for the core to push a task per object, and selects it.

// CHECK-LABEL: @shared_send_channel
// CHECK:       aie.dma_start(MM2S, 0, {{.*}}) {core_pushes}
// CHECK:       aie.dma_start(MM2S, 0, {{.*}}) {core_pushes}
module @shared_send_channel {
 aie.device(xcve2302) {
    %tile12 = aie.tile(1, 2)
    %tile32 = aie.tile(3, 2)
    %tile33 = aie.tile(3, 3)

    aie.objectfifo @of1 (%tile12, {%tile32}, 2 : i32) {prod_dma_channel = 0 : i32} : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @of2 (%tile12, {%tile33}, 2 : i32) {prod_dma_channel = 0 : i32} : !aie.objectfifo<memref<16xi32>>
  }
}

// -----

// A channel the tile does not have is still a collision with nothing to share.
// CHECK-LABEL: @out_of_range
// CHECK-NOT:   core_pushes
module @out_of_range {
 aie.device(xcve2302) {
    %tile12 = aie.tile(1, 2)
    %tile32 = aie.tile(3, 2)

    aie.objectfifo @of1 (%tile12, {%tile32}, 2 : i32) {prod_dma_channel = 0 : i32} : !aie.objectfifo<memref<16xi32>>
  }
}
