//===- bd_chain_on_core/iter_count.mlir ------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform="skip-verify=true" %s | FileCheck %s

// A bounded BD chain is not a MemTile property: `iter_count` on a fifo between
// two compute tiles ends both chains after the final iteration and carries the
// count on the channel's start queue.

module @iter_count_on_core {
  aie.device(npu1) {
    %tile02 = aie.tile(0, 2)
    %tile25 = aie.tile(2, 5)

    aie.objectfifo @of (%tile02, {%tile25}, 2 : i32) {iter_count = 4 : i32} : !aie.objectfifo<memref<16xi32>>
  }
}

// CHECK: %mem_0_2 = aie.mem(%tile_0_2) {
// CHECK:   %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb4, repeat_count = 3)
// CHECK: ^bb1:  // pred: ^bb0
// CHECK:   aie.dma_bd(%of_buff_0 : memref<16xi32> offset = {{.*}} len = {{.*}})
// CHECK:   aie.next_bd ^bb2
// CHECK: ^bb2:  // pred: ^bb1
// CHECK:   aie.dma_bd(%of_buff_1 : memref<16xi32> offset = {{.*}} len = {{.*}})
// CHECK:   aie.next_bd ^bb3
// CHECK: ^bb3:  // pred: ^bb2
// CHECK:   aie.end
// CHECK: ^bb4:  // pred: ^bb0
// CHECK:   aie.end

// CHECK: %mem_2_5 = aie.mem(%tile_2_5) {
// CHECK:   %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb4, repeat_count = 3)
// CHECK: ^bb3:  // pred: ^bb2
// CHECK:   aie.end
