//===- memtile_col0_initial_value.mlir -------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// An initialized buffer on an npu2 column-0 memtile stays off the first word.
// npu1 has no such defect, so its buffers keep address 0.

// RUN: aie-opt --aie-assign-buffer-addresses %s 2>&1 | FileCheck %s

// CHECK-LABEL: aie.device(npu2) @col0_init
// CHECK:       aie.buffer @w(%{{.*}}) {address = 4 : i32{{.*}}}
// CHECK:       aie.buffer @x(%{{.*}}) {address = 0 : i32{{.*}}}
module @test {
  aie.device(npu2) @col0_init {
    %m0 = aie.tile(0, 1)
    %m1 = aie.tile(1, 1)
    aie.buffer @w(%m0) : memref<4xi32> = dense<[1, 2, 3, 4]>
    aie.buffer @x(%m1) : memref<4xi32> = dense<[1, 2, 3, 4]>
    aie.memtile_dma(%m0) {
      aie.end
    }
    aie.memtile_dma(%m1) {
      aie.end
    }
  }

  // CHECK-LABEL: aie.device(npu2) @col0_uninit
  // CHECK:       aie.buffer @u(%{{.*}}) {address = 0 : i32{{.*}}}
  aie.device(npu2) @col0_uninit {
    %m0 = aie.tile(0, 1)
    aie.buffer @u(%m0) : memref<4xi32>
    aie.memtile_dma(%m0) {
      aie.end
    }
  }

  // CHECK-LABEL: aie.device(npu2) @col0_pinned
  // CHECK:       aie.buffer @p(%{{.*}}) {address = 0 : i32{{.*}}}
  // CHECK:       aie.buffer @q(%{{.*}}) {address = 16 : i32{{.*}}}
  aie.device(npu2) @col0_pinned {
    %m0 = aie.tile(0, 1)
    aie.buffer @p(%m0) {address = 0 : i32} : memref<4xi32> = dense<[1, 2, 3, 4]>
    aie.buffer @q(%m0) : memref<4xi32> = dense<[1, 2, 3, 4]>
    aie.memtile_dma(%m0) {
      aie.end
    }
  }

  // Only the initialized buffer avoids the first word: an uninitialized one
  // may take it, so a layout that fills the memtile exactly still places.
  // CHECK-LABEL: aie.device(npu2) @col0_exact_fit
  // CHECK:       aie.buffer @big(%{{.*}}) {address = 0 : i32{{.*}}}
  // CHECK:       aie.buffer @init(%{{.*}}) {address = 524272 : i32{{.*}}}
  aie.device(npu2) @col0_exact_fit {
    %m0 = aie.tile(0, 1)
    aie.buffer @big(%m0) : memref<131068xi32>
    aie.buffer @init(%m0) : memref<4xi32> = dense<[1, 2, 3, 4]>
    aie.memtile_dma(%m0) {
      aie.end
    }
  }

  // CHECK-LABEL: aie.device(npu1) @npu1_col0
  // CHECK:       aie.buffer @n(%{{.*}}) {address = 0 : i32{{.*}}}
  aie.device(npu1) @npu1_col0 {
    %m0 = aie.tile(0, 1)
    aie.buffer @n(%m0) : memref<4xi32> = dense<[1, 2, 3, 4]>
    aie.memtile_dma(%m0) {
      aie.end
    }
  }

  // CHECK-LABEL: aie.device(xcve2302) @not_npu
  // CHECK:       aie.buffer @v(%{{.*}}) {address = 0 : i32{{.*}}}
  aie.device(xcve2302) @not_npu {
    %m0 = aie.tile(0, 1)
    aie.buffer @v(%m0) : memref<4xi32> = dense<[1, 2, 3, 4]>
    aie.memtile_dma(%m0) {
      aie.end
    }
  }
}
