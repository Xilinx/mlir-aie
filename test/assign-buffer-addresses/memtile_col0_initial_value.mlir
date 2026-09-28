//===- memtile_col0_initial_value.mlir -------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// An initialized buffer on an NPU column-0 memtile stays off the first word.

// RUN: aie-opt --aie-assign-buffer-addresses %s 2>&1 | FileCheck %s

// CHECK-LABEL: aie.device(npu2) @col0_init
// CHECK:       %w = aie.buffer(%{{.*}}) {address = 4 : i32, {{.*}}sym_name = "w"}
// CHECK:       %x = aie.buffer(%{{.*}}) {address = 0 : i32, {{.*}}sym_name = "x"}
module @test {
  aie.device(npu2) @col0_init {
    %m0 = aie.tile(0, 1)
    %m1 = aie.tile(1, 1)
    %w = aie.buffer(%m0) {sym_name = "w"} : memref<4xi32> = dense<[1, 2, 3, 4]>
    %x = aie.buffer(%m1) {sym_name = "x"} : memref<4xi32> = dense<[1, 2, 3, 4]>
    aie.memtile_dma(%m0) {
      aie.end
    }
    aie.memtile_dma(%m1) {
      aie.end
    }
  }

  // CHECK-LABEL: aie.device(npu2) @col0_uninit
  // CHECK:       %u = aie.buffer(%{{.*}}) {address = 0 : i32, {{.*}}sym_name = "u"}
  aie.device(npu2) @col0_uninit {
    %m0 = aie.tile(0, 1)
    %u = aie.buffer(%m0) {sym_name = "u"} : memref<4xi32>
    aie.memtile_dma(%m0) {
      aie.end
    }
  }

  // CHECK-LABEL: aie.device(npu2) @col0_pinned
  // CHECK:       %p = aie.buffer(%{{.*}}) {address = 0 : i32, {{.*}}sym_name = "p"}
  // CHECK:       %q = aie.buffer(%{{.*}}) {address = 16 : i32, {{.*}}sym_name = "q"}
  aie.device(npu2) @col0_pinned {
    %m0 = aie.tile(0, 1)
    %p = aie.buffer(%m0) {address = 0 : i32, sym_name = "p"} : memref<4xi32> = dense<[1, 2, 3, 4]>
    %q = aie.buffer(%m0) {sym_name = "q"} : memref<4xi32> = dense<[1, 2, 3, 4]>
    aie.memtile_dma(%m0) {
      aie.end
    }
  }

  // CHECK-LABEL: aie.device(xcve2302) @not_npu
  // CHECK:       %v = aie.buffer(%{{.*}}) {address = 0 : i32, {{.*}}sym_name = "v"}
  aie.device(xcve2302) @not_npu {
    %m0 = aie.tile(0, 1)
    %v = aie.buffer(%m0) {sym_name = "v"} : memref<4xi32> = dense<[1, 2, 3, 4]>
    aie.memtile_dma(%m0) {
      aie.end
    }
  }
}
