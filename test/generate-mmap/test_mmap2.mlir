//===- test_mmap2.mlir -----------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2021-2022 Xilinx, Inc.
// Copyright (C) 2022-2024 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-translate --aie-generate-mmap %s | FileCheck %s

// CHECK-LABEL: Tile(2, 3)
// CHECK-NOT: _symbol a
// CHECK-LABEL: Tile(3, 2)
// CHECK: _symbol a 0x30000 16
// CHECK-LABEL: Tile(3, 3)
// CHECK: _symbol a 0x38000 16
// CHECK-LABEL: Tile(3, 4)
// CHECK: _symbol a 0x20000 16
// CHECK-LABEL: Tile(4, 3)
// CHECK: _symbol a 0x28000 16

module @test_mmap1 {
 aie.device(xcvc1902) {
  %tsame = aie.tile(3, 3)
  %twest = aie.tile(2, 3) // Different column
  %teast = aie.tile(4, 3) // Different column
  %tsouth = aie.tile(3, 2) // Different row
  %tnorth = aie.tile(3, 4) // Different row

  %bufsame = aie.buffer(%tsame) { sym_name = "a", address = 0x0 : i32 } : memref<4xi32>
 }
}

