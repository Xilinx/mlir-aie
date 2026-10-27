//===- test_mmap3.mlir -----------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2021-2022 Xilinx, Inc.
// Copyright (C) 2022-2024 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-translate --aie-generate-mmap %s | FileCheck %s

// CHECK-LABEL: Tile(3, 4)
// CHECK-NEXT: Memory map: name base_address num_bytes
// CHECK-NEXT: _symbol south 0x20000 16
// CHECK-NEXT: _symbol same 0x28000 16
// CHECK-NEXT: _symbol north 0x30000 16
// CHECK-NEXT: _symbol east 0x38000 16

module @test_mmap1 {
 aie.device(xcvc1902) {
  %tsame = aie.tile(3, 4)
  %twest = aie.tile(2, 4) // Different column
  %teast = aie.tile(4, 4) // Different column
  %tsouth = aie.tile(3, 3) // Different row
  %tnorth = aie.tile(3, 5) // Different row

  aie.buffer @same(%tsame) { address = 0x0 : i32 } : memref<4xi32>
  aie.buffer @east(%teast) { address = 0x0 : i32 } : memref<4xi32>
  aie.buffer @west(%twest) { address = 0x0 : i32 } : memref<4xi32>
  aie.buffer @south(%tsouth) { address = 0x0 : i32 } : memref<4xi32>
  aie.buffer @north(%tnorth) { address = 0x0 : i32 } : memref<4xi32>
 }
}
