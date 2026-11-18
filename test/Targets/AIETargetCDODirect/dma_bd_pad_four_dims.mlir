//===- dma_bd_pad_four_dims.mlir ---------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-assign-bd-ids --aie-assign-buffer-addresses %s | aie-translate --aie-generate-cdo --cdo-debug=true 2>&1 | FileCheck %s

// A four-dimensional MemTile walk hands only the pad fields d0..d2 to the BD:
// d0 pads 8 words after, d1 one row before and after.

// CHECK: (BlockWrite-DMAWriteCmd): Start Address: 0x00000000001A0000  Size: 8
// CHECK-NEXT: Address: 0x00000000001A0000 {{.*}} is: 0x00000100
// CHECK-NEXT: Address: 0x00000000001A0004 {{.*}} is: 0x00020000
// CHECK-NEXT: Address: 0x00000000001A0008 {{.*}} is: 0x00100000
// CHECK-NEXT: Address: 0x00000000001A000C {{.*}} is: 0x08040007
// CHECK-NEXT: Address: 0x00000000001A0010 {{.*}} is: 0x0004000F
// CHECK-NEXT: Address: 0x00000000001A0014 {{.*}} is: 0x0090001F

aie.device(npu2) {
  %t = aie.tile(0, 1)
  %buf = aie.buffer(%t) : memref<64xi32>
  %mem = aie.memtile_dma(%t) {
    aie.dma_start("MM2S", 0, ^bd0, ^end)
    ^bd0:
      aie.dma_bd(%buf : memref<64xi32> offset = 0 len = 256 sizes = [2, 2, 2, 8] strides = [32, 16, 8, 1] pad [<const_pad_before = 0, const_pad_after = 0>, <const_pad_before = 0, const_pad_after = 0>, <const_pad_before = 1, const_pad_after = 1>, <const_pad_before = 0, const_pad_after = 8>])
      aie.next_bd ^end
    ^end:
      aie.end
  }
}
