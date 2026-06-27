//===- dma_bd_iteration.mlir ------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc. or its affiliates
//
//===----------------------------------------------------------------------===//

// Verifies that a static aie.dma_bd with explicit iter_size/iter_stride lowers
// through AIERT (XAie_DmaSetBdIteration) into the BD iteration register.
//
// The iteration word lives at BD-base + 0x18. The driver packs LOGICAL values:
//   iter_size  = 4  -> wrap   = 4 - 1 = 3
//   iter_stride= 64 i32 elems = 64 32b words -> stepsize = 64 - 1 = 63 = 0x3F
// so the iteration field is 0x0006003F (current = 0).

// RUN: aie-translate --aie-generate-cdo %s --cdo-debug=true | FileCheck %s

// CHECK: (BlockWrite-DMAWriteCmd): Start Address: 0x00000000041A0000  Size: 8
// CHECK:     Address: 0x00000000041A0018  Data@ {{0x[0-9a-z]+}} is: 0x0006003F

module {
  aie.device(npu1) {
    %t = aie.tile(2, 1)
    %b = aie.buffer(%t) { sym_name = "b", address = 4096 : i32 } : memref<256xi32>
    %m = aie.memtile_dma(%t) {
      %s = aie.dma_start("MM2S", 0, ^bd0, ^end)
    ^bd0:
      aie.dma_bd(%b : memref<256xi32>, 0, 64, [<size = 8, stride = 8>, <size = 8, stride = 1>]) {iter_size = 4 : i32, iter_stride = 64 : i32, bd_id = 0 : i32}
      aie.next_bd ^end
    ^end:
      aie.end
    }
  }
}
