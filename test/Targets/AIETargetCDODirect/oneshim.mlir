//===- oneshim.mlir --------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2024 Advanced Micro Devices, Inc. or its affiliates
//
//===----------------------------------------------------------------------===//

// RUN: aie-translate --aie-generate-cdo %s --cdo-debug=true |& FileCheck %s

// CHECK: (BlockWrite-DMAWriteCmd): Start Address: 0x000000000001D000  Size: 8
// CHECK:     Address: 0x000000000001D000  Data@ {{0x[0-9a-z]+}} is: 0x00000004 
// CHECK:     Address: 0x000000000001D004  Data@ {{0x[0-9a-z]+}} is: 0x00000000 
// CHECK:     Address: 0x000000000001D008  Data@ {{0x[0-9a-z]+}} is: 0x00000000 
// CHECK:     Address: 0x000000000001D00C  Data@ {{0x[0-9a-z]+}} is: 0x00000000 
// CHECK:     Address: 0x000000000001D010  Data@ {{0x[0-9a-z]+}} is: 0x80000000 
// CHECK:     Address: 0x000000000001D014  Data@ {{0x[0-9a-z]+}} is: 0x00000000 
// CHECK:     Address: 0x000000000001D018  Data@ {{0x[0-9a-z]+}} is: 0x00000000 
// CHECK:     Address: 0x000000000001D01C  Data@ {{0x[0-9a-z]+}} is: 0x02000000 

// CHECK: (Write64): Address:  0x000000000001D204 Data:  0x80000000  
// CHECK: (MaskWrite64): Address: 0x000000000001D200  Mask: 0x00000000  Data: 0x00000001 

module {
 aie.device(npu1_4col) {
  %buffer = aie.external_buffer { sym_name = "buf" } : memref<16 x f32>
  %t00 = aie.tile(0, 0)
  aie.shim_dma(%t00)  {
      aie.dma_start(S2MM, 0, ^bd0, ^end)
    ^bd0:
      aie.dma_bd(%buffer : memref<16 x f32>, 0, 4)  {bd_id = 0 : i32}
      aie.next_bd ^end
    ^end:
      aie.end
  }
}
}