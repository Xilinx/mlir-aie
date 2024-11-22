//===- example0.mlir -------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2024 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-translate --aie-generate-cdo %s --cdo-debug=true |& FileCheck %s

// CHECK: (BlockWrite-DMAWriteCmd): Start Address: 0x0000000006300000  Size: 4
// CHECK:     Address: 0x0000000006300000  Data@ {{0x[0-9a-z]+}} is: 0x00000000 
// CHECK:     Address: 0x0000000006300004  Data@ {{0x[0-9a-z]+}} is: 0x00000001 
// CHECK:     Address: 0x0000000006300008  Data@ {{0x[0-9a-z]+}} is: 0x00000002 
// CHECK:     Address: 0x000000000630000C  Data@ {{0x[0-9a-z]+}} is: 0x00000003 

// CHECK: (BlockWrite-DMAWriteCmd): Start Address: 0x0000000008200000  Size: 6
// CHECK:     Address: 0x0000000008200000  Data@ {{0x[0-9a-z]+}} is: 0x00000000 
// CHECK:     Address: 0x0000000008200004  Data@ {{0x[0-9a-z]+}} is: 0x00000001 
// CHECK:     Address: 0x0000000008200008  Data@ {{0x[0-9a-z]+}} is: 0x00000002 
// CHECK:     Address: 0x000000000820000C  Data@ {{0x[0-9a-z]+}} is: 0x00000003 
// CHECK:     Address: 0x0000000008200010  Data@ {{0x[0-9a-z]+}} is: 0x00000004 
// CHECK:     Address: 0x0000000008200014  Data@ {{0x[0-9a-z]+}} is: 0x00000005 

// CHECK: (BlockWrite-DMAWriteCmd): Start Address: 0x0000000008400000  Size: 6
// CHECK:     Address: 0x0000000008400000  Data@ {{0x[0-9a-z]+}} is: 0x00000000 
// CHECK:     Address: 0x0000000008400004  Data@ {{0x[0-9a-z]+}} is: 0x00000001 
// CHECK:     Address: 0x0000000008400008  Data@ {{0x[0-9a-z]+}} is: 0x00000002 
// CHECK:     Address: 0x000000000840000C  Data@ {{0x[0-9a-z]+}} is: 0x00000003 
// CHECK:     Address: 0x0000000008400010  Data@ {{0x[0-9a-z]+}} is: 0x00000004 
// CHECK:     Address: 0x0000000008400014  Data@ {{0x[0-9a-z]+}} is: 0x00000005 

module {
  aie.device(npu1) {
    %t33 = aie.tile(3, 3)
    %t42 = aie.tile(4, 2)
    %t44 = aie.tile(4, 4)

    %buf33 = aie.buffer(%t33) { address = 0 : i32, mem_bank = 0 : i32, sym_name = "buf33" } : memref<2x2xi32> = dense<[[0, 1], [2, 3]]>
    %buf42 = aie.buffer(%t42) { address = 0 : i32, mem_bank = 0 : i32, sym_name = "buf42" } : memref<2x3xi32> = dense<[[0, 1, 2], [3, 4, 5]]>
    %buf44 = aie.buffer(%t44) { address = 0 : i32, mem_bank = 0 : i32, sym_name = "buf44" } : memref<3x2xi32> = dense<[[0, 1], [2, 3], [4, 5]]>
  }
}
