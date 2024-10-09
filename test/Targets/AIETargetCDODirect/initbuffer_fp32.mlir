//===- initbuffer_fp32.mlir -------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2024 Advanced Micro Devices, Inc. or its affiliates
//
//===----------------------------------------------------------------------===//

// RUN: aie-translate --aie-generate-cdo %s --cdo-debug=true |& FileCheck %s

// CHECK: (BlockWrite-DMAWriteCmd): Start Address: 0x0000000000100000  Size: 8
// CHECK:     Address: 0x0000000000100000  Data@ {{0x[0-9a-z]+}} is: 0x40533333 
// CHECK:     Address: 0x0000000000100004  Data@ {{0x[0-9a-z]+}} is: 0x3F800000 
// CHECK:     Address: 0x0000000000100008  Data@ {{0x[0-9a-z]+}} is: 0x40200000 
// CHECK:     Address: 0x000000000010000C  Data@ {{0x[0-9a-z]+}} is: 0x40400000 
// CHECK:     Address: 0x0000000000100010  Data@ {{0x[0-9a-z]+}} is: 0x40966666 
// CHECK:     Address: 0x0000000000100014  Data@ {{0x[0-9a-z]+}} is: 0x40A00000 
// CHECK:     Address: 0x0000000000100018  Data@ {{0x[0-9a-z]+}} is: 0xC0C66666 
// CHECK:     Address: 0x000000000010001C  Data@ {{0x[0-9a-z]+}} is: 0x0ACDC001 

module {
 aie.device(npu1_1col) {
  %tile_0_1 = aie.tile(0, 1)
  %mem_buff_0 = aie.buffer(%tile_0_1) {address = 0 : i32, mem_bank = 0 : i32, sym_name = "mem_buff_0"} : memref<8xf32> = dense<[3.3, 1.0, 2.5, 3.0, 4.7, 5.0, -6.2, 1.9813006e-32]> 
}
}