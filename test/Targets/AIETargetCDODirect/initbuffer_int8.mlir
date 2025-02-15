//===- initbuffer_int8.mlir -------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2024 Advanced Micro Devices, Inc. or its affiliates
//
//===----------------------------------------------------------------------===//

// RUN: aie-translate --aie-generate-cdo %s --cdo-debug=true |& FileCheck %s

// CHECK: (BlockWrite-DMAWriteCmd): Start Address: 0x0000000000100000  Size: 2
// CHECK:     Address: 0x0000000000100000  Data@ {{0x[0-9a-z]+}} is: 0xFD020100 
// CHECK:     Address: 0x0000000000100004  Data@ {{0x[0-9a-z]+}} is: 0x07060504 

module {
 aie.device(npu1_1col) {
  %tile_0_1 = aie.tile(0, 1)
  %mem_buff_0 = aie.buffer(%tile_0_1) {address = 0 : i32, mem_bank = 0 : i32, sym_name = "mem_buff_0"} : memref<8xi8> = dense<[0, 1, 2, -3, 4, 5, 6, 7]> 
}
}