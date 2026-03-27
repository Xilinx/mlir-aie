//===- address_aware_alloc.mlir ---------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-assign-buffer-addresses="alloc-scheme=bank-aware" %s | FileCheck %s
// CHECK: {{.*}} aie.buffer({{.*}}) {address = {{[0-9]+}} : i32, mem_bank = {{[0-9]+}} : i32, sym_name = "core02_buff_in"} : memref<256xi8>
// CHECK: {{.*}} aie.buffer({{.*}}) {address = 1024 : i32, mem_bank = {{[0-9]+}} : i32, sym_name = "core02_rtp_Buffer"} : memref<256xi8>
// CHECK: {{.*}} aie.buffer({{.*}}) {address = {{[0-9]+}} : i32, mem_bank = {{[0-9]+}} : i32, sym_name = "core02_buff_out"} : memref<256xi8>
// CHECK: {{.*}} aie.buffer({{.*}}) {address = {{[0-9]+}} : i32, mem_bank = {{[0-9]+}} : i32, sym_name = "_anonymous0"} : memref<500xi32>

module @test {
  aie.device(npu2) {
    %tile_0_2 = aie.tile(0, 2)
    %core02_buff_in = aie.buffer(%tile_0_2) {sym_name = "core02_buff_in"} : memref<256xi8> 
    %core02_rtp_Buffer = aie.buffer(%tile_0_2) {address = 1024 : i32, sym_name = "core02_rtp_Buffer"} : memref<256xi8> 
    %core02_buff_out = aie.buffer(%tile_0_2) {sym_name = "core02_buff_out"} : memref<256xi8>     
    %3 = aie.tile(4, 4)
    %4 = aie.buffer(%3) : memref<500xi32>
    aie.core(%tile_0_2) {
      aie.end
    }
    aie.core(%3) {
      aie.end
    }
  }
}
