//===- bank_aware_prealloc.mlir ---------------------------------*- MLIR -*-===//
//
// Copyright (C) 2022 Xilinx, Inc.
// Copyright (C) 2022-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-prepare-buffers --aie-assign-buffer-addresses %s | FileCheck %s
// CHECK: {{.*}} aie.buffer @core02_buff_internal({{.*}}) {address = {{[0-9]+}} : i32, mem_bank = 2 : i32} : memref<256xi8>
// CHECK: {{.*}} aie.buffer @core02_buff_in({{.*}}) {address = {{[0-9]+}} : i32, mem_bank = {{[0-9]+}} : i32} : memref<256xi8>
// CHECK: {{.*}} aie.buffer @core02_rtp_Buffer({{.*}}) {address = 1024 : i32, mem_bank = {{[0-9]+}} : i32} : memref<256xi8>
// CHECK: {{.*}} aie.buffer @core02_buff_out({{.*}}) {address = {{[0-9]+}} : i32, mem_bank = {{[0-9]+}} : i32} : memref<256xi8>
// CHECK: {{.*}} aie.buffer @core02_buff_out2({{.*}}) {address = 16384 : i32, mem_bank = 1 : i32} : memref<256xi8>
// CHECK: {{.*}} aie.buffer @buffer_4({{.*}}) {address = {{[0-9]+}} : i32, mem_bank = {{[0-9]+}} : memref<500xi32>

module @test {
  aie.device(npu2) {
    %tile_0_2 = aie.tile(0, 2)
    aie.buffer @core02_buff_internal(%tile_0_2) {mem_bank = 2: i32} : memref<256xi8>
    aie.buffer @core02_buff_in(%tile_0_2) : memref<256xi8>
    aie.buffer @core02_rtp_Buffer(%tile_0_2) {address = 1024 : i32} : memref<256xi8>
    aie.buffer @core02_buff_out(%tile_0_2) : memref<256xi8>
    aie.buffer @core02_buff_out2(%tile_0_2) {address=16384 : i32, mem_bank = 1: i32} : memref<256xi8>
    %3 = aie.tile(4, 4)
    aie.buffer @buffer_4(%3) : memref<500xi32>
    aie.core(%tile_0_2) {
      aie.end
    }
    aie.core(%3) {
      aie.end
    }
  }
}
