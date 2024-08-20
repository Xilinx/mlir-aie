//===- dma_to_npu.mlir -----------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2023 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file -aie-dma-to-npu %s | FileCheck %s

// TODO - more
// CHECK: module
// CHECK: aiex.npu.blockwrite
// CHECK: aiex.npu.address_patch
// CHECK-SAME: arg_idx = 0 : i32
// CHECK: aiex.npu.blockwrite
// CHECK: aiex.npu.address_patch
// CHECK-SAME: arg_idx = 1 : i32
module  {
  aie.device(npu1_4col) {
    memref.global "public" @toMem : memref<16xi32>
    memref.global "public" @fromMem : memref<16xi32>
    aiex.runtime_sequence(%arg0: memref<16xi32>, %arg1: memref<16xi32>) {
      aiex.npu.dma_memcpy_nd (0, 0, %arg0[0, 0, 0, 0][1, 1, 16, 16][0, 0, 64, 1]) { metadata = @toMem, id = 1 : i64 } : memref<16xi32>
      aiex.npu.dma_memcpy_nd (0, 1, %arg1[0, 0, 0, 16][1, 1, 16, 16][0, 0, 64, 1]) { metadata = @fromMem, id = 0 : i64 } : memref<16xi32>
    }
    aie.shim_dma_allocation @fromMem (MM2S, 0, 0)
    aie.shim_dma_allocation @toMem (S2MM, 0, 0)
  }
}

// -----

// CHECK: module
// CHECK: aiex.npu.address_patch
// CHECK-SAME: arg_idx = 0 : i32
// CHECK: aiex.npu.write32
// CHECK-SAME: value = 2147483649
// CHECK: aiex.npu.sync 
// CHECK-SAME: channel = 0 : i32
// CHECK-SAME: column = 0 : i32
// CHECK-SAME: column_num = 1 : i32
// CHECK-SAME: direction = 0 : i32
// CHECK-SAME: row = 0 : i32
// CHECK-SAME: row_num = 1 : i32
module  {
  aie.device(npu1_4col) {
    memref.global "public" @toMem : memref<16xi32>
    aiex.runtime_sequence(%arg0: memref<16xi32>, %arg1: memref<16xi32>) {
      aiex.npu.dma_memcpy_nd (0, 0, %arg0[0, 0, 0, 0][1, 1, 16, 16][0, 0, 64, 1]) { issue_token = true, metadata = @toMem, id = 1 : i64 } : memref<16xi32>
      aiex.npu.dma_wait {symbol = @toMem}
    }
    aie.shim_dma_allocation @toMem (S2MM, 0, 0)
  }
}

// -----

// CHECK: module
// CHECK: aiex.npu.blockwrite
// CHECK: aiex.npu.address_patch
// CHECK-SAME: arg_idx = 0 : i32
// CHECK: aiex.npu.write32
// CHECK-SAME: value = 2147483649
// CHECK: aiex.npu.sync 
// CHECK-SAME: channel = 1 : i32
// CHECK-SAME: column = 1 : i32
// CHECK-SAME: column_num = 1 : i32
// CHECK-SAME: direction = 1 : i32
// CHECK-SAME: row = 0 : i32
// CHECK-SAME: row_num = 1 : i32
module  {
  aie.device(npu1_4col) {
    memref.global "public" @toMem : memref<16xi32>
    aiex.runtime_sequence(%arg0: memref<16xi32>, %arg1: memref<16xi32>) {
      aiex.npu.dma_memcpy_nd (0, 0, %arg0[0, 0, 0, 0][1, 1, 16, 16][0, 0, 64, 1]) { issue_token = true, metadata = @toMem, id = 1 : i64 } : memref<16xi32>
      aiex.npu.dma_wait {symbol = @toMem}
    }
    aie.shim_dma_allocation @toMem (MM2S, 1, 1)
  }
}

// -----

// CHECK: runtime_sequence
// CHECK: aiex.npu.write32 {address = 2098576 : ui32, value = 1234 : ui32}
module {
  aie.device(npu1_1col) {
    %tile02 = aie.tile(0,2)
    %data = aie.buffer(%tile02) {address = 1024 : i32, sym_name = "data"} : memref<128xi32>
    aiex.runtime_sequence() {
      aiex.npu.write32 {buffer = @data, address = 100 : ui32, value = 1234 : ui32}
    }
  }
}

// -----

// CHECK: runtime_sequence
// CHECK: aiex.npu.blockwrite(%0) {address = 2098576 : ui32} : memref<4xi32>
module {
  aie.device(npu1_1col) {
    memref.global "private" constant @data : memref<4xi32> = dense<[4, 3, 2, 1]>
    %tile02 = aie.tile(0,2)
    %buf = aie.buffer(%tile02) {address = 1024 : i32, sym_name = "buf"} : memref<128xi32>
    aiex.runtime_sequence() {
      %0 = memref.get_global @data : memref<4xi32>
      aiex.npu.blockwrite(%0) {buffer = @buf, address = 100 : ui32, value = 1234 : ui32} : memref<4xi32>
   }
  }
}

// -----

// CHECK: runtime_sequence
// CHECK: aiex.npu.maskwrite32 {address = 3147552 : ui32, mask = 65535 : ui32, value = 321 : ui32}
module {
  aie.device(npu1_1col) {
    %tile03 = aie.tile(0,3)
    %s = aie.buffer(%tile03) {address = 1024 : i32, sym_name = "stuff"} : memref<128xi32>
    aiex.runtime_sequence() {
      aiex.npu.maskwrite32 {buffer = @stuff, address = 200 : ui32, value = 321 : ui32, mask = 0xffff : ui32}
    }
  }
}

// -----

// Issue packet header from shim dma bd
// CHECK: memref.global "private" constant @blockwrite_data_0 : memref<8xi32>
// CHECK-SAME: = dense<[{{.*}}, {{.*}}, 1074987008, {{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}]>
// CHECK: runtime_sequence
// CHECK: aiex.npu.blockwrite
// CHECK: aiex.npu.address_patch
// CHECK: aiex.npu.write32
module {
  aie.device(npu1_1col) {
    memref.global "public" @toMem : memref<16xi32>
    aiex.runtime_sequence(%arg0: memref<16xi32>) {
      aiex.npu.dma_memcpy_nd (0, 0, %arg0[0, 0, 0, 0][1, 1, 16, 16][0, 0, 64, 1], packet = <pkt_id = 2, pkt_type = 3>) { metadata = @toMem, id = 1 : i64 } : memref<16xi32>
    }
    aie.shim_dma_allocation @toMem (S2MM, 0, 0)
  }
}
