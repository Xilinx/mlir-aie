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
  aie.device(npu1) {
    aie.runtime_sequence(%arg0: memref<16xi32>, %arg1: memref<16xi32>) {
      aiex.npu.dma_memcpy_nd (%arg0[0, 0, 0, 0][1, 1, 16, 16][0, 0, 64, 1]) { metadata = @toMem, id = 1 : i64 } : memref<16xi32>
      aiex.npu.dma_memcpy_nd (%arg1[0, 0, 0, 16][1, 1, 16, 16][0, 0, 64, 1]) { metadata = @fromMem, id = 0 : i64 } : memref<16xi32>
    }
    %tile_0_0 = aie.tile(0, 0)
    aie.shim_dma_allocation @fromMem (%tile_0_0, MM2S, 0)
    aie.shim_dma_allocation @toMem (%tile_0_0, S2MM, 0)
  }
}

// -----

// CHECK: module
// CHECK: aiex.npu.address_patch
// CHECK-SAME: arg_idx = 0 : i32
// CHECK: %{{.+}} = arith.constant -2147483647 : i32
// CHECK: aiex.npu.write32
// sync(column, row, direction, channel, column_num, row_num) = (0, 0, 0, 0, 1, 1)
// CHECK: %[[COL:.+]] = arith.constant 0 : i32
// CHECK: %[[ROW:.+]] = arith.constant 0 : i32
// CHECK: %[[DIR:.+]] = arith.constant 0 : i32
// CHECK: %[[CHAN:.+]] = arith.constant 0 : i32
// CHECK: %[[CNUM:.+]] = arith.constant 1 : i32
// CHECK: %[[RNUM:.+]] = arith.constant 1 : i32
// CHECK: aiex.npu.sync(%[[COL]], %[[ROW]], %[[DIR]], %[[CHAN]], %[[CNUM]], %[[RNUM]])
module  {
  aie.device(npu1) {
    aie.runtime_sequence(%arg0: memref<16xi32>, %arg1: memref<16xi32>) {
      aiex.npu.dma_memcpy_nd (%arg0[0, 0, 0, 0][1, 1, 16, 16][0, 0, 64, 1]) { issue_token = true, metadata = @toMem, id = 1 : i64 } : memref<16xi32>
      aiex.npu.dma_wait {symbol = @toMem}
    }
    %tile_0_0 = aie.tile(0, 0)
    aie.shim_dma_allocation @toMem (%tile_0_0, S2MM, 0)
  }
}

// -----

// CHECK: module
// CHECK: aiex.npu.blockwrite
// CHECK: aiex.npu.address_patch
// CHECK-SAME: arg_idx = 0 : i32
// CHECK: %{{.+}} = arith.constant -2147483647 : i32
// CHECK: aiex.npu.write32
// sync(column, row, direction, channel, column_num, row_num) = (1, 0, 1, 1, 1, 1)
// CHECK: %[[COL:.+]] = arith.constant 1 : i32
// CHECK: %[[ROW:.+]] = arith.constant 0 : i32
// CHECK: %[[DIR:.+]] = arith.constant 1 : i32
// CHECK: %[[CHAN:.+]] = arith.constant 1 : i32
// CHECK: %[[CNUM:.+]] = arith.constant 1 : i32
// CHECK: %[[RNUM:.+]] = arith.constant 1 : i32
// CHECK: aiex.npu.sync(%[[COL]], %[[ROW]], %[[DIR]], %[[CHAN]], %[[CNUM]], %[[RNUM]])
module  {
  aie.device(npu1) {
    aie.runtime_sequence(%arg0: memref<16xi32>, %arg1: memref<16xi32>) {
      aiex.npu.dma_memcpy_nd (%arg0[0, 0, 0, 0][1, 1, 16, 16][0, 0, 64, 1]) { issue_token = true, metadata = @toMem, id = 1 : i64 } : memref<16xi32>
      aiex.npu.dma_wait {symbol = @toMem}
    }
    %tile_1_0 = aie.tile(1, 0)
    aie.shim_dma_allocation @toMem (%tile_1_0, MM2S, 1)
  }
}

// -----

// CHECK: runtime_sequence
// CHECK-DAG: %[[WA0:.+]] = arith.constant 2098576 : i32
// CHECK-DAG: %[[WV0:.+]] = arith.constant 1234 : i32
// CHECK: aiex.npu.write32(%[[WA0]], %[[WV0]])
module {
  aie.device(npu1_1col) {
    %tile02 = aie.tile(0,2)
    %data = aie.buffer(%tile02) {address = 1024 : i32, sym_name = "data"} : memref<128xi32>
    aie.runtime_sequence() {
      %addr = arith.constant 100 : i32
      %val = arith.constant 1234 : i32
      aiex.npu.write32(%addr, %val) {buffer = @data} : i32, i32
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
    aie.runtime_sequence() {
      %0 = memref.get_global @data : memref<4xi32>
      aiex.npu.blockwrite(%0) {buffer = @buf, address = 100 : ui32, value = 1234 : ui32} : memref<4xi32>
   }
  }
}

// -----

// CHECK: runtime_sequence
// CHECK-DAG: %[[MWA:.+]] = arith.constant 3147552 : i32
// CHECK-DAG: %[[MWV:.+]] = arith.constant 321 : i32
// CHECK-DAG: %[[MWM:.+]] = arith.constant 65535 : i32
// CHECK: aiex.npu.maskwrite32(%[[MWA]], %[[MWV]], %[[MWM]])
module {
  aie.device(npu1_1col) {
    %tile03 = aie.tile(0,3)
    %s = aie.buffer(%tile03) {address = 1024 : i32, sym_name = "stuff"} : memref<128xi32>
    aie.runtime_sequence() {
      %addr = arith.constant 200 : i32
      %val = arith.constant 321 : i32
      %mask = arith.constant 0xffff : i32
      aiex.npu.maskwrite32(%addr, %val, %mask) {buffer = @stuff} : i32, i32, i32
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
    aie.runtime_sequence(%arg0: memref<16xi32>) {
      aiex.npu.dma_memcpy_nd (%arg0[0, 0, 0, 0][1, 1, 16, 16][0, 0, 64, 1], packet = <pkt_id = 2, pkt_type = 3>) { metadata = @toMem, id = 1 : i64 } : memref<16xi32>
    }
    %tile_0_0 = aie.tile(0, 0)
    aie.shim_dma_allocation @toMem (%tile_0_0, S2MM, 0)
  }
}
