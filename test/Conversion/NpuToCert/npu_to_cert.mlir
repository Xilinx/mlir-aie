//===- npu_to_cert.mlir ----------------------------------------*- MLIR -*-===//
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// (c) Copyright 2025 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-npu-to-cert -split-input-file %s | FileCheck %s

// CHECK: aiex.cert.job
// CHECK: aiex.cert.write32(12345, 65244)
aie.device(npu2) {
  aie.runtime_sequence @configure() {
    aiex.npu.write32 {address = 12345 : ui32, value = 0xfedc : ui32}
  }
}

// -----

// CHECK: aiex.cert.uc_dma_chain @chain_0 {
// CHECK:   aiex.cert.uc_dma_bd @blockwrite_data, 4321, 11, false
// CHECK: aiex.cert.job(1) {
// CHECK:   aiex.cert.uc_dma_write_des_sync(@chain_0)
aie.device(npu2) {
  memref.global "private" constant @blockwrite_data : memref<11xi32> = dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]>
  aie.runtime_sequence @configure() {
    %0 = memref.get_global @blockwrite_data : memref<11xi32>
    aiex.npu.blockwrite(%0) {address = 4321 : ui32} : memref<11xi32>
  }
}

// -----

// CHECK: aiex.cert.wait_tcts(64, 7, 1)

aie.device(npu2) {
  aie.runtime_sequence @sync() {
    aiex.npu.sync {channel = 1 : i32, column = 2 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
  }
}

// -----

// CHECK: aiex.cert.apply_offset_57(@blockwrite_data_21, 1, 2)

aie.device(npu2) {
  memref.global "private" constant @blockwrite_data_21 : memref<9xi32> = dense<[1, 2, 3, 4, 5, 6, 7, 8, 9]>
  aie.runtime_sequence @sequence(%arg0: memref<9216xi32>, %arg1: memref<9216xi32>, %arg2: memref<9216xi32>) {
      %21 = memref.get_global @blockwrite_data_21 : memref<9xi32>
      aiex.npu.blockwrite(%21) {address = 6 : ui32} : memref<9xi32>
      aiex.npu.address_patch {addr = 10 : ui32, arg_idx = 2 : i32, arg_plus = 0 : i32}
  }
}

// -----

// CHECK: aiex.cert.uc_dma_chain @[[SYM:.*]] {
// CHECK:   aiex.cert.uc_dma_bd @blockwrite_data0, 4321, 1, true
// CHECK:   aiex.cert.uc_dma_bd @blockwrite_data1, 5432, 2, true
// CHECK:   aiex.cert.uc_dma_bd @blockwrite_data2, 6543, 3, false
// CHECK: }
// CHECK: aiex.cert.job(1) {
// CHECK:   aiex.cert.uc_dma_write_des_sync(@[[SYM]])
// CHECK: }

aie.device(npu2) {
  memref.global "private" constant @blockwrite_data0 : memref<1xi32> = dense<[1]>
  memref.global "private" constant @blockwrite_data1 : memref<2xi32> = dense<[2, 3]>
  memref.global "private" constant @blockwrite_data2 : memref<3xi32> = dense<[4, 5, 6]>
  aiex.cert.uc_dma_chain @chain_0 {
    aiex.cert.uc_dma_bd @blockwrite_data0, 4321, 1, false
  }
  aiex.cert.uc_dma_chain @chain_1 {
    aiex.cert.uc_dma_bd @blockwrite_data1, 5432, 2, false
  }
  aiex.cert.uc_dma_chain @chain_2 {
    aiex.cert.uc_dma_bd @blockwrite_data2, 6543, 3, false
  }
  aiex.cert.job(1) {
    aiex.cert.uc_dma_write_des_sync(@chain_0)
    aiex.cert.uc_dma_write_des_sync(@chain_1)
    aiex.cert.uc_dma_write_des_sync(@chain_2)
  }
}