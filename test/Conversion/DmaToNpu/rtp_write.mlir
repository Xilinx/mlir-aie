//===- rtp_write.mlir ------------------------------------------*- MLIR -*-===//
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2023 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-dma-to-npu %s | FileCheck %s
// CHECK-DAG: %[[RA0:.+]] = arith.constant 1536 : i32
// CHECK-DAG: %[[RV0:.+]] = arith.constant 50 : i32
// CHECK: aiex.npu.write32(%[[RA0]], %[[RV0]]) {column = 2 : i32, row = 3 : i32}
// CHECK-DAG: %[[RA1:.+]] = arith.constant 3216 : i32
// CHECK-DAG: %[[RV1:.+]] = arith.constant 99 : i32
// CHECK: aiex.npu.write32(%[[RA1]], %[[RV1]]) {column = 0 : i32, row = 2 : i32}

module {
  aie.device(npu1) {
    %0 = aie.tile(2, 3)
    %1 = aie.buffer(%0) {address = 1536 : i32, sym_name = "rtp"} : memref<16xi32>
    %2 = aie.tile(0, 2)
    %3 = aie.buffer(%2) {address = 3200 : i32, sym_name = "RTP"} : memref<16xi32>
    aie.runtime_sequence() {
      %v0 = arith.constant 50 : i32
      aiex.npu.rtp_write(@rtp, 0, %v0) : i32
      %v1 = arith.constant 99 : i32
      aiex.npu.rtp_write(@RTP, 4, %v1) : i32
    }
  }
}
