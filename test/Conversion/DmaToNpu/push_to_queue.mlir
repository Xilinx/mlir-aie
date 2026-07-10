//===- push_to_queue.mlir ---------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2023 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-dma-to-npu %s | FileCheck %s
// CHECK-DAG: %[[V0:.*]] = arith.constant -2147483645 : i32
// CHECK-DAG: %[[A0:.*]] = arith.constant 119308 : i32
// CHECK: aiex.npu.write32(%[[A0]], %[[V0]]) : i32, i32
// CHECK-DAG: %[[V1:.*]] = arith.constant 196610 : i32
// CHECK-DAG: %[[A1:.*]] = arith.constant 67228180 : i32
// CHECK: aiex.npu.write32(%[[A1]], %[[V1]]) : i32, i32

module {
  aie.device(npu1) {
    aie.runtime_sequence() {
      %rc0 = arith.constant 0 : i32
      %bd0 = arith.constant 3 : i32
      aiex.npu.push_queue (0, 0, S2MM:1) bd_id %bd0 repeat %rc0 {issue_token = true} : i32, i32
      %rc1 = arith.constant 3 : i32
      %bd1 = arith.constant 2 : i32
      aiex.npu.push_queue (2, 0, MM2S:0) bd_id %bd1 repeat %rc1 {issue_token = false} : i32, i32
    }
  }
}
