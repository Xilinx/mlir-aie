//===- push_to_queue.mlir ---------------------------------------*- MLIR -*-===//
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2023 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-dma-to-npu %s | FileCheck %s
// CHECK: %[[WA0:.+]] = arith.constant 119308 : i32
// CHECK: %[[WV0:.+]] = arith.constant -2147483645 : i32
// CHECK: aiex.npu.write32(%[[WA0]], %[[WV0]])
// CHECK: %[[WA1:.+]] = arith.constant 67228180 : i32
// CHECK: %[[WV1:.+]] = arith.constant 196610 : i32
// CHECK: aiex.npu.write32(%[[WA1]], %[[WV1]])

module {
  aie.device(npu1) {
    aie.runtime_sequence() {
      aiex.npu.push_queue (0, 0, S2MM:1) {issue_token = true, repeat_count = 0 : i32, bd_id = 3 : i32 }
      aiex.npu.push_queue (2, 0, MM2S:0) {issue_token = false, repeat_count = 3 : i32, bd_id = 2 : i32 }
    }
  }
}
