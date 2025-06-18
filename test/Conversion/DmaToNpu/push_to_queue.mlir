//===- push_to_queue.mlir ---------------------------------------*- MLIR -*-===//
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2023 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-dma-to-npu %s | FileCheck %s
// CHECK: aiex.npu.write32 {address = 119308 : ui32, value = 2147483651 : ui32}
// CHECK: aiex.npu.write32 {address = 67228180 : ui32, value = 196610 : ui32}

module {
  aie.device(npu1) {
    aiex.runtime_sequence() {
      aiex.npu.push_queue (0, 0, S2MM:1) {issue_token = true, repeat_count = 0 : i32, bd_id = 3 : i32 }
      aiex.npu.push_queue (2, 0, MM2S:0) {issue_token = false, repeat_count = 3 : i32, bd_id = 2 : i32 }
    }
  }
}
