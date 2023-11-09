//===- push_to_queue.mlir ---------------------------------------*- MLIR -*-===//
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2023 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-dma-to-ipu %s | FileCheck %s
// CHECK: AIEX.ipu.write32 {address = 119308 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483651 : ui32}
// CHECK: AIEX.ipu.write32 {address = 119316 : ui32, column = 2 : i32, row = 0 : i32, value = 196610 : ui32}

module {
  AIE.device(ipu) {
    memref.global "public" @toMem : memref<32xi32>
    memref.global "public" @fromMem : memref<32xi32>
    func.func @sequence() {
      AIEX.ipu.shimtile_push_queue {metadata = @toMem, issue_token = true, repeat_count = 0 : i32, bd_id = 3 : i32 }
      AIEX.ipu.shimtile_push_queue {metadata = @fromMem, issue_token = false, repeat_count = 3 : i32, bd_id = 2 : i32 }
      return
    }
    AIE.shimDMAAllocation @fromMem (MM2S, 0, 2)
    AIE.shimDMAAllocation @toMem (S2MM, 1, 0)
  }
}
