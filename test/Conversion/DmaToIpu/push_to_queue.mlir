//===- push_to_queue.mlir ---------------------------------------*- MLIR -*-===//
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2023 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-dma-to-ipu %s | FileCheck %s
// CHECK: aiex.ipu.write32 {address = 119308 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483651 : ui32}
// CHECK: aiex.ipu.write32 {address = 119316 : ui32, column = 2 : i32, row = 0 : i32, value = 196610 : ui32}

module {
  aie.device(ipu) {
    func.func @sequence() {
      %fromMem = aie.shim_dma_allocation(MM2S, 0, 2)
      %toMem = aie.shim_dma_allocation(S2MM, 1, 0)
      aiex.ipu.shimtile_push_queue(%fromMem) {issue_token = true, repeat_count = 0 : i32, bd_id = 3 : i32 }
      aiex.ipu.shimtile_push_queue(%toMem) {issue_token = false, repeat_count = 3 : i32, bd_id = 2 : i32 }
      return
    }
  }
}
