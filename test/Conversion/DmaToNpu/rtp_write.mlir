//===- rtp_write.mlir ------------------------------------------*- MLIR -*-===//
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2023-2025, Advanced Micro Devices, Inc. All rights reserved.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-dma-to-npu %s | FileCheck %s
// CHECK: aiex.npu.write32 {address = 1536 : ui32, column = 2 : i32, row = 3 : i32, value = 50 : ui32}
// CHECK: aiex.npu.write32 {address = 3216 : ui32, column = 0 : i32, row = 2 : i32, value = 99 : ui32}

module {
  aie.device(npu1) {
    %0 = aie.tile(2, 3)
    %1 = aie.buffer(%0) {address = 1536 : i32, sym_name = "rtp"} : memref<16xi32>
    %2 = aie.tile(0, 2)
    %3 = aie.buffer(%2) {address = 3200 : i32, sym_name = "RTP"} : memref<16xi32>
    aie.runtime_sequence() {
      aiex.npu.rtp_write(@rtp, 0, 50)
      aiex.npu.rtp_write(@RTP, 4, 99)
    }
  }
}
