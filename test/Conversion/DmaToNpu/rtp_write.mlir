//===- rtp_write.mlir ------------------------------------------*- MLIR -*-===//
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2023 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-dma-to-ipu %s | FileCheck %s
// CHECK: aiex.ipu.write32 {address = 1536 : ui32, column = 2 : i32, row = 3 : i32, value = 50 : ui32}
// CHECK: aiex.ipu.write32 {address = 3216 : ui32, column = 0 : i32, row = 2 : i32, value = 99 : ui32}

module {
  aie.device(ipu) {
    %0 = aie.tile(2, 3)
    %1 = aie.buffer(%0) {address = 1536 : i32, sym_name = "rtp"} : memref<16xi32>
    %2 = aie.tile(0, 2)
    %3 = aie.buffer(%2) {address = 3200 : i32, sym_name = "RTP"} : memref<16xi32>
    func.func @sequence() {
      aiex.ipu.rtp_write(2, 3, 0, 50) { buffer_sym_name = "rtp" }
      aiex.ipu.rtp_write(0, 2, 4, 99) { buffer_sym_name = "RTP" }
      return
    }
  }
}
