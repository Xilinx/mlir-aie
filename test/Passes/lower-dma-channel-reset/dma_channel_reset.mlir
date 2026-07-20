//===- dma_channel_reset.mlir ----------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-lower-dma-channel-reset %s | FileCheck %s

// Reset S2MM channel 0 on tile (0, 3): the pass pulses the channel control
// register reset bit (bit 1) by writing 0x2 then 0x0. The control register
// local address is 0x1DE00 (122368) for an aie2 core tile S2MM channel 0.
module {
  aie.device(npu2) {
    aie.runtime_sequence() {
      // CHECK: %[[V2:.*]] = arith.constant 2 : i32
      // CHECK: %[[A0:.*]] = arith.constant 122368 : i32
      // CHECK: aiex.npu.write32(%[[A0]], %[[V2]]) {column = 0 : i32, row = 3 : i32} : i32, i32
      // CHECK: %[[V0:.*]] = arith.constant 0 : i32
      // CHECK: %[[A1:.*]] = arith.constant 122368 : i32
      // CHECK: aiex.npu.write32(%[[A1]], %[[V0]]) {column = 0 : i32, row = 3 : i32} : i32, i32
      // CHECK-NOT: aiex.dma_channel_reset
      aiex.dma_channel_reset(0, 3, S2MM, 0)
    }
  }
}
