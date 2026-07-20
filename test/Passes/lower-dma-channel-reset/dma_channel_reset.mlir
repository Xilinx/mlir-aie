//===- dma_channel_reset.mlir ----------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-lower-dma-channel-reset %s | FileCheck %s

// Each aiex.dma_channel_reset lowers to a reset pulse on the channel control
// register: write the reset bit (0x2), then clear it (0x0). The control
// register local address comes from AIETargetModel::getDmaControlAddress, so it
// is correct for every tile class and direction.
module {
  aie.device(npu2) {
    aie.runtime_sequence() {
      // Core tile, S2MM channel 0: CTRL local 0x1DE00 = 122368.
      // CHECK: %[[V2A:.*]] = arith.constant 2 : i32
      // CHECK: %[[ADDRA:.*]] = arith.constant 122368 : i32
      // CHECK: aiex.npu.write32(%[[ADDRA]], %[[V2A]]) {column = 0 : i32, row = 3 : i32} : i32, i32
      // CHECK: %[[V0A:.*]] = arith.constant 0 : i32
      // CHECK: %[[ADDRA2:.*]] = arith.constant 122368 : i32
      // CHECK: aiex.npu.write32(%[[ADDRA2]], %[[V0A]]) {column = 0 : i32, row = 3 : i32} : i32, i32
      aiex.dma_channel_reset(0, 3, S2MM, 0)

      // Core tile, MM2S channel 0: CTRL local 0x1DE10 = 122384 (S2MM + 0x10).
      // CHECK: arith.constant 2 : i32
      // CHECK: %[[ADDRB:.*]] = arith.constant 122384 : i32
      // CHECK: aiex.npu.write32(%[[ADDRB]], %{{.*}}) {column = 0 : i32, row = 3 : i32} : i32, i32
      // CHECK: %[[ADDRB2:.*]] = arith.constant 122384 : i32
      // CHECK: aiex.npu.write32(%[[ADDRB2]], %{{.*}}) {column = 0 : i32, row = 3 : i32} : i32, i32
      aiex.dma_channel_reset(0, 3, MM2S, 0)

      // Mem tile, S2MM channel 0: CTRL local 0xA0600 = 656896.
      // CHECK: %[[ADDRC:.*]] = arith.constant 656896 : i32
      // CHECK: aiex.npu.write32(%[[ADDRC]], %{{.*}}) {column = 0 : i32, row = 1 : i32} : i32, i32
      // CHECK: %[[ADDRC2:.*]] = arith.constant 656896 : i32
      // CHECK: aiex.npu.write32(%[[ADDRC2]], %{{.*}}) {column = 0 : i32, row = 1 : i32} : i32, i32
      aiex.dma_channel_reset(0, 1, S2MM, 0)

      // Shim NOC tile, S2MM channel 0: CTRL local 0x1D200 = 119296.
      // CHECK: %[[ADDRD:.*]] = arith.constant 119296 : i32
      // CHECK: aiex.npu.write32(%[[ADDRD]], %{{.*}}) {column = 0 : i32, row = 0 : i32} : i32, i32
      // CHECK: %[[ADDRD2:.*]] = arith.constant 119296 : i32
      // CHECK: aiex.npu.write32(%[[ADDRD2]], %{{.*}}) {column = 0 : i32, row = 0 : i32} : i32, i32
      // CHECK-NOT: aiex.dma_channel_reset
      aiex.dma_channel_reset(0, 0, S2MM, 0)
    }
  }
}
