//===- dma_channel_reset.mlir ----------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --aie-lower-dma-channel-reset %s | FileCheck %s

// Each aiex.dma_channel_reset lowers to a masked reset pulse on the channel
// control register: set the reset bit (value 0x2, mask 0x2), then clear it
// (value 0x0, mask 0x2). Masking to bit 1 preserves the other CTRL fields
// instead of clobbering them. The control register local address comes from
// AIETargetModel::getDmaControlAddress, so it is correct for every accepted
// tile class and direction. Only core and mem tiles have a reset bit, so those
// are the only tiles the op accepts (see dma_channel_reset_invalid.mlir).

// The op covers both AIE2 (npu1) and AIE2P (npu2). The address is target-model
// driven, not hardcoded, so the same op lowers to the same local addresses on
// both: npu2 first, then npu1 below.

module {
  aie.device(npu2) {
    aie.runtime_sequence() {
      // Core tile, S2MM channel 0: CTRL local 0x1DE00 = 122368.
      // CHECK: %[[ADDRA:.*]] = arith.constant 122368 : i32
      // CHECK: %[[SETA:.*]] = arith.constant 2 : i32
      // CHECK: %[[MASKA:.*]] = arith.constant 2 : i32
      // CHECK: aiex.npu.maskwrite32(%[[ADDRA]], %[[SETA]], %[[MASKA]]) {column = 0 : i32, row = 3 : i32} : i32, i32, i32
      // CHECK: %[[ADDRA2:.*]] = arith.constant 122368 : i32
      // CHECK: %[[CLRA:.*]] = arith.constant 0 : i32
      // CHECK: %[[MASKA2:.*]] = arith.constant 2 : i32
      // CHECK: aiex.npu.maskwrite32(%[[ADDRA2]], %[[CLRA]], %[[MASKA2]]) {column = 0 : i32, row = 3 : i32} : i32, i32, i32
      aiex.dma_channel_reset(0, 3, S2MM, 0)

      // Core tile, MM2S channel 0: CTRL local 0x1DE10 = 122384 (S2MM + 0x10).
      // CHECK: %[[ADDRB:.*]] = arith.constant 122384 : i32
      // CHECK: aiex.npu.maskwrite32(%[[ADDRB]], %{{.*}}, %{{.*}}) {column = 0 : i32, row = 3 : i32} : i32, i32, i32
      // CHECK: %[[ADDRB2:.*]] = arith.constant 122384 : i32
      // CHECK: aiex.npu.maskwrite32(%[[ADDRB2]], %{{.*}}, %{{.*}}) {column = 0 : i32, row = 3 : i32} : i32, i32, i32
      aiex.dma_channel_reset(0, 3, MM2S, 0)

      // Mem tile, S2MM channel 0: CTRL local 0xA0600 = 656896.
      // CHECK: %[[ADDRC:.*]] = arith.constant 656896 : i32
      // CHECK: aiex.npu.maskwrite32(%[[ADDRC]], %{{.*}}, %{{.*}}) {column = 0 : i32, row = 1 : i32} : i32, i32, i32
      // CHECK: %[[ADDRC2:.*]] = arith.constant 656896 : i32
      // CHECK: aiex.npu.maskwrite32(%[[ADDRC2]], %{{.*}}, %{{.*}}) {column = 0 : i32, row = 1 : i32} : i32, i32, i32
      // CHECK-NOT: aiex.dma_channel_reset
      aiex.dma_channel_reset(0, 1, S2MM, 0)
    }
  }
}

// -----

// Same op on npu1 (AIE2): the emitted local addresses match npu2 above.
module {
  aie.device(npu1) {
    aie.runtime_sequence() {
      // Core tile, S2MM channel 0: CTRL local 0x1DE00 = 122368.
      // CHECK: %[[NADDRA:.*]] = arith.constant 122368 : i32
      // CHECK: aiex.npu.maskwrite32(%[[NADDRA]], %{{.*}}, %{{.*}}) {column = 0 : i32, row = 3 : i32} : i32, i32, i32
      // CHECK: %[[NADDRA2:.*]] = arith.constant 122368 : i32
      // CHECK: aiex.npu.maskwrite32(%[[NADDRA2]], %{{.*}}, %{{.*}}) {column = 0 : i32, row = 3 : i32} : i32, i32, i32
      aiex.dma_channel_reset(0, 3, S2MM, 0)

      // Mem tile, S2MM channel 0: CTRL local 0xA0600 = 656896.
      // CHECK: %[[NADDRC:.*]] = arith.constant 656896 : i32
      // CHECK: aiex.npu.maskwrite32(%[[NADDRC]], %{{.*}}, %{{.*}}) {column = 0 : i32, row = 1 : i32} : i32, i32, i32
      // CHECK: %[[NADDRC2:.*]] = arith.constant 656896 : i32
      // CHECK: aiex.npu.maskwrite32(%[[NADDRC2]], %{{.*}}, %{{.*}}) {column = 0 : i32, row = 1 : i32} : i32, i32, i32
      // CHECK-NOT: aiex.dma_channel_reset
      aiex.dma_channel_reset(0, 1, S2MM, 0)
    }
  }
}
