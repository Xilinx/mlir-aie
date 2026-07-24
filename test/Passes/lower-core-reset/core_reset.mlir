//===- core_reset.mlir -----------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt -split-input-file --aie-lower-core-reset %s | FileCheck %s --implicit-check-not=aiex.npu.write32

// Each aiex.core_reset lowers to a masked reset pulse on the tile's CORE_CONTROL
// register: set the reset bit (value 0x2, mask 0x2), then clear it (value 0x0,
// mask 0x2). Masking to bit 1 preserves the ENABLE field (bit 0) packed in the
// same word instead of clobbering it -- the --implicit-check-not on
// aiex.npu.write32 guards against any stray unmasked write sneaking in. CORE_CONTROL
// is at tile-local offset 0x32000 = 204800 on every core tile, so the address is
// the same for every accepted tile and only the column/row attributes change.
// The target tile is named by an SSA aie.tile value, like aiex.dma_channel_reset.
// Only core tiles have a core to reset, so those are the only tiles the op accepts
// (see core_reset_invalid.mlir).
module {
  aie.device(npu2) {
    %tile_0_3 = aie.tile(0, 3)
    %tile_1_2 = aie.tile(1, 2)
    aie.runtime_sequence() {
      // Core tile (0, 3): CORE_CONTROL local 0x32000 = 204800.
      // CHECK: %[[ADDRA:.*]] = arith.constant 204800 : i32
      // CHECK: %[[SETA:.*]] = arith.constant 2 : i32
      // CHECK: %[[MASKA:.*]] = arith.constant 2 : i32
      // CHECK: aiex.npu.maskwrite32(%[[ADDRA]], %[[SETA]], %[[MASKA]]) {column = 0 : i32, row = 3 : i32} : i32, i32, i32
      // CHECK: %[[ADDRA2:.*]] = arith.constant 204800 : i32
      // CHECK: %[[CLRA:.*]] = arith.constant 0 : i32
      // CHECK: %[[MASKA2:.*]] = arith.constant 2 : i32
      // CHECK: aiex.npu.maskwrite32(%[[ADDRA2]], %[[CLRA]], %[[MASKA2]]) {column = 0 : i32, row = 3 : i32} : i32, i32, i32
      aiex.core_reset(%tile_0_3)

      // A different core tile (1, 2): same CORE_CONTROL address, same reset value
      // and mask, only the column/row attributes change. Value and mask are pinned
      // exactly (not wildcards) so a per-tile value/mask bug cannot slip through.
      // CHECK: %[[ADDRB:.*]] = arith.constant 204800 : i32
      // CHECK: %[[SETB:.*]] = arith.constant 2 : i32
      // CHECK: %[[MASKB:.*]] = arith.constant 2 : i32
      // CHECK: aiex.npu.maskwrite32(%[[ADDRB]], %[[SETB]], %[[MASKB]]) {column = 1 : i32, row = 2 : i32} : i32, i32, i32
      // CHECK: %[[ADDRB2:.*]] = arith.constant 204800 : i32
      // CHECK: %[[CLRB:.*]] = arith.constant 0 : i32
      // CHECK: %[[MASKB2:.*]] = arith.constant 2 : i32
      // CHECK: aiex.npu.maskwrite32(%[[ADDRB2]], %[[CLRB]], %[[MASKB2]]) {column = 1 : i32, row = 2 : i32} : i32, i32, i32
      // CHECK-NOT: aiex.core_reset
      aiex.core_reset(%tile_1_2)
    }
  }
}

// -----

// CORE_CONTROL is at the same 0x32000 offset on npu1 (Phoenix), so the lowering
// is device-independent across the AIE2 family. This pins the npu1 address so a
// future change that made the address device-conditional and got npu1 wrong would
// fail here.
module {
  aie.device(npu1) {
    %tile_0_3 = aie.tile(0, 3)
    aie.runtime_sequence() {
      // CHECK: %[[ADDRC:.*]] = arith.constant 204800 : i32
      // CHECK: %[[SETC:.*]] = arith.constant 2 : i32
      // CHECK: %[[MASKC:.*]] = arith.constant 2 : i32
      // CHECK: aiex.npu.maskwrite32(%[[ADDRC]], %[[SETC]], %[[MASKC]]) {column = 0 : i32, row = 3 : i32} : i32, i32, i32
      aiex.core_reset(%tile_0_3)
    }
  }
}
