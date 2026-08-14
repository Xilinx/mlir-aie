//===- npu_read_reg_instgen.mlir ---------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-translate --aie-npu-to-binary -aie-output-binary=false %s | FileCheck %s

// aiex.npu.read_reg lowers to one TXN_OPC_READ_REGS (0x82) custom-op entry per
// op: opcode word, 16-byte size word, count=1, then the resolved address
// (tile(2,3) offset 0x100 resolves to 0x04300100, the same tile-base|offset
// formula NpuWrite32Op::getAbsoluteAddress uses).

// CHECK: 06040100
// CHECK: 00000108
// CHECK: 00000001
// CHECK: 00000020
module {
  aie.device(npu2) {
    %tile = aie.tile(2, 3)
    aie.runtime_sequence() {
      // CHECK: 00000082
      // CHECK: 00000010
      // CHECK: 00000001
      // CHECK: 04300100
      aiex.npu.read_reg(%tile, 0x100)
    }
  }
}
