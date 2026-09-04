//===- read_reg.mlir --------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s | FileCheck %s

// aiex.npu.read_reg names its target with an SSA aie.tile value, like
// aiex.dma_channel_reset / aiex.core_reset. Unlike those ops it has no
// tile-type restriction -- core, mem, and shim tiles all have readable
// registers -- so all three accept here.

// CHECK-LABEL: aie.device(npu2)
// CHECK: aiex.npu.read_reg(%[[CORE:.*]], 256)
// CHECK: aiex.npu.read_reg(%[[MEM:.*]], 512)
// CHECK: aiex.npu.read_reg(%[[SHIM:.*]], 0)
module {
  aie.device(npu2) {
    %core_tile = aie.tile(0, 2)
    %mem_tile = aie.tile(0, 1)
    %shim_tile = aie.tile(0, 0)
    aie.runtime_sequence() {
      aiex.npu.read_reg(%core_tile, 0x100)
      aiex.npu.read_reg(%mem_tile, 0x200)
      aiex.npu.read_reg(%shim_tile, 0)
    }
  }
}
