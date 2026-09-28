//===- scratchpad_offset_alignment_warning_once.mlir ------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Materialization inlines a runtime sequence at every `aiex.run`, so a DMA op
// with a narrow runtime offset reaches the NPU lowering once per call. The
// alignment warning is about the source op, so it fires once, not per call.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --pass-pipeline='builtin.module(aie-lower-scratchpad-parameters,aie-materialize-runtime-sequences,aie.device(aie-dma-to-npu))' %s -o /dev/null 2>&1 | FileCheck %s

// CHECK: warning: runtime offset parameter on a 16-bit element type
// CHECK-NOT: warning:

module {
  aiex.scratchpad_parameter @off : i32
  aie.device(npu1) @copy {
    %t = aie.tile(0, 0)
    aie.shim_dma_allocation @a (%t, MM2S, 0)
    aie.runtime_sequence @seq(%in: memref<64xi16>) {
      aiex.npu.dma_memcpy_nd(%in[0,0,0,0][1,1,1,64][0,0,0,1])
        {id = 0 : i64, metadata = @a, offset_parameter = @off} : memref<64xi16>
    }
  }
  aie.device(npu1) @main {
    aie.runtime_sequence @sequence(%in: memref<64xi16>) {
      aiex.configure @copy {
        aiex.run @seq(%in) : (memref<64xi16>)
        aiex.run @seq(%in) : (memref<64xi16>)
        aiex.run @seq(%in) : (memref<64xi16>)
      }
    }
  }
}
