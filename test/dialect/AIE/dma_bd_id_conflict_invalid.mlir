//===- dma_bd_id_conflict_invalid.mlir -------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --verify-diagnostics %s

// A dma_bd cannot carry both the static bd_id attribute and the runtime
// bd_id_val operand.
module {
  aie.device(npu2) {
    %t = aie.tile(0, 2)
    %b = aie.buffer(%t) : memref<4xi32>
    aie.mem(%t) {
        aie.dma_start(MM2S, 0, ^bd0, ^end)
      ^bd0:
        %id = arith.constant 3 : i32
        // expected-error@+1 {{bd_id and bd_id_val are mutually exclusive}}
        aie.dma_bd(%b : memref<4xi32> offset = 0 len = 4) bd_id_val %id : i32 { bd_id = 3 : i32 }
        aie.next_bd ^end
      ^end:
        aie.end
    }
  }
}
