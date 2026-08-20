//===- dma_repeat_count_errors.mlir ----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --verify-diagnostics %s

// repeat_count is 0-based (the task runs repeat_count + 1 times) and lowers to a
// fixed-width start-queue field, so it is range-checked against the target.

// Exceeds the AIE2 start-queue repeat field (8-bit, so max value 255).
module {
  aie.device(npu2) {
    %t = aie.tile(0, 2)
    %b = aie.buffer(%t) : memref<8xi32>
    aie.mem(%t) {
      // expected-error @+1 {{repeat_count 256 is out of range [0, 255]}}
      aie.dma_start(S2MM, 0, ^bd0, ^end, repeat_count = 256)
    ^bd0:
      aie.dma_bd(%b : memref<8xi32> offset = 0 len = 8) { bd_id = 0 : i32 }
      aie.next_bd ^end
    ^end:
      aie.end
    }
  }
}

// -----

// The boundary value 255 (256 task runs) is accepted.
module {
  aie.device(npu2) {
    %t = aie.tile(0, 2)
    %b = aie.buffer(%t) : memref<8xi32>
    aie.mem(%t) {
      aie.dma_start(S2MM, 0, ^bd0, ^end, repeat_count = 255)
    ^bd0:
      aie.dma_bd(%b : memref<8xi32> offset = 0 len = 8) { bd_id = 0 : i32 }
      aie.next_bd ^end
    ^end:
      aie.end
    }
  }
}

// -----

// AIE1 has no DMA task repeat, so any nonzero repeat_count is rejected.
module {
  aie.device(xcvc1902) {
    %t = aie.tile(3, 3)
    %b = aie.buffer(%t) : memref<8xi32>
    aie.mem(%t) {
      // expected-error @+1 {{repeat_count 1 is out of range [0, 0]}}
      aie.dma_start(S2MM, 0, ^bd0, ^end, repeat_count = 1)
    ^bd0:
      aie.dma_bd(%b : memref<8xi32> offset = 0 len = 8) { bd_id = 0 : i32 }
      aie.next_bd ^end
    ^end:
      aie.end
    }
  }
}
