//===- dma_repeat_count_errors.mlir ----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --verify-diagnostics %s

module {
  aie.device(npu2) {
    %t = aie.tile(0, 2)
    aie.buffer @b(%t) : memref<8xi32>
    aie.mem(%t) {
      // expected-error @+1 {{repeat_count 256 is out of range [0, 255]}}
      aie.dma_start(S2MM, 0, ^bd0, ^end, repeat_count = 256)
    ^bd0:
      %b = aie.get_buffer @b : memref<8xi32>
      aie.dma_bd(%b : memref<8xi32> offset = 0 len = 8) { bd_id = 0 : i32 }
      aie.next_bd ^end
    ^end:
      aie.end
    }
  }
}

// -----

module {
  aie.device(npu2) {
    %t = aie.tile(0, 2)
    aie.buffer @b(%t) : memref<8xi32>
    aie.mem(%t) {
      // maximum repeat_count => ok
      aie.dma_start(S2MM, 0, ^bd0, ^end, repeat_count = 255)
    ^bd0:
      %b = aie.get_buffer @b : memref<8xi32>
      aie.dma_bd(%b : memref<8xi32> offset = 0 len = 8) { bd_id = 0 : i32 }
      aie.next_bd ^end
    ^end:
      aie.end
    }
  }
}

// -----

module {
  aie.device(xcvc1902) {
    %t = aie.tile(3, 3)
    aie.buffer @b(%t) : memref<8xi32>
    aie.mem(%t) {
      // expected-error @+1 {{repeat_count is not supported on this target}}
      aie.dma_start(S2MM, 0, ^bd0, ^end, repeat_count = 1)
    ^bd0:
      %b = aie.get_buffer @b : memref<8xi32>
      aie.dma_bd(%b : memref<8xi32> offset = 0 len = 8) { bd_id = 0 : i32 }
      aie.next_bd ^end
    ^end:
      aie.end
    }
  }
}
