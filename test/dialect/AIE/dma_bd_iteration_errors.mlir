//===- dma_bd_iteration_errors.mlir ----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --verify-diagnostics %s

// The grouped #aie.bd_iteration attribute always carries size/stride/current,
// so partial-field combinations are unrepresentable -- only value ranges are
// checked here. Bounds live in the verifyBDIteration helper.

// iteration on an AIE1 target -> arch gate.
module {
  aie.device(xcvc1902) {
    %t = aie.tile(3, 3)
    %b = aie.buffer(%t) : memref<256xi32>
    %m = aie.mem(%t) {
      %s = aie.dma_start(S2MM, 0, ^bd0, ^end)
    ^bd0:
      // expected-error @+1 {{BD iteration is not supported on this target}}
      aie.dma_bd(%b : memref<256xi32> offset = 0 len = 64) { iteration = #aie.bd_iteration<size = 4, stride = 16, current = 0> }
      aie.next_bd ^end
    ^end:
      aie.end
    }
  }
}

// -----

// size < 1.
module {
  aie.device(npu2) {
    %t = aie.tile(0, 1)
    %b = aie.buffer(%t) : memref<256xi32>
    %m = aie.memtile_dma(%t) {
      %s = aie.dma_start(S2MM, 0, ^bd0, ^end)
    ^bd0:
      // expected-error @+1 {{BD iteration size must be in [1, 64]}}
      aie.dma_bd(%b : memref<256xi32> offset = 0 len = 64) { iteration = #aie.bd_iteration<size = 0, stride = 16, current = 0> }
      aie.next_bd ^end
    ^end:
      aie.end
    }
  }
}

// -----

// size > 64.
module {
  aie.device(npu2) {
    %t = aie.tile(0, 1)
    %b = aie.buffer(%t) : memref<256xi32>
    %m = aie.memtile_dma(%t) {
      %s = aie.dma_start(S2MM, 0, ^bd0, ^end)
    ^bd0:
      // expected-error @+1 {{BD iteration size must be in [1, 64]}}
      aie.dma_bd(%b : memref<256xi32> offset = 0 len = 64) { iteration = #aie.bd_iteration<size = 65, stride = 16, current = 0> }
      aie.next_bd ^end
    ^end:
      aie.end
    }
  }
}

// -----

// stride < 1.
module {
  aie.device(npu2) {
    %t = aie.tile(0, 1)
    %b = aie.buffer(%t) : memref<256xi32>
    %m = aie.memtile_dma(%t) {
      %s = aie.dma_start(S2MM, 0, ^bd0, ^end)
    ^bd0:
      // expected-error @+1 {{BD iteration stride must be in [1, 131072] 32-bit words}}
      aie.dma_bd(%b : memref<256xi32> offset = 0 len = 64) { iteration = #aie.bd_iteration<size = 4, stride = 0, current = 0> }
      aie.next_bd ^end
    ^end:
      aie.end
    }
  }
}

// -----

// sub-word element type's stride does not align to 32-bit boundary.
module {
  aie.device(npu2) {
    %t = aie.tile(0, 1)
    %b = aie.buffer(%t) : memref<256xi16>
    %m = aie.memtile_dma(%t) {
      %s = aie.dma_start(S2MM, 0, ^bd0, ^end)
    ^bd0:
      // expected-error @+1 {{BD iteration stride must be aligned to 32-bit words}}
      aie.dma_bd(%b : memref<256xi16> offset = 0 len = 64) { iteration = #aie.bd_iteration<size = 2, stride = 3, current = 0> }
      aie.next_bd ^end
    ^end:
      aie.end
    }
  }
}

// -----

// stride exceeds MemTile step range.
module {
  aie.device(npu2) {
    %t = aie.tile(0, 1)
    %b = aie.buffer(%t) : memref<256xi32>
    %m = aie.memtile_dma(%t) {
      %s = aie.dma_start(S2MM, 0, ^bd0, ^end)
    ^bd0:
      // expected-error @+1 {{BD iteration stride must be in [1, 131072] 32-bit words}}
      aie.dma_bd(%b : memref<256xi32> offset = 0 len = 64) { iteration = #aie.bd_iteration<size = 4, stride = 200000, current = 0> }
      aie.next_bd ^end
    ^end:
      aie.end
    }
  }
}

// -----

// stride exceeds CoreTile's (smaller) step range.
module {
  aie.device(npu2) {
    %t = aie.tile(0, 2)
    %b = aie.buffer(%t) : memref<256xi32>
    %m = aie.mem(%t) {
      %s = aie.dma_start(S2MM, 0, ^bd0, ^end)
    ^bd0:
      // expected-error @+1 {{BD iteration stride must be in [1, 8192] 32-bit words}}
      aie.dma_bd(%b : memref<256xi32> offset = 0 len = 64) { iteration = #aie.bd_iteration<size = 2, stride = 8193, current = 0> }
      aie.next_bd ^end
    ^end:
      aie.end
    }
  }
}

// -----

// current >= size.
module {
  aie.device(npu2) {
    %t = aie.tile(0, 1)
    %b = aie.buffer(%t) : memref<256xi32>
    %m = aie.memtile_dma(%t) {
      %s = aie.dma_start(S2MM, 0, ^bd0, ^end)
    ^bd0:
      // expected-error @+1 {{BD iteration current must be in [0, size)}}
      aie.dma_bd(%b : memref<256xi32> offset = 0 len = 64) { iteration = #aie.bd_iteration<size = 4, stride = 16, current = 4> }
      aie.next_bd ^end
    ^end:
      aie.end
    }
  }
}
