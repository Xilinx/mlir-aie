//===- dma_bd_iteration_errors.mlir ----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --verify-diagnostics %s

// iteration on an AIE1 target -> arch gate.
module {
  aie.device(xcvc1902) {
    %t = aie.tile(3, 3)
    %b = aie.buffer(%t) : memref<256xi32>
    %m = aie.mem(%t) {
      %s = aie.dma_start(S2MM, 0, ^bd0, ^end)
    ^bd0:
      // expected-error @+1 {{BD iteration is not supported on this target}}
      aie.dma_bd(%b : memref<256xi32> offset = 0 len = 64) { iteration_size = 4 : i32, iteration_stride = 16 : i32 }
      aie.next_bd ^end
    ^end:
      aie.end
    }
  }
}

// -----

// iteration_stride set without iteration_size.
module {
  aie.device(npu2) {
    %t = aie.tile(0, 1)
    %b = aie.buffer(%t) : memref<256xi32>
    %m = aie.memtile_dma(%t) {
      %s = aie.dma_start(S2MM, 0, ^bd0, ^end)
    ^bd0:
      // expected-error @+1 {{iteration_stride requires iteration_size to be set}}
      aie.dma_bd(%b : memref<256xi32> offset = 0 len = 64) { iteration_stride = 16 : i32 }
      aie.next_bd ^end
    ^end:
      aie.end
    }
  }
}

// -----

// iteration_size < 1.
module {
  aie.device(npu2) {
    %t = aie.tile(0, 1)
    %b = aie.buffer(%t) : memref<256xi32>
    %m = aie.memtile_dma(%t) {
      %s = aie.dma_start(S2MM, 0, ^bd0, ^end)
    ^bd0:
      // expected-error @+1 {{iteration_size must be in [1, 64]}}
      aie.dma_bd(%b : memref<256xi32> offset = 0 len = 64) { iteration_size = 0 : i32, iteration_stride = 16 : i32 }
      aie.next_bd ^end
    ^end:
      aie.end
    }
  }
}

// -----

// iteration_size > 64.
module {
  aie.device(npu2) {
    %t = aie.tile(0, 1)
    %b = aie.buffer(%t) : memref<256xi32>
    %m = aie.memtile_dma(%t) {
      %s = aie.dma_start(S2MM, 0, ^bd0, ^end)
    ^bd0:
      // expected-error @+1 {{iteration_size must be in [1, 64]}}
      aie.dma_bd(%b : memref<256xi32> offset = 0 len = 64) { iteration_size = 65 : i32, iteration_stride = 16 : i32 }
      aie.next_bd ^end
    ^end:
      aie.end
    }
  }
}

// -----

// iteration_stride < 1.
module {
  aie.device(npu2) {
    %t = aie.tile(0, 1)
    %b = aie.buffer(%t) : memref<256xi32>
    %m = aie.memtile_dma(%t) {
      %s = aie.dma_start(S2MM, 0, ^bd0, ^end)
    ^bd0:
      // expected-error @+1 {{iteration_stride must be in [1, 131072] 32-bit words}}
      aie.dma_bd(%b : memref<256xi32> offset = 0 len = 64) { iteration_size = 4 : i32, iteration_stride = -5 : i32 }
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
      // expected-error @+1 {{iteration_stride must result in a stride aligned to 32-bit words}}
      aie.dma_bd(%b : memref<256xi16> offset = 0 len = 64) { iteration_size = 2 : i32, iteration_stride = 3 : i32 }
      aie.next_bd ^end
    ^end:
      aie.end
    }
  }
}

// -----

// iteration_stride exceeds step range.
module {
  aie.device(npu2) {
    %t = aie.tile(0, 1)
    %b = aie.buffer(%t) : memref<256xi32>
    %m = aie.memtile_dma(%t) {
      %s = aie.dma_start(S2MM, 0, ^bd0, ^end)
    ^bd0:
      // expected-error @+1 {{iteration_stride must be in [1, 131072] 32-bit words}}
      aie.dma_bd(%b : memref<256xi32> offset = 0 len = 64) { iteration_size = 4 : i32, iteration_stride = 200000 : i32 }
      aie.next_bd ^end
    ^end:
      aie.end
    }
  }
}

// -----

// iteration_stride exceeds CoreTile's step range.
module {
  aie.device(npu2) {
    %t = aie.tile(0, 2)
    %b = aie.buffer(%t) : memref<256xi32>
    %m = aie.mem(%t) {
      %s = aie.dma_start(S2MM, 0, ^bd0, ^end)
    ^bd0:
      // expected-error @+1 {{iteration_stride must be in [1, 8192] 32-bit words}}
      aie.dma_bd(%b : memref<256xi32> offset = 0 len = 64) { iteration_size = 2 : i32, iteration_stride = 8193 : i32 }
      aie.next_bd ^end
    ^end:
      aie.end
    }
  }
}

// -----

// iteration_size > 1 requires iteration_stride to be set.
module {
  aie.device(npu2) {
    %t = aie.tile(0, 1)
    %b = aie.buffer(%t) : memref<256xi32>
    %m = aie.memtile_dma(%t) {
      %s = aie.dma_start(S2MM, 0, ^bd0, ^end)
    ^bd0:
      // expected-error @+1 {{iteration_stride must be set when iteration_size > 1}}
      aie.dma_bd(%b : memref<256xi32> offset = 0 len = 64) { iteration_size = 4 : i32 }
      aie.next_bd ^end
    ^end:
      aie.end
    }
  }
}

// -----

// iteration_current set without iteration_size.
module {
  aie.device(npu2) {
    %t = aie.tile(0, 1)
    %b = aie.buffer(%t) : memref<256xi32>
    %m = aie.memtile_dma(%t) {
      %s = aie.dma_start(S2MM, 0, ^bd0, ^end)
    ^bd0:
      // expected-error @+1 {{iteration_current requires iteration_size to be set}}
      aie.dma_bd(%b : memref<256xi32> offset = 0 len = 64) { iteration_current = 2 : i32 }
      aie.next_bd ^end
    ^end:
      aie.end
    }
  }
}

// -----

// iteration_current < 0.
module {
  aie.device(npu2) {
    %t = aie.tile(0, 1)
    %b = aie.buffer(%t) : memref<256xi32>
    %m = aie.memtile_dma(%t) {
      %s = aie.dma_start(S2MM, 0, ^bd0, ^end)
    ^bd0:
      // expected-error @+1 {{iteration_current must be in [0, iteration_size)}}
      aie.dma_bd(%b : memref<256xi32> offset = 0 len = 64) { iteration_size = 4 : i32, iteration_stride = 16 : i32, iteration_current = -1 : i32 }
      aie.next_bd ^end
    ^end:
      aie.end
    }
  }
}

// -----

// iteration_current >= iteration_size.
module {
  aie.device(npu2) {
    %t = aie.tile(0, 1)
    %b = aie.buffer(%t) : memref<256xi32>
    %m = aie.memtile_dma(%t) {
      %s = aie.dma_start(S2MM, 0, ^bd0, ^end)
    ^bd0:
      // expected-error @+1 {{iteration_current must be in [0, iteration_size)}}
      aie.dma_bd(%b : memref<256xi32> offset = 0 len = 64) { iteration_size = 4 : i32, iteration_stride = 16 : i32, iteration_current = 4 : i32 }
      aie.next_bd ^end
    ^end:
      aie.end
    }
  }
}
