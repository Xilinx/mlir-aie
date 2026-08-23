//===----------------------------------------------------------------------===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// DMABDOp::verify's size/stride checks were off by (roughly) 512: the size
// check rejected size >= 513 while claiming a limit of 1023, and the stride
// check rejected stride >= 524288 while claiming a limit of 1048576. Both
// are now derived from the tile-aware target-model accessors
// (getDmaBdWrapBits / getDmaBdStepBits), so the checked bound always matches
// the reported bound.
//
// Size cases run on a MemTile (10-bit unbiased wrap => max 1023), matching
// the original bug report and the hardware evidence (BUGFIX-PLAN.md
// experiments 9a/9b/9c: sizes 512, 600, and 1023 all measured correct via
// the task path; only the static-path verifier was wrong).
//
// Stride cases run on a ShimNOCTile rather than a MemTile: a MemTile's step
// field is 17-bit (max stride 131072), which is smaller than the numeric
// values this bug's original report uses (524288, 1048575, 1048576) -- those
// values correspond to the Shim's 20-bit step field (max stride 1048576).
// Moving these cases to a ShimNOCTile keeps the numbers meaningful under the
// tile-aware bound while preserving the original test's intent.

// RUN: aie-opt --verify-diagnostics --split-input-file %s

// Size cases: MemTile, 10-bit unbiased wrap, max 1023.

// Control: size = 512 is comfortably under both the old buggy threshold (513)
// and the real limit (1023); must stay accepted.
module {
  aie.device(npu2) {
    %t1 = aie.tile(1, 1)
    %buf = aie.buffer(%t1) : memref<16384xi32>
    aie.memtile_dma(%t1) {
      aie.dma_start(MM2S, 0, ^bd0, ^end)
      ^bd0:
        aie.dma_bd(%buf : memref<16384xi32> offset = 0 len = 4096 sizes = [512, 4] strides = [8, 1])
        aie.next_bd ^end
      ^end:
        aie.end
    }
  }
}

// -----

// Load-bearing: size = 600 is past the old buggy 513 cutoff but well under
// the real 1023 limit; hardware-measured correct (experiment 9a). Must
// become accepted.
module {
  aie.device(npu2) {
    %t1 = aie.tile(1, 1)
    %buf = aie.buffer(%t1) : memref<16384xi32>
    aie.memtile_dma(%t1) {
      aie.dma_start(MM2S, 0, ^bd0, ^end)
      ^bd0:
        aie.dma_bd(%buf : memref<16384xi32> offset = 0 len = 4096 sizes = [600, 4] strides = [8, 1])
        aie.next_bd ^end
      ^end:
        aie.end
    }
  }
}

// -----

// Upper boundary: size = 1023 is exactly the real 10-bit wrap limit
// (hardware-measured correct, experiment 9b). Must become accepted.
module {
  aie.device(npu2) {
    %t1 = aie.tile(1, 1)
    %buf = aie.buffer(%t1) : memref<16384xi32>
    aie.memtile_dma(%t1) {
      aie.dma_start(MM2S, 0, ^bd0, ^end)
      ^bd0:
        aie.dma_bd(%buf : memref<16384xi32> offset = 0 len = 4096 sizes = [1023, 4] strides = [8, 1])
        aie.next_bd ^end
      ^end:
        aie.end
    }
  }
}

// -----

// Negative companion: size = 1024 is one past the real limit and must stay
// rejected (experiment 9c: task path correctly rejects this).
module {
  aie.device(npu2) {
    %t1 = aie.tile(1, 1)
    %buf = aie.buffer(%t1) : memref<16384xi32>
    aie.memtile_dma(%t1) {
      aie.dma_start(MM2S, 0, ^bd0, ^end)
      ^bd0:
        // expected-error@+1 {{Size may not exceed 1023.}}
        aie.dma_bd(%buf : memref<16384xi32> offset = 0 len = 4096 sizes = [1024, 4] strides = [8, 1])
        aie.next_bd ^end
      ^end:
        aie.end
    }
  }
}

// -----

// Stride cases: ShimNOCTile, 20-bit step field, biased by actual - 1, so
// the field admits an actual stride up to 2^20 == 1048576 (inclusive).

// Control: stride = 524287 is one below the old buggy cutoff (524288);
// must stay accepted.
module {
  aie.device(npu2) {
    %t0 = aie.tile(0, 0)
    aie.shim_dma(%t0) {
      aie.dma_start(MM2S, 0, ^bd0, ^end)
      ^bd0:
        %buf = aie.external_buffer { sym_name = "buf_524287" } : memref<524291xi32>
        aie.dma_bd(%buf : memref<524291xi32> offset = 0 len = 8 sizes = [2, 2] strides = [524287, 1])
        aie.next_bd ^end
      ^end:
        aie.end
    }
  }
}

// -----

// stride = 524288 is exactly the old buggy cutoff but well under the real
// 1048576 limit. Must become accepted.
module {
  aie.device(npu2) {
    %t0 = aie.tile(0, 0)
    aie.shim_dma(%t0) {
      aie.dma_start(MM2S, 0, ^bd0, ^end)
      ^bd0:
        %buf = aie.external_buffer { sym_name = "buf_524288" } : memref<524292xi32>
        aie.dma_bd(%buf : memref<524292xi32> offset = 0 len = 8 sizes = [2, 2] strides = [524288, 1])
        aie.next_bd ^end
      ^end:
        aie.end
    }
  }
}

// -----

// stride = 1048575 is one below the real limit. Must become accepted.
module {
  aie.device(npu2) {
    %t0 = aie.tile(0, 0)
    aie.shim_dma(%t0) {
      aie.dma_start(MM2S, 0, ^bd0, ^end)
      ^bd0:
        %buf = aie.external_buffer { sym_name = "buf_1048575" } : memref<1048579xi32>
        aie.dma_bd(%buf : memref<1048579xi32> offset = 0 len = 8 sizes = [2, 2] strides = [1048575, 1])
        aie.next_bd ^end
      ^end:
        aie.end
    }
  }
}

// -----

// Upper boundary: stride = 1048576 is exactly 2^20, the true limit for a
// 20-bit step field biased by actual - 1. This was rejected under the old
// hardcoded check (whose own error message claimed this exact number as the
// limit, yet rejected it anyway); it must become accepted now that the
// check and the message agree.
module {
  aie.device(npu2) {
    %t0 = aie.tile(0, 0)
    aie.shim_dma(%t0) {
      aie.dma_start(MM2S, 0, ^bd0, ^end)
      ^bd0:
        %buf = aie.external_buffer { sym_name = "buf_1048576" } : memref<1048580xi32>
        aie.dma_bd(%buf : memref<1048580xi32> offset = 0 len = 8 sizes = [2, 2] strides = [1048576, 1])
        aie.next_bd ^end
      ^end:
        aie.end
    }
  }
}

// -----

// Negative companion: stride = 1048577 is one past the real limit and must
// stay rejected. (The original bug report's negative case used 1048576,
// which was the message's stated limit under the old hardcoded check; with
// the tile-aware fix, 1048576 is itself the true, inclusive limit -- see the
// case above -- so the one-past value that now pins the true boundary is
// 1048577.)
module {
  aie.device(npu2) {
    %t0 = aie.tile(0, 0)
    aie.shim_dma(%t0) {
      aie.dma_start(MM2S, 0, ^bd0, ^end)
      ^bd0:
        %buf = aie.external_buffer { sym_name = "buf_1048577" } : memref<1048581xi32>
        // expected-error@+1 {{Stride may not exceed 1048576.}}
        aie.dma_bd(%buf : memref<1048581xi32> offset = 0 len = 8 sizes = [2, 2] strides = [1048577, 1])
        aie.next_bd ^end
      ^end:
        aie.end
    }
  }
}
