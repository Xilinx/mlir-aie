//===----------------------------------------------------------------------===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --verify-diagnostics %s

// Iteration stride must be positive when iteration size > 1.
module {
  aie.device(npu1) {
    %t = aie.tile(2, 1)
    %b = aie.buffer(%t) { sym_name = "b" } : memref<256xi32>
    %m = aie.memtile_dma(%t) {
      %s = aie.dma_start("MM2S", 0, ^bd0, ^end)
    ^bd0:
      // expected-error@+1 {{Iteration stride must be a positive integer when iteration size > 1.}}
      aie.dma_bd(%b : memref<256xi32>, 0, 64, [<size = 8, stride = 8>, <size = 8, stride = 1>]) {iter_size = 4 : i32, iter_stride = 0 : i32}
      aie.next_bd ^end
    ^end:
      aie.end
    }
  }
}

// -----

// A core tile allows at most 3 data-layout dimensions.
module {
  aie.device(npu1) {
    %t = aie.tile(3, 3)
    %b = aie.buffer(%t) { sym_name = "b" } : memref<256xi32>
    %m = aie.mem(%t) {
      %s = aie.dma_start("MM2S", 0, ^bd0, ^end)
    ^bd0:
      // expected-error@+1 {{Cannot give more than 3 dimensions for step sizes and wraps in this tile}}
      aie.dma_bd(%b : memref<256xi32>, 0, 64, [<size = 2, stride = 32>, <size = 2, stride = 16>, <size = 2, stride = 8>, <size = 8, stride = 1>])
      aie.next_bd ^end
    ^end:
      aie.end
    }
  }
}

// -----

// A memtile allows 4 data-layout dimensions (this must verify cleanly).
module {
  aie.device(npu1) {
    %t = aie.tile(2, 1)
    %b = aie.buffer(%t) { sym_name = "b" } : memref<1024xi32>
    %m = aie.memtile_dma(%t) {
      %s = aie.dma_start("MM2S", 0, ^bd0, ^end)
    ^bd0:
      aie.dma_bd(%b : memref<1024xi32>, 0, 64, [<size = 2, stride = 256>, <size = 2, stride = 64>, <size = 2, stride = 8>, <size = 8, stride = 1>])
      aie.next_bd ^end
    ^end:
      aie.end
    }
  }
}
