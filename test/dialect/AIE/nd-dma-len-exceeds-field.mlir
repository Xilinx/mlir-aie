//===- nd-dma-len-exceeds-field.mlir ---------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --verify-diagnostics --split-input-file %s

module {
  aie.device(npu1) {
    %t1 = aie.tile(1, 1)
    %buf = aie.buffer(%t1) : memref<131072xi32>
    %mem = aie.memtile_dma(%t1) {
      aie.dma_start("MM2S", 0, ^bd0, ^end)
      ^bd0:
        // expected-error@+1 {{buffer descriptor length (131072 32-bit words) exceeds the maximum of 131071 words supported by this tile type}}
        aie.dma_bd(%buf : memref<131072xi32> offset = 0 len = 131072)
        aie.next_bd ^end
      ^end:
        aie.end
    }
  }
}

// -----

module {
  aie.device(npu1) {
    %t1 = aie.tile(1, 1)
    %buf = aie.buffer(%t1) : memref<131071xi32>
    %mem = aie.memtile_dma(%t1) {
      aie.dma_start("MM2S", 0, ^bd0, ^end)
      ^bd0:
        aie.dma_bd(%buf : memref<131071xi32> offset = 0 len = 131071)
        aie.next_bd ^end
      ^end:
        aie.end
    }
  }
}

// -----

module {
  aie.device(npu1) {
    %tile02 = aie.tile(0, 2)
    %buf02 = aie.buffer(%tile02) : memref<16384xi32>
    %mem02 = aie.mem(%tile02) {
      %srcDma = aie.dma_start("MM2S", 0, ^bd0, ^end)
      ^bd0:
        // expected-error@+1 {{buffer descriptor length (16384 32-bit words) exceeds the maximum of 16383 words supported by this tile type}}
        aie.dma_bd(%buf02 : memref<16384xi32> offset = 0 len = 16384)
        aie.next_bd ^end
      ^end:
        aie.end
    }
  }
}

// -----

module {
  aie.device(npu1) {
    %tile02 = aie.tile(0, 2)
    %buf02 = aie.buffer(%tile02) : memref<16383xi32>
    %mem02 = aie.mem(%tile02) {
      %srcDma = aie.dma_start("MM2S", 0, ^bd0, ^end)
      ^bd0:
        aie.dma_bd(%buf02 : memref<16383xi32> offset = 0 len = 16383)
        aie.next_bd ^end
      ^end:
        aie.end
    }
  }
}
