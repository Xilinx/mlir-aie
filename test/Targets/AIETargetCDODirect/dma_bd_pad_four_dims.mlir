//===- dma_bd_pad_four_dims.mlir -------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-translate --aie-generate-cdo %s --cdo-debug=true --split-input-file 2>&1 | FileCheck %s

// A MemTile walk over four dimensions keeps its padding on d0 to d2.
//
// xaie2pgbl_params.h:
//   Mem word 4 (+0x10):  d2 pad before[30:27] | (d2 wrap)[26:17] | (d2 step-1)[16:0]
//   Mem word 5 (+0x14):  d2 pad after[31:28] | d1[27:23] | d0[22:17] | (d3 step-1)[16:0]

// d2 padded after by 1: 1<<28 | (64-1)
// CHECK: Address: {{0x[0-9A-Fa-f]+14}}  Data@ {{0x[0-9a-z]+}} is: 0x1000003F
module {
  aie.device(npu2) {
    %t = aie.tile(0, 1)
    %b = aie.buffer(%t) { address = 524288 : i32, sym_name = "b" } : memref<256xi32>
    %m = aie.memtile_dma(%t) {
      %s = aie.dma_start(MM2S, 0, ^bd0, ^end)
    ^bd0:
      aie.dma_bd(%b : memref<256xi32> offset = 0 len = 128 sizes = [2, 3, 2, 8] strides = [64, 16, 8, 1] pad [<const_pad_before = 0, const_pad_after = 0>, <const_pad_before = 0, const_pad_after = 1>, <const_pad_before = 0, const_pad_after = 0>, <const_pad_before = 0, const_pad_after = 0>]) { bd_id = 0 : i32 }
      aie.next_bd ^end
    ^end:
      aie.end
    }
  }
}

// -----

// d2 padded before by 1: 1<<27 | 3<<17 | (16-1)
// CHECK: Address: {{0x[0-9A-Fa-f]+10}}  Data@ {{0x[0-9a-z]+}} is: 0x0806000F
module {
  aie.device(npu2) {
    %t = aie.tile(0, 1)
    %b = aie.buffer(%t) { address = 524288 : i32, sym_name = "b" } : memref<256xi32>
    %m = aie.memtile_dma(%t) {
      %s = aie.dma_start(MM2S, 0, ^bd0, ^end)
    ^bd0:
      aie.dma_bd(%b : memref<256xi32> offset = 0 len = 128 sizes = [2, 3, 2, 8] strides = [64, 16, 8, 1] pad [<const_pad_before = 0, const_pad_after = 0>, <const_pad_before = 1, const_pad_after = 0>, <const_pad_before = 0, const_pad_after = 0>, <const_pad_before = 0, const_pad_after = 0>]) { bd_id = 0 : i32 }
      aie.next_bd ^end
    ^end:
      aie.end
    }
  }
}

// -----

// d1 padded after by 1 and d0 by 8: 1<<23 | 8<<17 | (64-1)
// CHECK: Address: {{0x[0-9A-Fa-f]+14}}  Data@ {{0x[0-9a-z]+}} is: 0x0090003F
module {
  aie.device(npu2) {
    %t = aie.tile(0, 1)
    %b = aie.buffer(%t) { address = 524288 : i32, sym_name = "b" } : memref<512xi32>
    %m = aie.memtile_dma(%t) {
      %s = aie.dma_start(MM2S, 0, ^bd0, ^end)
    ^bd0:
      aie.dma_bd(%b : memref<512xi32> offset = 0 len = 288 sizes = [2, 3, 2, 8] strides = [64, 16, 8, 1] pad [<const_pad_before = 0, const_pad_after = 0>, <const_pad_before = 0, const_pad_after = 0>, <const_pad_before = 0, const_pad_after = 1>, <const_pad_before = 0, const_pad_after = 8>]) { bd_id = 0 : i32 }
      aie.next_bd ^end
    ^end:
      aie.end
    }
  }
}
