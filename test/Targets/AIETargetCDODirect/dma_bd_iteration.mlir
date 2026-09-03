//===- dma_bd_iteration.mlir -----------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-translate --aie-generate-cdo %s --cdo-debug=true --split-input-file 2>&1 | FileCheck %s

// --cdo-debug prints one line per register write:
//   Address: <addr>  Data@ <host-ptr> is: <value>
//
// The value is the fields OR'd at their bit offsets. iteration_size and the
// (word-scaled) iteration_stride are stored as (value - 1); iteration_current as-is.
// step = iteration_stride * elem_bytes / 4 (the stride in 32-bit words).
//
// xaie2pgbl_params.h:
//   Core  word 4 (+0x10):  current[24:19] | (size-1)[18:13] | (step-1)[12:0]
//   Mem   word 6 (+0x18):  current[28:23] | (size-1)[22:17] | (step-1)[16:0]
//   Shim  word 6 (+0x18):  current[31:26] | (size-1)[25:20] | (step-1)[19:0]

// MemTile, 4-dim access + iteration: 0<<23 | (4-1)<<17 | (16-1)
// CHECK: Address: {{0x[0-9A-Fa-f]+18}}  Data@ {{0x[0-9a-z]+}} is: 0x0006000F
module {
  aie.device(npu2) {
    %t = aie.tile(0, 1)
    %b = aie.buffer(%t) { address = 524288 : i32, sym_name = "b" } : memref<256xi32>
    %m = aie.memtile_dma(%t) {
      %s = aie.dma_start(S2MM, 0, ^bd0, ^end)
    ^bd0:
      aie.dma_bd(%b : memref<256xi32> offset = 0 len = 64 sizes = [2, 2, 2, 2] strides = [8, 4, 2, 1]) { bd_id = 0 : i32, iteration = #aie.bd_iteration<size = 4, stride = 16, current = 0> }
      aie.next_bd ^end
    ^end:
      aie.end
    }
  }
}

// -----

// MemTile, 4-dim access, no iteration: 0x00000000
// CHECK: Address: {{0x[0-9A-Fa-f]+18}}  Data@ {{0x[0-9a-z]+}} is: 0x00000000
module {
  aie.device(npu2) {
    %t = aie.tile(0, 1)
    %b = aie.buffer(%t) { address = 524288 : i32, sym_name = "b" } : memref<256xi32>
    %m = aie.memtile_dma(%t) {
      %s = aie.dma_start(S2MM, 0, ^bd0, ^end)
    ^bd0:
      aie.dma_bd(%b : memref<256xi32> offset = 0 len = 64 sizes = [2, 2, 2, 2] strides = [8, 4, 2, 1]) { bd_id = 0 : i32 }
      aie.next_bd ^end
    ^end:
      aie.end
    }
  }
}

// -----

// MemTile, iteration on a linear BD (no access dims): 0<<23 | (4-1)<<17 | (16-1)
// CHECK: Address: {{0x[0-9A-Fa-f]+18}}  Data@ {{0x[0-9a-z]+}} is: 0x0006000F
module {
  aie.device(npu2) {
    %t = aie.tile(0, 1)
    %b = aie.buffer(%t) { address = 524288 : i32, sym_name = "b" } : memref<256xi32>
    %m = aie.memtile_dma(%t) {
      %s = aie.dma_start(S2MM, 0, ^bd0, ^end)
    ^bd0:
      aie.dma_bd(%b : memref<256xi32> offset = 0 len = 64) { bd_id = 0 : i32, iteration = #aie.bd_iteration<size = 4, stride = 16, current = 0> }
      aie.next_bd ^end
    ^end:
      aie.end
    }
  }
}

// -----

// MemTile, nonzero current: 2<<23 | (4-1)<<17 | (16-1)
// CHECK: Address: {{0x[0-9A-Fa-f]+18}}  Data@ {{0x[0-9a-z]+}} is: 0x0106000F
module {
  aie.device(npu2) {
    %t = aie.tile(0, 1)
    %b = aie.buffer(%t) { address = 524288 : i32, sym_name = "b" } : memref<256xi32>
    %m = aie.memtile_dma(%t) {
      %s = aie.dma_start(S2MM, 0, ^bd0, ^end)
    ^bd0:
      aie.dma_bd(%b : memref<256xi32> offset = 0 len = 64) { bd_id = 0 : i32, iteration = #aie.bd_iteration<size = 4, stride = 16, current = 2> }
      aie.next_bd ^end
    ^end:
      aie.end
    }
  }
}

// -----

// MemTile, i64 iteration_stride 4 = step 8: 0<<23 | (4-1)<<17 | (8-1)
// CHECK: Address: {{0x[0-9A-Fa-f]+18}}  Data@ {{0x[0-9a-z]+}} is: 0x00060007
module {
  aie.device(npu2) {
    %t = aie.tile(0, 1)
    %b = aie.buffer(%t) { address = 524288 : i32, sym_name = "b" } : memref<256xi64>
    %m = aie.memtile_dma(%t) {
      %s = aie.dma_start(S2MM, 0, ^bd0, ^end)
    ^bd0:
      aie.dma_bd(%b : memref<256xi64> offset = 0 len = 64) { bd_id = 0 : i32, iteration = #aie.bd_iteration<size = 4, stride = 4, current = 0> }
      aie.next_bd ^end
    ^end:
      aie.end
    }
  }
}

// -----

// MemTile, max size: 0<<23 | (64-1)<<17 | (16-1)
// CHECK: Address: {{0x[0-9A-Fa-f]+18}}  Data@ {{0x[0-9a-z]+}} is: 0x007E000F
module {
  aie.device(npu2) {
    %t = aie.tile(0, 1)
    %b = aie.buffer(%t) { address = 524288 : i32, sym_name = "b" } : memref<256xi32>
    %m = aie.memtile_dma(%t) {
      %s = aie.dma_start(S2MM, 0, ^bd0, ^end)
    ^bd0:
      aie.dma_bd(%b : memref<256xi32> offset = 0 len = 64) { bd_id = 0 : i32, iteration = #aie.bd_iteration<size = 64, stride = 16, current = 0> }
      aie.next_bd ^end
    ^end:
      aie.end
    }
  }
}

// -----

// MemTile, max step: 0<<23 | (4-1)<<17 | (131072-1)
// CHECK: Address: {{0x[0-9A-Fa-f]+18}}  Data@ {{0x[0-9a-z]+}} is: 0x0007FFFF
module {
  aie.device(npu2) {
    %t = aie.tile(0, 1)
    %b = aie.buffer(%t) { address = 524288 : i32, sym_name = "b" } : memref<256xi32>
    %m = aie.memtile_dma(%t) {
      %s = aie.dma_start(S2MM, 0, ^bd0, ^end)
    ^bd0:
      aie.dma_bd(%b : memref<256xi32> offset = 0 len = 64) { bd_id = 0 : i32, iteration = #aie.bd_iteration<size = 4, stride = 131072, current = 0> }
      aie.next_bd ^end
    ^end:
      aie.end
    }
  }
}

// -----

// MemTile, size 1 disables iteration (stride ignored): 0x00000000
// CHECK: Address: {{0x[0-9A-Fa-f]+18}}  Data@ {{0x[0-9a-z]+}} is: 0x00000000
module {
  aie.device(npu2) {
    %t = aie.tile(0, 1)
    %b = aie.buffer(%t) { address = 524288 : i32, sym_name = "b" } : memref<256xi32>
    %m = aie.memtile_dma(%t) {
      %s = aie.dma_start(S2MM, 0, ^bd0, ^end)
    ^bd0:
      aie.dma_bd(%b : memref<256xi32> offset = 0 len = 64) { bd_id = 0 : i32, iteration = #aie.bd_iteration<size = 1, stride = 16, current = 0> }
      aie.next_bd ^end
    ^end:
      aie.end
    }
  }
}

// -----

// CoreTile: 0<<19 | (4-1)<<13 | (16-1)
// CHECK: Address: {{0x[0-9A-Fa-f]+10}}  Data@ {{0x[0-9a-z]+}} is: 0x0000600F
module {
  aie.device(npu2) {
    %t = aie.tile(0, 2)
    %b = aie.buffer(%t) { address = 1024 : i32, sym_name = "cb" } : memref<256xi32>
    %m = aie.mem(%t) {
      %s = aie.dma_start(S2MM, 0, ^bd0, ^end)
    ^bd0:
      aie.dma_bd(%b : memref<256xi32> offset = 0 len = 64) { bd_id = 0 : i32, iteration = #aie.bd_iteration<size = 4, stride = 16, current = 0> }
      aie.next_bd ^end
    ^end:
      aie.end
    }
  }
}

// -----

// ShimTile: iteration_stride larger than MemTile's: 0<<26 | (4-1)<<20 | (200000-1)
// CHECK: Address: {{0x[0-9A-Fa-f]+18}}  Data@ {{0x[0-9a-z]+}} is: 0x00330D3F
module {
  aie.device(npu2) {
    %b = aie.external_buffer { sym_name = "eb" } : memref<256xi32>
    %t = aie.tile(0, 0)
    aie.shim_dma(%t) {
      %s = aie.dma_start(S2MM, 0, ^bd0, ^end)
    ^bd0:
      aie.dma_bd(%b : memref<256xi32> offset = 0 len = 64) { bd_id = 0 : i32, iteration = #aie.bd_iteration<size = 4, stride = 200000, current = 0> }
      aie.next_bd ^end
    ^end:
      aie.end
    }
  }
}
