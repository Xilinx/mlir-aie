//===- bank_aware_alloc_memory_exhausted.mlir ------------------*- MLIR -*-===//
//
// Copyright (C) 2024 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// The design asks for 69632 bytes on a 65536-byte core tile, so neither scheme
// can place it. Both are expected to report, and the pass to fail.
//
// The RUN line used to omit `| FileCheck %s`, so none of the assertions below
// ran and the test only checked the exit code; three of them had gone stale
// enough that they could never have matched.
// RUN: not aie-opt --aie-assign-bd-ids --aie-assign-buffer-addresses %s 2>&1 | FileCheck %s

// CHECK: warning: Failed to allocate buffer: "_anonymous5" with size: 4096 bytes.
// CHECK: warning: Not all requested buffers fit in the available memory.
// CHECK: note: Current configuration of buffers in bank(s) : MemoryMap:
// CHECK: (no stack allocated)
// CHECK:         bank : 0        0x0-0x3FFF
// CHECK:                 _anonymous0     : 0x0-0x5FFF     (24576 bytes)
// CHECK:         bank : 1        0x4000-0x7FFF
// CHECK:                 _anonymous1     : 0x6000-0xBFFF          (24576 bytes)
// CHECK:         bank : 2        0x8000-0xBFFF
// CHECK:         bank : 3        0xC000-0xFFFF
// CHECK:                 _anonymous2     : 0xC000-0xD7FF          (6144 bytes)
// CHECK:                 _anonymous3     : 0xD800-0xEFFF          (6144 bytes)
// CHECK:                 _anonymous4     : 0xF000-0xFFFF          (4096 bytes)
// CHECK: warning: Bank-aware allocation failed, trying basic sequential allocation.
// CHECK: error: 'aie.tile' op allocated buffers exceeded available memory
// CHECK: error: 'aie.tile' op Basic sequential allocation also failed.

module {
  aie.device(npu1_2col) {
    %tile_0_2 = aie.tile(0, 2)
    %C_L1L2_0_0_buff_0 = aie.buffer(%tile_0_2) : memref<64x96xf32>
    %C_L1L2_0_0_buff_1 = aie.buffer(%tile_0_2) : memref<64x96xf32>
    %B_L2L1_0_0_cons_buff_0 = aie.buffer(%tile_0_2) : memref<32x96xbf16>
    %B_L2L1_0_0_cons_buff_1 = aie.buffer(%tile_0_2) : memref<32x96xbf16>
    %A_L2L1_0_0_cons_buff_0 = aie.buffer(%tile_0_2) : memref<64x32xbf16>
    %A_L2L1_0_0_cons_buff_1 = aie.buffer(%tile_0_2) : memref<64x32xbf16>
  }
}
