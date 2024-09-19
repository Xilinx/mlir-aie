//===- simple.mlir ---------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: not aie-opt --aie-assign-bd-ids --aie-assign-buffer-addresses %s
  
// CHECK: error: Failed to allocate buffer: "_anonymous0" with size: 24576 bytes.
// CHECK:     %C_L1L2_0_0_buff_0 = aie.buffer(%tile_0_2) : memref<64x96xf32> 
// CHECK:                          ^
// CHECK: note: see current operation: %1 = "aie.buffer"(%0) <{sym_name = "_anonymous0"}> : (index) -> memref<64x96xf32>
// CHECK: error: 'aie.tile' op All requested buffers don't fit in the available memory: Bank aware

// CHECK: note: see current operation: %0 = "aie.tile"() <{col = 0 : i32, row = 2 : i32}> : () -> index
// CHECK: note: Current configuration of buffers in bank(s) : MemoryMap:
// CHECK: (no stack allocated)
// CHECK:         bank : 0        0x0-0x3FFF
// CHECK:         bank : 1        0x4000-0x7FFF
// CHECK:         bank : 2        0x8000-0xBFFF
// CHECK:         bank : 3        0xC000-0xFFFF

// CHECK: error: 'aie.tile' op allocated buffers exceeded available memory: Sequential
// CHECK: (no stack allocated)

// CHECK: note: see current operation: %0 = "aie.tile"() <{col = 0 : i32, row = 2 : i32}> : () -> index
// CHECK:         _anonymous0     : 0x0-0x5FFF    (24576 bytes)
// CHECK:         _anonymous1     : 0x6000-0xBFFF         (24576 bytes)
// CHECK:         _anonymous2     : 0xC000-0xD7FF         (6144 bytes)
// CHECK:         _anonymous3     : 0xD800-0xEFFF         (6144 bytes)
// CHECK:         _anonymous4     : 0xF000-0xFFFF         (4096 bytes)
// CHECK:         _anonymous5     : 0x10000-0x10FFF       (4096 bytes)

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