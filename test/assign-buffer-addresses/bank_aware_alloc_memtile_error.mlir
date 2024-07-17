//===- memtile_error.mlir ---------------------------------------*- MLIR
//-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: not aie-opt --aie-assign-buffer-addresses="alloc-scheme=bank-aware" %s
// 2>&1 | FileCheck %s CHECK:   error: Failed to allocate buffer: "a" with size:
// 528000 bytes. CHECK:      %b1 = aie.buffer(%0) { sym_name = "a" } :
// memref<132000xi32> CHECK:            ^ CHECK:    note: see current operation:
// %1 = "aie.buffer"(%0) <{sym_name = "a"}> : (index) -> memref<132000xi32>
// CHECK:    error: 'aie.tile' op All requested buffers doesn't fit in the
// available memory: Bank aware

// CHECK:      %0 = aie.tile(3, 1)
// CHECK:           ^
// CHECK:    note: see current operation: %0 = "aie.tile"() <{col = 3 : i32, row
// = 1 : i32}> : () -> index CHECK:    note: Current configuration of buffers in
// bank(s) : MemoryMap: CHECK:    (no stack allocated) CHECK:            bank :
// 0        0x0-0xFFFF CHECK:            bank : 1        0x10000-0x1FFFF CHECK:
// bank : 2        0x20000-0x2FFFF CHECK:            bank : 3 0x30000-0x3FFFF
// CHECK:            bank : 4        0x40000-0x4FFFF
// CHECK:            bank : 5        0x50000-0x5FFFF
// CHECK:            bank : 6        0x60000-0x6FFFF
// CHECK:            bank : 7        0x70000-0x7FFFF

module @test {
  aie.device(xcve2302) {
    % 0 = aie.tile(3, 1) % b1 =
              aie.buffer(% 0){sym_name = "a"} : memref<132000xi32> aie
                                                    .memtile_dma(% 0) {
      aie.end
    }
  }
}
