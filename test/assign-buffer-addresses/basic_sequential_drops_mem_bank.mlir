//===- basic_sequential_drops_mem_bank.mlir -------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// basic-sequential has no notion of banks, so it drops a user's mem_bank pin
// and warns. Only an explicit selection of the scheme reaches this: the
// fallback path refuses to drop a pin.

// RUN: aie-opt --aie-assign-buffer-addresses="alloc-scheme=basic-sequential" %s 2>&1 | FileCheck %s

module @test {
  aie.device(xcvc1902) {
    %t = aie.tile(4, 4)
    // CHECK: warning: basic-sequential allocation ignores mem_bank; dropping the pin on: "pinned"
    // CHECK: {address = {{[0-9]+}} : i32, sym_name = "pinned"}
    %b = aie.buffer(%t) { sym_name = "pinned", mem_bank = 2 : i32 } : memref<64xi32>
    aie.core(%t) {
      aie.end
    }
  }
}
