//===- bank_aware_prealloc_error.mlir --------------------------*- MLIR -*-===//
//
// Copyright (C) 2024 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --verify-diagnostics --aie-assign-buffer-addresses='alloc-scheme=bank-aware' %s
module @test0 {
  aie.device(npu1) {

    // expected-error @+1 {{'aie.tile' op Bank-aware allocation failed.}}
    %tile34 = aie.tile(3, 4)
    %buf0 = aie.buffer(%tile34) : memref<200xi32>
    %buf1 = aie.buffer(%tile34) : memref<100xi32>
    // expected-error @+1 {{'aie.buffer' op mem_bank attribute is inconsistent with address attribute}}
    %buf2 = aie.buffer(%tile34) { sym_name = "b", address = 4096 : i32, mem_bank = 2:i32 } : memref<1024xi32>
    %buf5 = aie.buffer(%tile34) : memref<800xi32>
  }
}

// -----

module @test1 {
  aie.device(npu1) {
    // expected-error@+1 {{'aie.tile' op Bank-aware allocation failed.}}
    %tile34 = aie.tile(3, 4)
    %buf0 = aie.buffer(%tile34) { sym_name = "a", address = 0 : i32 } : memref<1024xi32>
    // expected-error@+1 {{'aie.buffer' op would override allocated address}}
    %buf2 = aie.buffer(%tile34) { sym_name = "b", address = 1024 : i32 } : memref<1024xi32>
  }
}


// -----

module @test2 {
  aie.device(npu1) {
    // expected-error@+1 {{'aie.tile' op Bank-aware allocation failed.}}
    %tile34 = aie.tile(3, 4)
    %buf0 = aie.buffer(%tile34) { sym_name = "a", mem_bank = 2 : i32 } : memref<1024xi32>
    // expected-error@+1 {{'aie.buffer' op mem_bank attribute value is out of range}}
    %buf1 = aie.buffer(%tile34) { sym_name = "b", mem_bank = 4 : i32 } : memref<1024xi32>

  }
}

// -----

// A pinned address that starts inside the tile but whose buffer runs off the
// end of it is reported against the buffer, rather than as an unattributed
// "allocated buffers exceeded available memory" for the whole tile.
module @test3 {
  aie.device(npu1) {
    // expected-error@+1 {{'aie.tile' op Bank-aware allocation failed.}}
    %tile34 = aie.tile(3, 4)
    // expected-error@+1 {{'aie.buffer' op address attribute would place the buffer past the end of the tile's memory}}
    %buf0 = aie.buffer(%tile34) { sym_name = "a", address = 65024 : i32 } : memref<512xi32>
  }
}

// -----

// An explicit mem_bank stays a hard constraint. A buffer larger than a bank
// may straddle bank boundaries only when it is the allocator choosing the
// placement; asking for a specific bank that cannot hold it is still an error.
module @test4 {
  aie.device(npu1) {
    // expected-error@+1 {{'aie.tile' op Bank-aware allocation failed.}}
    %tile34 = aie.tile(3, 4)
    // expected-error@+1 {{'aie.buffer' op would override existing mem_bank}}
    %buf0 = aie.buffer(%tile34) { sym_name = "a", mem_bank = 0 : i32 } : memref<8192xi32>
  }
}

// -----

// The stack is occupied space like any other: a pinned address landing inside
// it collides rather than silently overlapping it.
module @test5 {
  aie.device(npu1) {
    // expected-error@+1 {{'aie.tile' op Bank-aware allocation failed.}}
    %tile34 = aie.tile(3, 4)
    // expected-error@+1 {{'aie.buffer' op would override allocated address}}
    %buf0 = aie.buffer(%tile34) { sym_name = "a", address = 512 : i32 } : memref<16xi32>
    aie.core(%tile34) {
      aie.end
    } {stack_size = 1024 : i32}
  }
}

// -----

// A reservation larger than any run the tile can offer is reported against the
// tile rather than silently producing a layout the core cannot link into.
module @test6 {
  aie.device(npu2) {
    // expected-warning @below {{buffers leave only 52224 contiguous bytes for the core's data sections, which need 60000 bytes}}
    // expected-error @below {{'aie.tile' op Bank-aware allocation failed.}}
    %tile_0_2 = aie.tile(0, 2)
    %a = aie.buffer(%tile_0_2) {sym_name = "a"} : memref<4096xi8>
    %b = aie.buffer(%tile_0_2) {sym_name = "b"} : memref<4096xi8>
    %c = aie.buffer(%tile_0_2) {sym_name = "c"} : memref<4096xi8>
    aie.core(%tile_0_2) {
      aie.end
    } {stack_size = 1024 : i32, reserved_data_size = 60000 : i32}
  }
}
