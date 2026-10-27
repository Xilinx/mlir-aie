//===- bank_aware_prealloc_error.mlir --------------------------*- MLIR -*-===//
//
// Copyright (C) 2024 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --verify-diagnostics --aie-assign-buffer-addresses %s
module @test0 {
  aie.device(npu1) {

    %tile34 = aie.tile(3, 4)
    aie.buffer @buf0(%tile34) : memref<200xi32>
    aie.buffer @buf1(%tile34) : memref<100xi32>
    // expected-error @+1 {{'aie.buffer' op address 0x1000 lies in bank 0, but mem_bank requests bank 2}}
    aie.buffer @b(%tile34) { address = 4096 : i32, mem_bank = 2:i32 } : memref<1024xi32>
    aie.buffer @buf5(%tile34) : memref<800xi32>
  }
}

// -----

module @test1 {
  aie.device(npu1) {
    %tile34 = aie.tile(3, 4)
    aie.buffer @a(%tile34) { address = 0 : i32 } : memref<1024xi32>
    // expected-error@+1 {{'aie.buffer' op would override allocated address}}
    aie.buffer @b(%tile34) { address = 1024 : i32 } : memref<1024xi32>
  }
}


// -----

module @test2 {
  aie.device(npu1) {
    %tile34 = aie.tile(3, 4)
    aie.buffer @a(%tile34) { mem_bank = 2 : i32 } : memref<1024xi32>
    // expected-error@+1 {{'aie.buffer' op mem_bank attribute value is out of range}}
    aie.buffer @b(%tile34) { mem_bank = 4 : i32 } : memref<1024xi32>

  }
}

// -----

// A pinned address that starts inside the tile but whose buffer extends past
// the end of it is reported against the buffer, and not as an unattributed
// "allocated buffers exceeded available memory" for the whole tile.
module @test3 {
  aie.device(npu1) {
    %tile34 = aie.tile(3, 4)
    // expected-error@+1 {{'aie.buffer' op address attribute would place the buffer past the end of the tile's memory}}
    aie.buffer @a(%tile34) { address = 65024 : i32 } : memref<512xi32>
  }
}

// -----

// An explicit mem_bank stays a hard constraint. A buffer larger than a bank may
// straddle bank boundaries when the allocator chooses the placement. A request
// for a specific bank that cannot hold the buffer is an error.
module @test4 {
  aie.device(npu1) {
    %tile34 = aie.tile(3, 4)
    // expected-error@+1 {{'aie.buffer' op buffer "a" requires 32768 bytes, which cannot fit in bank 0 (16384 bytes total)}}
    aie.buffer @a(%tile34) { mem_bank = 0 : i32 } : memref<8192xi32>
  }
}

// -----

// The stack is occupied space like any other, so a pinned address landing
// inside it collides.
module @test5 {
  aie.device(npu1) {
    %tile34 = aie.tile(3, 4)
    // expected-error@+1 {{'aie.buffer' op would override allocated address}}
    aie.buffer @a(%tile34) { address = 512 : i32 } : memref<16xi32>
    aie.core(%tile34) {
      aie.end
    } {stack_size = 1024 : i32}
  }
}

// -----

// A reservation the tile cannot satisfy alongside its buffers fails. 60000
// bytes fit on their own, reserved at 0x400, but they leave too little for the
// three buffers, so the failure is reported against the first buffer with
// nowhere to go, with a remark naming the reservation as the cause.
module @test6 {
  aie.device(npu2) {
    // expected-warning @below {{Not all requested buffers fit in the available memory}}
    // expected-note @below {{Current configuration of buffers in bank(s)}}
    %tile_0_2 = aie.tile(0, 2)
    aie.buffer @a(%tile_0_2) : memref<4096xi8>
    // expected-error @below {{could not be placed: buffer "b" needs 4096 bytes}}
    aie.buffer @b(%tile_0_2) : memref<4096xi8>
    aie.buffer @c(%tile_0_2) : memref<4096xi8>
    aie.core(%tile_0_2) {
      aie.end
    } {stack_size = 1024 : i32, data_size = 60000 : i32}
  }
}

// -----

// A pinned address that is not aligned to the tile's load/store bus, on a
// buffer that did not opt out of alignment. Reported against the buffer, and
// terminal: the pass emits an error and does not retry.
module @unaligned_pinned_address {
  aie.device(npu1) {
    %tile34 = aie.tile(3, 4)
    // expected-error@+1 {{'aie.buffer' op address attribute value must be aligned to 256 bits when the aligned attribute is set}}
    aie.buffer @a(%tile34) { address = 1028 : i32 } : memref<16xi32>
  }
}

// -----

// A pinned address at or beyond the end of tile memory falls outside every
// bank, so the bank check rejects it before the past-the-end check runs.
module @pinned_address_outside_all_banks {
  aie.device(npu1) {
    %tile34 = aie.tile(3, 4)
    // expected-error@+1 {{'aie.buffer' op address attribute does not fall within any bank range}}
    aie.buffer @a(%tile34) { address = 65536 : i32 } : memref<16xi32>
  }
}
