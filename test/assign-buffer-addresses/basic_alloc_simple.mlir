//===- basic_alloc_simple.mlir ---------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --aie-assign-buffer-addresses="alloc-scheme=basic-sequential" %s | FileCheck %s
// CHECK:   {{.*}} aie.buffer({{.*}}) {address = 3104 : i32, sym_name = "a"} : memref<16xi8>
// CHECK:   {{.*}} aie.buffer({{.*}}) {address = 1024 : i32, sym_name = "b"} : memref<512xi32>
// CHECK:   {{.*}} aie.buffer({{.*}}) {address = 3072 : i32, sym_name = "c"} : memref<16xi16>
// CHECK:   {{.*}} aie.buffer({{.*}}) {address = 1024 : i32, sym_name = "_anonymous0"} : memref<500xi32>
// CHECK:   {{.*}} aie.buffer({{.*}}) {address = 4096 : i32, sym_name = "a"} : memref<1024xi32>
// CHECK:   {{.*}} aie.buffer({{.*}}) {address = 8192 : i32, sym_name = "b"} : memref<1031xi32>
// CHECK:   {{.*}} aie.buffer({{.*}}) {address = 12316 : i32, aligned = false, sym_name = "c"} : memref<100xi32>
// CHECK:   {{.*}} aie.buffer({{.*}}) {address = 12736 : i32, sym_name = "d"} : memref<1xi32>
// CHECK:   {{.*}} aie.buffer({{.*}}) {address = 4096 : i32, sym_name = "a2"} : memref<1xi32>
// CHECK:   {{.*}} aie.buffer({{.*}}) {address = 4100 : i32, aligned = false, sym_name = "b2"} : memref<1024xi32>
module @test {
  aie.device(xcvc1902) {
    %0 = aie.tile(3, 3)
    %b1 = aie.buffer(%0) { sym_name = "a" } : memref<16xi8>
    %1 = aie.buffer(%0) { sym_name = "b" } : memref<512xi32>
    %b2 = aie.buffer(%0) { sym_name = "c" } : memref<16xi16>
    %3 = aie.tile(4, 4)
    %4 = aie.buffer(%3) : memref<500xi32>
    aie.core(%0) {
      aie.end
    }
    aie.core(%3) {
      aie.end
    }
  }
}

// -----

module @test_align{

  aie.device(npu2){
      %0 = aie.tile(3, 3)
      %b1 = aie.buffer(%0) { address=4096 : i32,  sym_name = "a"} : memref<1024xi32>
      %b2 = aie.buffer(%0) { sym_name = "b"} : memref<1031xi32>
      %b3 = aie.buffer(%0) { sym_name = "c", aligned=false } : memref<100xi32>
      %b4 = aie.buffer(%0) { sym_name = "d"} : memref<1xi32>

      aie.core(%0){
        aie.end
      }{stackSize = 4096 : i32}
  }


}


// -----

module @test_align2{

  aie.device(npu2) {
    %0 = aie.tile(3, 3)
    %b1 = aie.buffer(%0) { address = 4096 : i32, sym_name = "a2" } : memref<1xi32>
    %b2 = aie.buffer(%0) { address = 4100 : i32, sym_name = "b2", aligned=false } : memref<1024xi32>

    aie.core(%0) {
      aie.end
    }{stackSize = 4096 : i32}

  }

}



// -----
// CHECK: aie.buffer(%{{.*}}) {address = 0 : i32, aligned = false, sym_name = "p1"} : memref<10xi8>
// CHECK: aie.buffer(%{{.*}}) {address = 64 : i32, sym_name = "p2"} : memref<32xi8>
// CHECK: aie.buffer(%{{.*}}) {address = 96 : i32, sym_name = "b"} : memref<40xi8>

module @test {
  aie.device(npu1) {
    %tile22 = aie.tile(2, 2)

    // Unaligned pre-allocated buffer, occupies 0x00..0x09.
    %p1 = aie.buffer(%tile22) { sym_name = "p1", address = 0 : i32, aligned = false } : memref<10xi8>
    // Aligned pre-allocated buffer, occupies 0x40..0x5F.
    %p2 = aie.buffer(%tile22) { sym_name = "p2", address = 64 : i32 } : memref<32xi8>
    // Dynamic aligned buffer (40 bytes). Pre-fix: placed at 0x20 and aliases
    // p2. Post-fix: must be placed at 0x60. Recall npu1 has alignment requirement of 128-bit witdth
    %b  = aie.buffer(%tile22) { sym_name = "b" } : memref<40xi8>
  }
}