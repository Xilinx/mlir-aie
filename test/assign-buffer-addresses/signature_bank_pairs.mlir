// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt --aie-assign-buffer-addresses %s | FileCheck %s
// RUN: aie-opt --aie-assign-buffer-addresses %s | aie-opt --aie-assign-buffer-addresses | FileCheck %s

// Occupy the first allowed bank: each resource pair must use its other bank,
// not whichever unconstrained bank the allocator happens to prefer.
// CHECK: aie.buffer @ab{{.*}}mem_bank = 1 : i32
// CHECK: aie.buffer @ac{{.*}}mem_bank = 2 : i32
// CHECK: aie.buffer @ad{{.*}}mem_bank = 3 : i32
// CHECK: aie.buffer @bc{{.*}}mem_bank = 2 : i32
// CHECK: aie.buffer @bd{{.*}}mem_bank = 3 : i32
// CHECK: aie.buffer @cd{{.*}}mem_bank = 3 : i32
module {
  aie.device(npu2) {
    %t0 = aie.tile(0, 2)
    aie.buffer @pin0(%t0) {mem_bank = 0 : i32} : memref<16384xi8>
    aie.buffer @ab(%t0) : memref<64xi8, 9>
    %t1 = aie.tile(1, 2)
    aie.buffer @pin1(%t1) {mem_bank = 0 : i32} : memref<16384xi8>
    aie.buffer @ac(%t1) : memref<64xi8, 10>
    %t2 = aie.tile(2, 2)
    aie.buffer @pin2(%t2) {mem_bank = 0 : i32} : memref<16384xi8>
    aie.buffer @ad(%t2) : memref<64xi8, 11>
    %t3 = aie.tile(3, 2)
    aie.buffer @pin3(%t3) {mem_bank = 1 : i32} : memref<16384xi8>
    aie.buffer @bc(%t3) : memref<64xi8, 12>
    %t4 = aie.tile(4, 2)
    aie.buffer @pin4(%t4) {mem_bank = 1 : i32} : memref<16384xi8>
    aie.buffer @bd(%t4) : memref<64xi8, 13>
    %t5 = aie.tile(5, 2)
    aie.buffer @pin5(%t5) {mem_bank = 2 : i32} : memref<16384xi8>
    aie.buffer @cd(%t5) : memref<64xi8, 14>
  }
}
