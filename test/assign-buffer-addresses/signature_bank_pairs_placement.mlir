// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt --aie-assign-buffer-addresses --split-input-file %s | FileCheck %s
// RUN: aie-opt --aie-assign-buffer-addresses --split-input-file %s | aie-opt --aie-assign-buffer-addresses --split-input-file | FileCheck %s

// Backtrack from placing AB in bank A: AC has no room in bank C.
// CHECK: aie.buffer @ab{{.*}}address = 16384 : i32, mem_bank = 1 : i32
// CHECK: aie.buffer @ac{{.*}}address = 0 : i32, mem_bank = 0 : i32
module {
  aie.device(npu2) {
    %t = aie.tile(0, 2)
    aie.buffer @pin(%t) {address = 32768 : i32} : memref<16384xi8>
    aie.buffer @ab(%t) : memref<16384xi8, 9>
    aie.buffer @ac(%t) : memref<16384xi8, 10>
  }
}

// -----

// Adjacent allowed banks can hold a spanning buffer.
// CHECK: aie.buffer @bc_spanning{{.*}}address = 16384 : i32, mem_bank = 1 : i32
module {
  aie.device(npu2) {
    %t = aie.tile(0, 2)
    aie.buffer @bc_spanning(%t) : memref<32768xi8, 12>
  }
}

// -----

// An explicit bank narrows a pair; explicit addresses remain unchanged.
// CHECK: aie.buffer @ac_pinned{{.*}}mem_bank = 2 : i32
// CHECK: aie.buffer @bd_address{{.*}}address = 49152 : i32, mem_bank = 3 : i32
// CHECK: aie.buffer @ab_address_spanning{{.*}}address = 16352 : i32, mem_bank = 0 : i32
module {
  aie.device(npu2) {
    %t = aie.tile(0, 2)
    aie.buffer @ac_pinned(%t) {mem_bank = 2 : i32} : memref<64xi8, 10>
    aie.buffer @bd_address(%t) {address = 49152 : i32, mem_bank = 3 : i32} : memref<64xi8, 13>
    aie.buffer @ab_address_spanning(%t) {address = 16352 : i32} : memref<64xi8, 9>
  }
}

// -----

// Empty buffers need no free bytes, even when both allowed banks are full.
// CHECK: aie.buffer @empty_ac{{.*}}address = 0 : i32, mem_bank = 0 : i32
module {
  aie.device(npu2) {
    %t = aie.tile(0, 2)
    aie.buffer @a(%t) {address = 0 : i32} : memref<16384xi8>
    aie.buffer @c(%t) {address = 32768 : i32} : memref<16384xi8>
    aie.buffer @empty_ac(%t) : memref<0xi8, 10>
  }
}
