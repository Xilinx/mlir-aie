//===- npu_blockwrite_data.mlir --------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-translate --aie-npu-to-binary -aie-output-binary=false %s | FileCheck %s

// Blockwrites sharing a global each carry its words, and a 32-bit global of
// any signedness is written as its bits.

// CHECK: 00201000
// CHECK-NEXT: 00000018
// CHECK-NEXT: 12345678
// CHECK-NEXT: FFFFFFFF
// CHECK: 00202000
// CHECK-NEXT: 00000018
// CHECK-NEXT: 12345678
// CHECK-NEXT: FFFFFFFF
// CHECK: 00203000
// CHECK-NEXT: 00000018
// CHECK-NEXT: FFFFFFFE
// CHECK-NEXT: 00000007
// CHECK: 00204000
// CHECK-NEXT: 00000014
// CHECK-NEXT: 80000000
module {
  aie.device(npu2) {
    memref.global "private" constant @words : memref<2xi32> = dense<[305419896, -1]>
    memref.global "private" constant @signed : memref<2xsi32> = dense<[-2, 7]>
    memref.global "private" constant @unsigned : memref<1xui32> = dense<[2147483648]>
    aie.runtime_sequence() {
      %0 = memref.get_global @words : memref<2xi32>
      aiex.npu.blockwrite(%0) {address = 0x1000 : ui32, column = 0 : i32, row = 2 : i32} : memref<2xi32>
      %1 = memref.get_global @words : memref<2xi32>
      aiex.npu.blockwrite(%1) {address = 0x2000 : ui32, column = 0 : i32, row = 2 : i32} : memref<2xi32>
      %2 = memref.get_global @signed : memref<2xsi32>
      aiex.npu.blockwrite(%2) {address = 0x3000 : ui32, column = 0 : i32, row = 2 : i32} : memref<2xsi32>
      %3 = memref.get_global @unsigned : memref<1xui32>
      aiex.npu.blockwrite(%3) {address = 0x4000 : ui32, column = 0 : i32, row = 2 : i32} : memref<1xui32>
    }
  }
}
