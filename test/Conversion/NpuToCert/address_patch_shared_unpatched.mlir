// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Only a patched blockwrite needs a private template: an unpatched one keeps
// sharing the original, and a template used by one BD is patched in place.

// RUN: aie-opt --aie-npu-to-cert --split-input-file %s | FileCheck %s

// CHECK-LABEL: aie.device(xcve3858)
// CHECK-DAG: memref.global "private" constant @shared : memref<2xi32> = dense<[1, 0]>
// CHECK-DAG: memref.global "private" constant @[[COPY:shared_[0-9]+]] : memref<2xi32> = dense<[1, 8]>
// CHECK-DAG: memref.global "private" constant @alone : memref<2xi32> = dense<[3, 4]>
// CHECK-DAG: aiex.cert.uc_dma_bd @shared, 36864, 2, false
// CHECK-DAG: aiex.cert.uc_dma_bd @[[COPY]], 36912, 2, false
// CHECK-DAG: aiex.cert.uc_dma_bd @alone, 36960, 2, false
// CHECK: aiex.cert.apply_offset_57(@[[COPY]], 1, 0)
// CHECK: aiex.cert.apply_offset_57(@alone, 1, 1)
module {
  aie.device(xcve3858) {
    memref.global "private" constant @shared : memref<2xi32> = dense<[1, 0]>
    memref.global "private" constant @alone : memref<2xi32> = dense<[3, 0]>
    aie.runtime_sequence @seq(%a: memref<64xi32>, %b: memref<64xi32>) {
      %0 = memref.get_global @shared : memref<2xi32>
      aiex.npu.blockwrite(%0) {address = 36864 : ui32} : memref<2xi32>
      %1 = memref.get_global @shared : memref<2xi32>
      aiex.npu.blockwrite(%1) {address = 36912 : ui32} : memref<2xi32>
      %c8 = arith.constant 8 : i32
      aiex.npu.address_patch(%c8 : i32) {addr = 36916 : ui32, arg_idx = 0 : i32}
      %2 = memref.get_global @alone : memref<2xi32>
      aiex.npu.blockwrite(%2) {address = 36960 : ui32} : memref<2xi32>
      %c4 = arith.constant 4 : i32
      aiex.npu.address_patch(%c4 : i32) {addr = 36964 : ui32, arg_idx = 1 : i32}
    }
  }
}

// -----

// One getter feeding several blockwrites (as after CSE) is a shared template
// too: an unpatched use keeps the original, each patched use gets its own.
// CHECK-LABEL: aie.device(xcve3858)
// CHECK-DAG: memref.global "private" constant @shared : memref<2xi32> = dense<[1, 0]>
// CHECK-DAG: memref.global "private" constant @[[C1:shared_[0-9]+]] : memref<2xi32> = dense<[1, 8]>
// CHECK-DAG: memref.global "private" constant @[[C2:shared_[0-9]+]] : memref<2xi32> = dense<[1, 16]>
// CHECK-DAG: aiex.cert.uc_dma_bd @shared, 36864, 2, false
// CHECK-DAG: aiex.cert.uc_dma_bd @[[C1]], 36912, 2, false
// CHECK-DAG: aiex.cert.uc_dma_bd @[[C2]], 36960, 2, false
// CHECK: aiex.cert.apply_offset_57(@[[C1]], 1, 0)
// CHECK: aiex.cert.apply_offset_57(@[[C2]], 1, 1)
module {
  aie.device(xcve3858) {
    memref.global "private" constant @shared : memref<2xi32> = dense<[1, 0]>
    aie.runtime_sequence @seq(%a: memref<64xi32>, %b: memref<64xi32>) {
      %0 = memref.get_global @shared : memref<2xi32>
      aiex.npu.blockwrite(%0) {address = 36864 : ui32} : memref<2xi32>
      aiex.npu.blockwrite(%0) {address = 36912 : ui32} : memref<2xi32>
      %c8 = arith.constant 8 : i32
      aiex.npu.address_patch(%c8 : i32) {addr = 36916 : ui32, arg_idx = 0 : i32}
      aiex.npu.blockwrite(%0) {address = 36960 : ui32} : memref<2xi32>
      %c16 = arith.constant 16 : i32
      aiex.npu.address_patch(%c16 : i32) {addr = 36964 : ui32, arg_idx = 1 : i32}
    }
  }
}
