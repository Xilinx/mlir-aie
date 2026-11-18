//===- fresh_symbol_names.mlir ------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Fresh names made by -aie-npu-to-cert skip symbols that already exist in the
// device, including names the pass itself made earlier. @chain_1 is taken by a
// global, so the chains are @chain_0, @chain_2, ... in blockwrite order.
// @big_split_0 is taken, so the 16000-byte @big splits into @big_split_0_0 and
// @big_split_1, and each of those, still at the 8000-byte threshold, splits
// again.
//
// RUN: aie-opt --aie-npu-to-cert %s | FileCheck %s

// CHECK-DAG: memref.global "private" constant @chain_1 : memref<1xi32> = dense<7>
// CHECK-DAG: memref.global "private" constant @big_split_0 : memref<1xi32> = dense<8>
// CHECK-DAG: memref.global "private" constant @big_split_0_0 : memref<2000xi32>
// CHECK-DAG: memref.global "private" constant @big_split_0_0_split_0 : memref<1000xi32>
// CHECK-DAG: memref.global "private" constant @big_split_0_0_split_1 : memref<1000xi32>
// CHECK-DAG: memref.global "private" constant @big_split_1 : memref<2000xi32>
// CHECK-DAG: memref.global "private" constant @big_split_1_split_0 : memref<1000xi32>
// CHECK-DAG: memref.global "private" constant @big_split_1_split_1 : memref<1000xi32>
// CHECK:      aiex.cert.uc_dma_chain @chain_0 {
// CHECK-NEXT:   aiex.cert.uc_dma_bd @a, 4096, 1, false
// CHECK-NEXT: }
// CHECK-NEXT: aiex.cert.uc_dma_chain @chain_2 {
// CHECK-NEXT:   aiex.cert.uc_dma_bd @big_split_0_0_split_0, 8192, 1000, false
// CHECK-NEXT: }
// CHECK-NEXT: aiex.cert.uc_dma_chain @chain_3 {
// CHECK-NEXT:   aiex.cert.uc_dma_bd @big_split_0_0_split_1, 12192, 1000, false
// CHECK-NEXT: }
// CHECK-NEXT: aiex.cert.uc_dma_chain @chain_4 {
// CHECK-NEXT:   aiex.cert.uc_dma_bd @big_split_1_split_0, 16192, 1000, false
// CHECK-NEXT: }
// CHECK-NEXT: aiex.cert.uc_dma_chain @chain_5 {
// CHECK-NEXT:   aiex.cert.uc_dma_bd @big_split_1_split_1, 20192, 1000, false
// CHECK-NEXT: }
// CHECK-NEXT: aiex.cert.uc_dma_chain @chain_6 {
// CHECK-NEXT:   aiex.cert.uc_dma_bd @b, 4352, 2, false
// CHECK-NEXT: }
// CHECK:      aiex.cert.job(1) {
// CHECK-NEXT:   aiex.cert.uc_dma_write_des_sync(@chain_0)
// CHECK-NEXT:   aiex.cert.uc_dma_write_des_sync(@chain_2)
// CHECK-NEXT:   aiex.cert.uc_dma_write_des_sync(@chain_3)
// CHECK-NEXT:   aiex.cert.uc_dma_write_des_sync(@chain_4)
// CHECK-NEXT:   aiex.cert.uc_dma_write_des_sync(@chain_5)
// CHECK-NEXT:   aiex.cert.uc_dma_write_des_sync(@chain_6)
// CHECK-NEXT: } {cert.configure}

module {
  aie.device(npu2) {
    memref.global "private" constant @chain_1 : memref<1xi32> = dense<[7]>
    memref.global "private" constant @big_split_0 : memref<1xi32> = dense<[8]>
    memref.global "private" constant @a : memref<1xi32> = dense<[1]>
    memref.global "private" constant @b : memref<2xi32> = dense<[2, 3]>
    memref.global "private" constant @big : memref<4000xi32> = dense<5>
    aie.runtime_sequence @configure() {
      %a = memref.get_global @a : memref<1xi32>
      aiex.npu.blockwrite(%a) {address = 4096 : ui32} : memref<1xi32>
      %big = memref.get_global @big : memref<4000xi32>
      aiex.npu.blockwrite(%big) {address = 8192 : ui32} : memref<4000xi32>
      %b = memref.get_global @b : memref<2xi32>
      aiex.npu.blockwrite(%b) {address = 4352 : ui32} : memref<2xi32>
      aie.end
    }
  }
}
