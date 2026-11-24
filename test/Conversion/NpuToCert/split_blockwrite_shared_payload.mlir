//===- split_blockwrite_shared_payload.mlir -----------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Two blockwrites split the same oversized payload. The second split has to
// see the globals the first one created, so it takes the _0-suffixed names,
// and the chains skip @chain_2, which a global already holds. The assembly
// resolves every chain and payload the pass created.
//
// RUN: aie-opt --aie-npu-to-cert %s | FileCheck %s
// RUN: aie-opt --aie-npu-to-cert %s | aie-translate --aie-cert-to-asm | FileCheck %s --check-prefix=ASM

// CHECK-DAG: memref.global "private" constant @big_split_0 : memref<1500xi32>
// CHECK-DAG: memref.global "private" constant @big_split_1 : memref<1500xi32>
// CHECK-DAG: memref.global "private" constant @big_split_0_0 : memref<1500xi32>
// CHECK-DAG: memref.global "private" constant @big_split_1_0 : memref<1500xi32>
// CHECK-DAG: memref.global "private" constant @chain_2 : memref<1xi32> = dense<7>
// CHECK:      aiex.cert.uc_dma_chain @chain_0 {
// CHECK-NEXT:   aiex.cert.uc_dma_bd @big_split_0_0, 8192, 1500, false
// CHECK:      aiex.cert.uc_dma_chain @chain_1 {
// CHECK-NEXT:   aiex.cert.uc_dma_bd @big_split_1_0, 14192, 1500, false
// CHECK:      aiex.cert.uc_dma_chain @chain_3 {
// CHECK-NEXT:   aiex.cert.uc_dma_bd @big_split_0, 65536, 1500, false
// CHECK:      aiex.cert.uc_dma_chain @chain_4 {
// CHECK-NEXT:   aiex.cert.uc_dma_bd @big_split_1, 71536, 1500, false
// CHECK:      aiex.cert.job(1) {
// CHECK-NEXT:   aiex.cert.uc_dma_write_des_sync(@chain_0)
// CHECK-NEXT:   aiex.cert.uc_dma_write_des_sync(@chain_1)
// CHECK-NEXT:   aiex.cert.uc_dma_write_des_sync(@chain_3)
// CHECK-NEXT:   aiex.cert.uc_dma_write_des_sync(@chain_4)
// CHECK-NEXT: } {cert.configure}

// ASM-LABEL: chain_0:
// ASM-NEXT:    UC_DMA_BD       0, 0x00002000, @big_split_0_0, 1500, 0, 0
// ASM-LABEL: chain_4:
// ASM-NEXT:    UC_DMA_BD       0, 0x00011770, @big_split_1, 1500, 0, 0
// ASM-DAG:   {{^}}big_split_0_0:
// ASM-DAG:   {{^}}big_split_1:

module {
  aie.device(npu2) {
    memref.global "private" constant @big : memref<3000xi32> = dense<5>
    memref.global "private" constant @chain_2 : memref<1xi32> = dense<[7]>
    aie.runtime_sequence @configure() {
      %a = memref.get_global @big : memref<3000xi32>
      aiex.npu.blockwrite(%a) {address = 8192 : ui32} : memref<3000xi32>
      %b = memref.get_global @big : memref<3000xi32>
      aiex.npu.blockwrite(%b) {address = 65536 : ui32} : memref<3000xi32>
      aie.end
    }
  }
}
