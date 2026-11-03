// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt --aie-npu-to-cert -split-input-file %s | FileCheck %s --implicit-check-not=aiex.npu.blockwrite
// RUN: aie-opt --aie-npu-to-cert -split-input-file %s | aie-translate --aie-cert-to-asm -split-input-file | FileCheck %s --check-prefix=ASM

// An empty copy creates no chain, descriptor, or DMA submission.
// CHECK-LABEL: aie.device(xcve3858)
// CHECK-NOT: aiex.cert.uc_dma
// CHECK: aiex.cert.job(1)
// CHECK-NOT: aiex.cert.uc_dma
// ASM: START_JOB 1
// ASM-NOT: uC_DMA_WRITE_DES_SYNC
// ASM: END_JOB
// ASM-NOT: UC_DMA_BD
aie.device(xcve3858) {
  memref.global "private" constant @empty : memref<0xi32> = dense<>
  aie.runtime_sequence @configure() {
    %empty = memref.get_global @empty : memref<0xi32>
    aiex.npu.blockwrite(%empty) {address = 4096 : ui32} : memref<0xi32>
  }
}

// -----

// Empty copies before, between, and after real copies consume no chain names
// and do not reorder the real transfers.
// CHECK-LABEL: aie.device(xcve3858)
// CHECK: aiex.cert.uc_dma_chain @chain_0 {
// CHECK-NEXT: aiex.cert.uc_dma_bd @first, 4096, 1, false
// CHECK: aiex.cert.uc_dma_chain @chain_1 {
// CHECK-NEXT: aiex.cert.uc_dma_bd @second, 8192, 2, false
// CHECK-NOT: aiex.cert.uc_dma_chain
// CHECK: aiex.cert.job(1) {
// CHECK-NEXT: aiex.cert.uc_dma_write_des_sync(@chain_0)
// CHECK-NEXT: aiex.cert.uc_dma_write_des_sync(@chain_1)
// CHECK-NOT: aiex.cert.uc_dma
// ASM: START_JOB 1
// ASM-NEXT: uC_DMA_WRITE_DES_SYNC @chain_0
// ASM-NEXT: uC_DMA_WRITE_DES_SYNC @chain_1
// ASM-NEXT: END_JOB
// ASM: UC_DMA_BD 0, 0x00001000, @first, 1, 0, 0
// ASM: UC_DMA_BD 0, 0x00002000, @second, 2, 0, 0
// ASM-NOT: UC_DMA_BD
aie.device(xcve3858) {
  memref.global "private" constant @empty : memref<0xi32> = dense<>
  memref.global "private" constant @first : memref<1xi32> = dense<7>
  memref.global "private" constant @second : memref<2xi32> = dense<[8, 9]>
  aie.runtime_sequence @configure() {
    %empty = memref.get_global @empty : memref<0xi32>
    %first = memref.get_global @first : memref<1xi32>
    %second = memref.get_global @second : memref<2xi32>
    aiex.npu.blockwrite(%empty) {address = 0 : ui32} : memref<0xi32>
    aiex.npu.blockwrite(%first) {address = 4096 : ui32} : memref<1xi32>
    aiex.npu.blockwrite(%empty) {address = 0 : ui32} : memref<0xi32>
    aiex.npu.blockwrite(%second) {address = 8192 : ui32} : memref<2xi32>
    aiex.npu.blockwrite(%empty) {address = 0 : ui32} : memref<0xi32>
  }
}

// -----

// Generated names skip an existing chain name and keep advancing afterwards.
// CHECK-LABEL: aie.device(xcve3858)
// CHECK: aiex.cert.uc_dma_chain @chain_1 {
// CHECK-NEXT: aiex.cert.uc_dma_bd @user, 0, 1, false
// CHECK: aiex.cert.uc_dma_chain @chain_0 {
// CHECK-NEXT: aiex.cert.uc_dma_bd @a, 4096, 1, false
// CHECK: aiex.cert.uc_dma_chain @chain_2 {
// CHECK-NEXT: aiex.cert.uc_dma_bd @b, 8192, 1, false
// CHECK: aiex.cert.uc_dma_chain @chain_3 {
// CHECK-NEXT: aiex.cert.uc_dma_bd @c, 12288, 1, false
// CHECK: aiex.cert.job(1) {
// CHECK-NEXT: aiex.cert.uc_dma_write_des_sync(@chain_0)
// CHECK-NEXT: aiex.cert.uc_dma_write_des_sync(@chain_2)
// CHECK-NEXT: aiex.cert.uc_dma_write_des_sync(@chain_3)
// ASM: START_JOB 1
// ASM-NEXT: uC_DMA_WRITE_DES_SYNC @chain_0
// ASM-NEXT: uC_DMA_WRITE_DES_SYNC @chain_2
// ASM-NEXT: uC_DMA_WRITE_DES_SYNC @chain_3
aie.device(xcve3858) {
  memref.global "private" constant @user : memref<1xi32> = dense<1>
  memref.global "private" constant @a : memref<1xi32> = dense<2>
  memref.global "private" constant @b : memref<1xi32> = dense<3>
  memref.global "private" constant @c : memref<1xi32> = dense<4>
  aiex.cert.uc_dma_chain @chain_1 {
    aiex.cert.uc_dma_bd @user, 0, 1, false
  }
  aie.runtime_sequence @configure() {
    %a = memref.get_global @a : memref<1xi32>
    %b = memref.get_global @b : memref<1xi32>
    %c = memref.get_global @c : memref<1xi32>
    aiex.npu.blockwrite(%a) {address = 4096 : ui32} : memref<1xi32>
    aiex.npu.blockwrite(%b) {address = 8192 : ui32} : memref<1xi32>
    aiex.npu.blockwrite(%c) {address = 12288 : ui32} : memref<1xi32>
  }
}
