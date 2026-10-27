// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt --aie-npu-to-cert --split-input-file %s | FileCheck %s

// Generated DMA-chain symbols must avoid buffer declarations, not just other
// chains or memref globals. The standard symbol verifier checks the result.
// CHECK-LABEL: aie.device(npu2) {
// CHECK-DAG: aie.buffer @chain_0(
// CHECK-DAG: aiex.cert.uc_dma_chain @chain_1 {
// CHECK: aiex.cert.uc_dma_write_des_sync(@chain_1)
aie.device(npu2) @main {
  %tile = aie.tile(0, 2)
  aie.buffer @chain_0(%tile) : memref<4xi32>
  memref.global "private" constant @data : memref<1xi32> = dense<[7]>
  aie.runtime_sequence @configure() {
    %data = memref.get_global @data : memref<1xi32>
    aiex.npu.blockwrite(%data) {address = 4096 : ui32} : memref<1xi32>
  }
}

// -----

// Locks share that namespace too.
// CHECK-LABEL: aie.device(npu2) {
// CHECK-DAG: aie.lock @chain_0(
// CHECK-DAG: aiex.cert.uc_dma_chain @chain_1 {
// CHECK: aiex.cert.uc_dma_write_des_sync(@chain_1)
aie.device(npu2) @main {
  %tile = aie.tile(0, 2)
  aie.lock @chain_0(%tile, 0)
  memref.global "private" constant @data : memref<1xi32> = dense<[7]>
  aie.runtime_sequence @configure() {
    %data = memref.get_global @data : memref<1xi32>
    aiex.npu.blockwrite(%data) {address = 4096 : ui32} : memref<1xi32>
  }
}

// -----

// Repeated lookup must skip occupied names of different declaration kinds.
// CHECK-LABEL: aie.device(npu2) {
// CHECK-DAG: aie.buffer @chain_0(
// CHECK-DAG: aie.lock @chain_1(
// CHECK-DAG: aiex.cert.uc_dma_chain @chain_2 {
// CHECK: aiex.cert.uc_dma_write_des_sync(@chain_2)
aie.device(npu2) @main {
  %tile = aie.tile(0, 2)
  aie.buffer @chain_0(%tile) : memref<4xi32>
  aie.lock @chain_1(%tile, 0)
  memref.global "private" constant @data : memref<1xi32> = dense<[7]>
  aie.runtime_sequence @configure() {
    %data = memref.get_global @data : memref<1xi32>
    aiex.npu.blockwrite(%data) {address = 4096 : ui32} : memref<1xi32>
  }
}
