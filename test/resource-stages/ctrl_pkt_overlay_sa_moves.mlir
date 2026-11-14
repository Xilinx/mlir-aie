//===- ctrl_pkt_overlay_sa_moves.mlir --------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Three logical shims on two shim tiles leave none empty, so the SA reaches
// the one placement that fits (an input alone where the control packet goes)
// only by moving a shim onto an occupied tile, not just by swapping two.

// RUN: aie-opt --verify-diagnostics --pass-pipeline='builtin.module(aie-resource-stages{generate-ctrl-pkt-overlay=true placer=sa_placer sa-seed=1})' %s | FileCheck %s
// RUN: aie-opt --verify-diagnostics --pass-pipeline='builtin.module(aie-resource-stages{generate-ctrl-pkt-overlay=true placer=sa_placer sa-seed=2})' %s | FileCheck %s
// RUN: aie-opt --verify-diagnostics --pass-pipeline='builtin.module(aie-resource-stages{generate-ctrl-pkt-overlay=true placer=sa_placer sa-seed=3})' %s | FileCheck %s
// RUN: aie-opt --verify-diagnostics --pass-pipeline='builtin.module(aie-resource-stages{generate-ctrl-pkt-overlay=true placer=sa_placer sa-seed=4})' %s | FileCheck %s
// RUN: aie-opt --verify-diagnostics --pass-pipeline='builtin.module(aie-resource-stages{generate-ctrl-pkt-overlay=true placer=sa_placer sa-seed=5})' %s | FileCheck %s
// RUN: aie-opt --verify-diagnostics --pass-pipeline='builtin.module(aie-resource-stages{generate-ctrl-pkt-overlay=true placer=sa_placer sa-seed=6})' %s | FileCheck %s
// RUN: aie-opt --verify-diagnostics --pass-pipeline='builtin.module(aie-resource-stages{generate-ctrl-pkt-overlay=true placer=sa_placer sa-seed=7})' %s | FileCheck %s
// RUN: aie-opt --verify-diagnostics --pass-pipeline='builtin.module(aie-resource-stages{generate-ctrl-pkt-overlay=true placer=sa_placer sa-seed=8})' %s | FileCheck %s

// CHECK-LABEL: module @overlay_two_columns
// CHECK-DAG: aie.shim_dma_allocation @ctrlpkt_col0_mm2s_chan0(
// CHECK-DAG: aie.shim_dma_allocation @ctrlpkt_col1_mm2s_chan0(
// CHECK-DAG: aie.shim_dma_allocation @a__shim_alloc(
// CHECK-DAG: aie.shim_dma_allocation @b__shim_alloc(
// CHECK-DAG: aie.shim_dma_allocation @d__shim_alloc(
module @overlay_two_columns {
  aie.device(npu2_2col) {
    %s0 = aie.logical_tile<ShimNOCTile>(?, ?)
    %s1 = aie.logical_tile<ShimNOCTile>(?, ?)
    %s2 = aie.logical_tile<ShimNOCTile>(?, ?)
    %c0 = aie.logical_tile<CoreTile>(?, ?)
    %c1 = aie.logical_tile<CoreTile>(?, ?)
    aie.objectfifo @a(%s0, {%c0}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @b(%s1, {%c0}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @d(%s2, {%c1}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.core(%c0) {
      %a = aie.objectfifo.acquire @a(Consume, 1) : memref<16xi32>
      %b = aie.objectfifo.acquire @b(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @a(Consume, 1)
      aie.objectfifo.release @b(Consume, 1)
      aie.end
    }
    aie.core(%c1) {
      %d = aie.objectfifo.acquire @d(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @d(Consume, 1)
      aie.end
    }
    aie.runtime_sequence @run(%input : memref<16xi32>) {
      aiex.npu.dma_memcpy_nd(%input[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0, 1]) {metadata = @a, id = 0 : i64} : memref<16xi32>
      aiex.npu.dma_memcpy_nd(%input[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0, 1]) {metadata = @b, id = 1 : i64} : memref<16xi32>
      aiex.npu.dma_memcpy_nd(%input[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0, 1]) {metadata = @d, id = 2 : i64} : memref<16xi32>
      aiex.control_packet {address = 126976 : ui32, data = array<i32: 0>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.npu.dma_memcpy_nd(%input[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0, 1]) {metadata = @a, id = 0 : i64} : memref<16xi32>
      aiex.npu.dma_memcpy_nd(%input[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0, 1]) {metadata = @b, id = 1 : i64} : memref<16xi32>
      aiex.npu.dma_memcpy_nd(%input[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0, 1]) {metadata = @d, id = 2 : i64} : memref<16xi32>
    }
  }
}
