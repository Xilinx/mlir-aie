//===- test_sa_shared_mem_transport.mlir -------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// A core passes rows over shared memory to four others: SA places all four
// on tiles sharing a memory module with it, so split moves none by DMA.

// RUN: aie-opt --aie-place-tiles='placer=sa_placer sa-seed=7' --aie-objectfifo-split %s | FileCheck %s

// CHECK-LABEL: @shared_mem_fan_out
// CHECK-NOT: dma_endpoint @near{{[0-9]}}

module @shared_mem_fan_out {
  aie.device(npu2) {
    %t0 = aie.logical_tile<CoreTile>(?, ?)
    %t1 = aie.logical_tile<CoreTile>(?, ?)
    %t2 = aie.logical_tile<CoreTile>(?, ?)
    %t3 = aie.logical_tile<CoreTile>(?, ?)
    %t4 = aie.logical_tile<CoreTile>(?, ?)
    aie.objectfifo @near1(%t0, {%t1}, 2 : i32) {transport = #aie.transport<shared_mem>} : !aie.objectfifo<memref<8xi32>>
    aie.objectfifo @near2(%t0, {%t2}, 2 : i32) {transport = #aie.transport<shared_mem>} : !aie.objectfifo<memref<8xi32>>
    aie.objectfifo @near3(%t0, {%t3}, 2 : i32) {transport = #aie.transport<shared_mem>} : !aie.objectfifo<memref<8xi32>>
    aie.objectfifo @near4(%t0, {%t4}, 2 : i32) {transport = #aie.transport<shared_mem>} : !aie.objectfifo<memref<8xi32>>
    aie.core(%t0) { aie.end }
    aie.core(%t1) { aie.end }
    aie.core(%t2) { aie.end }
    aie.core(%t3) { aie.end }
    aie.core(%t4) { aie.end }
  }
}
