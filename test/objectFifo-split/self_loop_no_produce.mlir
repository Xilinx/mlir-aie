//===- self_loop_no_produce.mlir ----------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectfifo-split %s | FileCheck %s

// A self-loop whose core never acquires Produce (only Consume): the
// producing end finds no Produce acquire to size from and falls back to the
// declared depth, 6. The consuming end holds the largest Consume acquire plus
// one, 2. Nothing fills this pool, so the stateful transform refuses it; this
// test exercises the split alone.
// CHECK: aie.objectfifo.pool @f__pool(%{{.*}}) {depth = 6 : i32
// CHECK: aie.objectfifo.pool @f__cons__pool(%{{.*}}) {depth = 2 : i32
module @self_loop_dma_no_produce {
  aie.device(npu2) {
    %p = aie.tile(1, 3)
    aie.objectfifo @f(%p, {%p}, 6 : i32) {transport = #aie.transport<dma>} : !aie.objectfifo<memref<16xi32>>
    aie.core(%p) {
      %b0 = aie.objectfifo.acquire @f(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f(Consume, 1)
      aie.end
    }
  }
}
