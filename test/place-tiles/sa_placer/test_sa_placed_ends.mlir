//===- test_sa_placed_ends.mlir --------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-place-tiles="placer=sa_placer sa-seed=1" --mlir-pass-statistics %s -o %t 2> %t.err
// RUN: FileCheck %s --input-file=%t
// RUN: FileCheck %s --check-prefix=STATS --input-file=%t.err
// RUN: aie-opt --pass-pipeline='builtin.module(aie.device(aie-objectFifo-stateful-transform,aie-assign-buffer-addresses))' %t -o /dev/null

// The SA charges a fifo whose producer sits on a placed tile, and that
// tile's own buffers and stack. @p holds 50 KB, so @f's two 12 KB objects
// fit only in @l's memory: @l goes to (1, 2), where the split keeps them
// in @l's memory with no allocate.
// CHECK: aie.tile(1, 2)
// CHECK-NOT: aie.objectfifo.allocate
// STATS: 1 sa-lock-groups
// STATS-NEXT: 0 sa-lock-penalty

module @placed_producer {
  aie.device(npu2) {
    %p = aie.tile(2, 2)
    %l = aie.logical_tile<CoreTile>(?, ?)
    %big = aie.buffer(%p) {sym_name = "big"} : memref<12800xi32>
    aie.objectfifo @f(%p, {%l}, 2 : i32) : !aie.objectfifo<memref<3072xi32>>
    aie.core(%p) {
      %0 = aie.objectfifo.acquire @f(Produce, 1) : memref<3072xi32>
      aie.objectfifo.release @f(Produce, 1)
      aie.end
    }
    aie.core(%l) {
      %0 = aie.objectfifo.acquire @f(Consume, 1) : memref<3072xi32>
      aie.objectfifo.release @f(Consume, 1)
      aie.end
    }
  }
}
