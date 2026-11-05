//===- walk_whole_dims.mlir ------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --aie-objectfifo-split --aie-objectfifo-verify --aie-assign-packet-ids --aie-objectfifo-allocate --verify-diagnostics %s | FileCheck %s
// RUN: aie-opt --split-input-file --aie-objectFifo-stateful-transform --aie-assign-bd-ids --verify-diagnostics %s -o /dev/null

// A draining end that lays out dimensions on a contiguous pool walks a stream
// window, so a pool with one is never walked whole: two 14-deep links on one
// mem tile need one pool walked, and it is @inb's, though @ina's is as deep
// and comes first.
// CHECK-LABEL: module @other_pool
// CHECK: aie.objectfifo.pool @ina__cons__pool(%{{.*}}) {buffers = [{{[^]]*}}], depth = 14
// CHECK: aie.objectfifo.pool @inb__cons__pool(%{{.*}}) {buffers = [@inb__cons__buf0], contiguous, depth = 14
module @other_pool {
  aie.device(npu2) {
    %s0 = aie.tile(0, 0)
    %m = aie.tile(0, 1)
    %t0 = aie.tile(0, 2)
    %t1 = aie.tile(1, 2)
    aie.objectfifo @ina(%s0, {%m}, [1 : i32, 14 : i32]) : !aie.objectfifo<memref<64xi32>>
    aie.objectfifo @outa(%m dimensionsToStream [<size = 8, stride = 1>, <size = 8, stride = 8>], {%t0}, 1 : i32) : !aie.objectfifo<memref<64xi32>>
    aie.objectfifo.link [@ina] -> [@outa] ([] [])
    aie.objectfifo @inb(%s0, {%m}, [1 : i32, 14 : i32]) : !aie.objectfifo<memref<64xi32>>
    aie.objectfifo @outb(%m, {%t1}, 1 : i32) : !aie.objectfifo<memref<64xi32>>
    aie.objectfifo.link [@inb] -> [@outb] ([] [])
    aie.core(%t0) {
      %x = aie.objectfifo.acquire @outa(Consume, 1) : memref<64xi32>
      aie.objectfifo.release @outa(Consume, 1)
      aie.end
    }
    aie.core(%t1) {
      %x = aie.objectfifo.acquire @outb(Consume, 1) : memref<64xi32>
      aie.objectfifo.release @outb(Consume, 1)
      aie.end
    }
  }
}

// -----

// With no other pool to walk, the shortfall itself is reported.
module @none_left {
  // expected-remark@+1 {{could not find a spill-aware allocation}}
  aie.device(npu2) {
    %s0 = aie.tile(0, 0)
    %m = aie.tile(0, 1)
    %t0 = aie.tile(0, 2)
    aie.objectfifo @in(%s0, {%m}, [1 : i32, 30 : i32]) : !aie.objectfifo<memref<64xi32>>
    // expected-error@+1 {{needs 30 buffer descriptors, but no choice of channels fits its tile's 2 unpinned DMA ends, 60 buffer descriptors in all, into the BD ids their channels share, and no pool on the tile is left to walk whole}}
    aie.objectfifo @out(%m dimensionsToStream [<size = 8, stride = 1>, <size = 8, stride = 8>], {%t0}, 1 : i32) : !aie.objectfifo<memref<64xi32>>
    aie.objectfifo.link [@in] -> [@out] ([] [])
    aie.core(%t0) {
      %x = aie.objectfifo.acquire @out(Consume, 1) : memref<64xi32>
      aie.objectfifo.release @out(Consume, 1)
      aie.end
    }
  }
}
