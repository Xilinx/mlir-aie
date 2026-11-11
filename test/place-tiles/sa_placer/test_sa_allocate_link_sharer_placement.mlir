//===- test_sa_allocate_link_sharer_placement.mlir -------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --aie-place-tiles="placer=sa_placer sa-seed=1" --aie-objectFifo-stateful-transform --aie-assign-buffer-addresses %s 2>&1 | FileCheck %s
// RUN: aie-opt --split-input-file --aie-place-tiles="placer=sa_placer sa-seed=2" --aie-objectFifo-stateful-transform --aie-assign-buffer-addresses %s 2>&1 | FileCheck %s
// RUN: aie-opt --split-input-file --aie-place-tiles="placer=sa_placer sa-seed=3" --aie-objectFifo-stateful-transform --aie-assign-buffer-addresses %s 2>&1 | FileCheck %s

// @out shares the link's pool on (0, 2) through its allocate only where its
// consumer core reaches (0, 2)'s memory; elsewhere it falls back to DMA and
// keeps its four objects on its consumer's tile. The SA plans the pool's
// depth where the core sits, as the split decides it.

// (0, 2) has room for two of @out's 8 KiB objects, not four, so the core sits
// where @out falls back to DMA and the pool keeps @in0's two.
// CHECK-NOT: warning
// CHECK-LABEL: module @pool_full
// CHECK: aie.lock({{.*}}) {init = 2 : i32, sym_name = "in0__cons__prodlock0"}
module @pool_full {
  aie.device(npu2_1col) {
    %s = aie.tile(0, 0)
    %l = aie.tile(0, 2)
    %c = aie.logical_tile<CoreTile>(?, ?)
    %big = aie.buffer(%l) {sym_name = "big"} : memref<10240xi32>
    aie.objectfifo @in0(%s, {%l}, 2 : i32) : !aie.objectfifo<memref<2048xi32>>
    aie.objectfifo @out(%l, {%c}, [2, 4]) : !aie.objectfifo<memref<2048xi32>>
    aie.objectfifo.link [@in0] -> [@out] ([] [])
    aie.objectfifo.allocate @out (%l)
    aie.core(%c) {
      %0, %1, %2 = aie.objectfifo.acquire @out(Consume, 3) : memref<2048xi32>, memref<2048xi32>, memref<2048xi32>
      aie.objectfifo.release @out(Consume, 3)
      aie.end
    }
  }
}

// -----

// The core's own memory is full, so it sits beside (0, 2) and @out's objects
// are the pool's four.
// CHECK-LABEL: module @consumer_full
// CHECK: aie.lock({{.*}}) {init = 4 : i32, sym_name = "in0__cons__prodlock0"}
// CHECK-NOT: warning
module @consumer_full {
  aie.device(npu2_1col) {
    %s = aie.tile(0, 0)
    %l = aie.tile(0, 2)
    %c = aie.logical_tile<CoreTile>(?, ?)
    %big = aie.buffer(%c) {sym_name = "big"} : memref<10240xi32>
    aie.objectfifo @in0(%s, {%l}, 2 : i32) : !aie.objectfifo<memref<2048xi32>>
    aie.objectfifo @out(%l, {%c}, [2, 4]) : !aie.objectfifo<memref<2048xi32>>
    aie.objectfifo.link [@in0] -> [@out] ([] [])
    aie.objectfifo.allocate @out (%l)
    aie.core(%c) {
      %0, %1, %2 = aie.objectfifo.acquire @out(Consume, 3) : memref<2048xi32>, memref<2048xi32>, memref<2048xi32>
      aie.objectfifo.release @out(Consume, 3)
      aie.end
    }
  }
}
