//===- test_sa_allocate_link_misfit.mlir -----------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --aie-place-tiles="placer=sa_placer sa-seed=1" --aie-objectFifo-stateful-transform --aie-assign-buffer-addresses %s | FileCheck %s
// RUN: aie-opt --split-input-file --aie-place-tiles="placer=sa_placer sa-seed=2" --aie-objectFifo-stateful-transform --aie-assign-buffer-addresses %s | FileCheck %s
// RUN: aie-opt --split-input-file --aie-place-tiles="placer=sa_placer sa-seed=3" --aie-objectFifo-stateful-transform --aie-assign-buffer-addresses %s | FileCheck %s

// A merge's input or a dispatch's output cannot share the link's pool, so the
// SA places its ends where the allocate is not honoured, and the split moves it
// by DMA with objects of its own, as it does any link participant that does not
// own the pool: the link tile goes to (0,4), not to (0,3), which is nearer but
// where A on (0,2) reaches its memory.

// CHECK-LABEL: module @merge_alloc
// CHECK-DAG: %[[A:.*]] = aie.tile(0, 2)
// CHECK-DAG: %[[L:.*]] = aie.tile(0, 4)
// CHECK-DAG: aie.buffer(%[[A]]) {{.*}}sym_name = "in0__buf0"} : memref<1024xi32>
module @merge_alloc {
  aie.device(npu2_1col) {
    %s = aie.logical_tile<ShimNOCTile>(?, ?)
    %l = aie.logical_tile<CoreTile>(?, ?)
    %a = aie.tile(0, 2)
    %b = aie.tile(0, 5)
    aie.objectfifo @in0(%a, {%l}, 2 : i32) : !aie.objectfifo<memref<1024xi32>>
    aie.objectfifo @in1(%b, {%l}, 2 : i32) : !aie.objectfifo<memref<1024xi32>>
    aie.objectfifo @out(%l, {%s}, 2 : i32) : !aie.objectfifo<memref<1024xi32>>
    aie.objectfifo.link [@in0, @in1] -> [@out] ([] []) {mode = #aie.link_mode<time>}
    aie.objectfifo.allocate @in0 (%l)
    aie.core(%a) {
      %0 = aie.objectfifo.acquire @in0(Produce, 1) : memref<1024xi32>
      aie.objectfifo.release @in0(Produce, 1)
      aie.end
    }
    aie.core(%b) {
      %0 = aie.objectfifo.acquire @in1(Produce, 1) : memref<1024xi32>
      aie.objectfifo.release @in1(Produce, 1)
      aie.end
    }
  }
}

// -----

// CHECK-LABEL: module @dispatch_alloc
// CHECK-DAG: %[[A:.*]] = aie.tile(0, 2)
// CHECK-DAG: %[[L:.*]] = aie.tile(0, 4)
// CHECK-DAG: aie.buffer(%[[A]]) {{.*}}sym_name = "out0__cons__buf0"} : memref<1024xi32>
module @dispatch_alloc {
  aie.device(npu2_1col) {
    %s = aie.logical_tile<ShimNOCTile>(?, ?)
    %l = aie.logical_tile<CoreTile>(?, ?)
    %a = aie.tile(0, 2)
    %b = aie.tile(0, 5)
    aie.objectfifo @in(%s, {%l}, 2 : i32) : !aie.objectfifo<memref<1024xi32>>
    aie.objectfifo @out0(%l, {%a}, 2 : i32) : !aie.objectfifo<memref<1024xi32>>
    aie.objectfifo @out1(%l, {%b}, 2 : i32) : !aie.objectfifo<memref<1024xi32>>
    aie.objectfifo.link [@in] -> [@out0, @out1] ([] []) {mode = #aie.link_mode<time>}
    aie.objectfifo.allocate @out0 (%l)
    aie.core(%a) {
      %0 = aie.objectfifo.acquire @out0(Consume, 1) : memref<1024xi32>
      aie.objectfifo.release @out0(Consume, 1)
      aie.end
    }
    aie.core(%b) {
      %0 = aie.objectfifo.acquire @out1(Consume, 1) : memref<1024xi32>
      aie.objectfifo.release @out1(Consume, 1)
      aie.end
    }
  }
}
