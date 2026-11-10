//===- test_sa_allocate_link_pool_depth.mlir -------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --aie-place-tiles="placer=sa_placer sa-seed=1" --aie-objectFifo-stateful-transform --aie-assign-buffer-addresses %s | FileCheck %s
// RUN: aie-opt --split-input-file --aie-place-tiles="placer=sa_placer sa-seed=2" --aie-objectFifo-stateful-transform --aie-assign-buffer-addresses %s | FileCheck %s
// RUN: aie-opt --split-input-file --aie-place-tiles="placer=sa_placer sa-seed=3" --aie-objectFifo-stateful-transform --aie-assign-buffer-addresses %s | FileCheck %s

// The SA plans a shared link pool at the depth the split gives it
// (ObjectFifoLinkOp::getPoolDepth), and the placement lowers: two objects for
// in0's dispatch to four outputs, where its declared five would take 20 BDs.
// CHECK-LABEL: module @dispatch_declared_5
// CHECK: aie.lock({{.*}}) {init = 2 : i32, sym_name = "in0__cons__prodlock0"}
// CHECK-NOT: in0__cons__buf2
module @dispatch_declared_5 {
  aie.device(npu2) {
    %s0 = aie.logical_tile<ShimNOCTile>(?, ?)
    %s1 = aie.logical_tile<ShimNOCTile>(?, ?)
    %s2 = aie.logical_tile<ShimNOCTile>(?, ?)
    %s3 = aie.logical_tile<ShimNOCTile>(?, ?)
    %a = aie.logical_tile<CoreTile>(?, ?)
    %l = aie.logical_tile<CoreTile>(?, ?)
    aie.objectfifo @in0(%a, {%l}, [5, 2]) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @o0(%l, {%s0}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @o1(%l, {%s1}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @o2(%l, {%s2}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @o3(%l, {%s3}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo.link [@in0] -> [@o0, @o1, @o2, @o3] ([] []) {mode = #aie.link_mode<time>}
    aie.objectfifo.allocate @in0 (%l)
    aie.core(%a) {
      %0 = aie.objectfifo.acquire @in0(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @in0(Produce, 1)
      aie.end
    }
  }
}

// -----

// out shares in0's pool through its own allocate and its core acquires three,
// so the pool takes four.
// CHECK-LABEL: module @sharer_deeper
// CHECK: aie.lock({{.*}}) {init = 4 : i32, sym_name = "in0__cons__prodlock0"}
module @sharer_deeper {
  aie.device(npu2_1col) {
    %s = aie.logical_tile<ShimNOCTile>(?, ?)
    %l = aie.logical_tile<CoreTile>(?, ?)
    %c = aie.logical_tile<CoreTile>(?, ?)
    aie.objectfifo @in0(%s, {%l}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @out(%l, {%c}, [2, 4]) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo.link [@in0] -> [@out] ([] [])
    aie.objectfifo.allocate @out (%l)
    aie.core(%c) {
      %0, %1, %2 = aie.objectfifo.acquire @out(Consume, 3) : memref<16xi32>, memref<16xi32>, memref<16xi32>
      aie.objectfifo.release @out(Consume, 3)
      aie.end
    }
  }
}
