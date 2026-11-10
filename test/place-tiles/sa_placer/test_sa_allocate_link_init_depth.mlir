//===- test_sa_allocate_link_init_depth.mlir -------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-place-tiles="placer=sa_placer sa-seed=1" --aie-objectFifo-stateful-transform --aie-assign-buffer-addresses %s | FileCheck %s
// RUN: aie-opt --aie-place-tiles="placer=sa_placer sa-seed=2" --aie-objectFifo-stateful-transform --aie-assign-buffer-addresses %s | FileCheck %s
// RUN: aie-opt --aie-place-tiles="placer=sa_placer sa-seed=3" --aie-objectFifo-stateful-transform --aie-assign-buffer-addresses %s | FileCheck %s

// in0 owns the link's pool and shares it through its allocate, so the pool
// takes in0's deeper producer depth (ObjectFifoLinkOp::getPoolDepth) and its
// four initial objects, which the SA's plan of the pool charges too. Before,
// the SA placed it silently and the split failed on "more 'init_values' than
// 'depth' objects".

// CHECK-DAG: aie.buffer({{.*}}) {{.*}}sym_name = "in0__cons__buf0"} : memref<16xi32> = dense<5>
// CHECK-DAG: aie.buffer({{.*}}) {{.*}}sym_name = "in0__cons__buf3"} : memref<16xi32> = dense<8>
// CHECK-DAG: aie.lock({{.*}}) {init = 4 : i32, sym_name = "in0__cons__conslock0"}
module @input_owner_deeper_producer {
  aie.device(npu2_1col) {
    %s = aie.logical_tile<ShimNOCTile>(?, ?)
    %a = aie.logical_tile<CoreTile>(?, ?)
    %l = aie.logical_tile<CoreTile>(?, ?)
    aie.objectfifo @in0(%a, {%l}, [4, 2]) : !aie.objectfifo<memref<16xi32>> = [dense<5> : memref<16xi32>, dense<6> : memref<16xi32>, dense<7> : memref<16xi32>, dense<8> : memref<16xi32>]
    aie.objectfifo @out(%l, {%s}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo.link [@in0] -> [@out] ([] [])
    aie.objectfifo.allocate @in0 (%l)
    aie.core(%a) {
      %0 = aie.objectfifo.acquire @in0(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @in0(Produce, 1)
      aie.end
    }
  }
}
