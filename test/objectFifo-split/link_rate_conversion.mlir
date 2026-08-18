//===- link_rate_conversion.mlir ----------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --aie-objectfifo-split --aie-objectfifo-verify %s | FileCheck %s

// A one-to-one link uses one synchronization segment for the pooled object.
// Different object sizes on its endpoints set stream transfer granularity.

module @linear {
  aie.device(npu1_1col) {
    %shim = aie.tile(0, 0)
    %mem = aie.tile(0, 1)
    %core = aie.tile(0, 2)

    aie.objectfifo @wide(%shim, {%mem}, 2 : i32)
      : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @narrow(%mem, {%core}, 2 : i32)
      : !aie.objectfifo<memref<8xi32>>
    aie.objectfifo.link [@wide] -> [@narrow] ([] [])

    %c = aie.core(%core) {
      %object = aie.objectfifo.acquire @narrow(Consume) : memref<8xi32>
      aie.objectfifo.release @narrow(Consume) [1]
      aie.end
    }
  }
}

// CHECK-LABEL: module @linear
// CHECK: aie.objectfifo.pool @wide_cons_pool(%{{.*}}) {{.*}}segments = [#aie.objectfifo_segment<offset = 0, size = 16>]{{.*}} : memref<16xi32>
// CHECK: aie.objectfifo.dma_endpoint @wide_cons_dma(%{{.*}}) fills @wide_cons_pool
// CHECK: aie.objectfifo.dma_endpoint @narrow_prod_dma(%{{.*}}) drains @wide_cons_pool

// -----

module @dimensions {
  aie.device(npu1_1col) {
    %shim = aie.tile(0, 0)
    %mem = aie.tile(0, 1)
    %core = aie.tile(0, 2)

    aie.objectfifo @wide(%shim, {%mem}, 2 : i32)
      : !aie.objectfifo<memref<160xi32>>
    aie.objectfifo @narrow(
      %mem dimensionsToStream [<size = 4, stride = 40>,
                               <size = 8, stride = 5>,
                               <size = 5, stride = 1>],
      {%core}, 2 : i32) : !aie.objectfifo<memref<40xi32>>
    aie.objectfifo.link [@wide] -> [@narrow] ([] [])

    %c = aie.core(%core) {
      %object = aie.objectfifo.acquire @narrow(Consume) : memref<40xi32>
      aie.objectfifo.release @narrow(Consume) [1]
      aie.end
    }
  }
}

// CHECK-LABEL: module @dimensions
// CHECK: aie.objectfifo.pool @wide_cons_pool(%{{.*}}) {{.*}}segments = [#aie.objectfifo_segment<offset = 0, size = 160>]{{.*}} : memref<160xi32>
// CHECK: aie.objectfifo.dma_endpoint @narrow_prod_dma(%{{.*}}) drains @wide_cons_pool
// CHECK-SAME: dimensions = #aie<bd_dim_layout_array_array
// CHECK-SAME: <size = 4, stride = 40>
// CHECK-SAME: <size = 8, stride = 5>
// CHECK-SAME: <size = 5, stride = 1>
