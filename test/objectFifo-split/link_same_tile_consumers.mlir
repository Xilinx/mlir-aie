// RUN: aie-opt --aie-objectfifo-split %s | FileCheck %s

// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Both consumer ends of @in sit on one mem tile, each read by its own link:
// the first link takes the first end, 20 deep, the second the other, 10 deep.
// The BD ids planned before the split are for those same ends.

module @same_tile_consumers {
  aie.device(npu2) {
    %m = aie.tile(0, 1)
    %t2 = aie.tile(0, 2)
    %t3 = aie.tile(0, 3)
    %t4 = aie.tile(0, 4)
    aie.objectfifo @in(%t2, {%m, %m}, [1 : i32, 20 : i32, 10 : i32]) : !aie.objectfifo<memref<64xi32>>
    aie.objectfifo @a(%m, {%t3}, 2 : i32) : !aie.objectfifo<memref<64xi32>>
    aie.objectfifo @b(%m, {%t4}, 2 : i32) : !aie.objectfifo<memref<64xi32>>
    aie.objectfifo.link [@in] -> [@a] ([] [])
    aie.objectfifo.link [@in] -> [@b] ([] [])
  }
}

// CHECK-LABEL: @same_tile_consumers
// CHECK-DAG: aie.objectfifo.pool @in__cons0__pool(%{{.*}}) {depth = 20 : i32
// CHECK-DAG: aie.objectfifo.pool @in__cons1__pool(%{{.*}}) {depth = 10 : i32
// CHECK-DAG: aie.objectfifo.dma_endpoint @in__cons0__dma(%{{.*}}) fills @in__cons0__pool
// CHECK-DAG: aie.objectfifo.dma_endpoint @in__cons1__dma(%{{.*}}) fills @in__cons1__pool
