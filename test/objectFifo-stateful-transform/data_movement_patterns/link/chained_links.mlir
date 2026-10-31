//===- chained_links.mlir ---------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform %s | FileCheck %s

// A fifo out of one link may go into the next: an 80-row map waits in two
// mem tiles, 40 rows each, each pool walked whole by one descriptor a side.
// CHECK-LABEL: @chained
// CHECK-DAG: aie.flow(%shim_noc_tile_0_0, DMA : 0, %mem_tile_0_1, DMA : 0)
// CHECK-DAG: aie.flow(%mem_tile_0_1, DMA : 0, %mem_tile_1_1, DMA : 0)
// CHECK-DAG: aie.flow(%mem_tile_1_1, DMA : 0, %tile_0_2, DMA : 0)
// CHECK-DAG: aie.buffer(%mem_tile_0_1) {sym_name = "in__cons__buf0"} : memref<2560xi32>
// CHECK-DAG: aie.buffer(%mem_tile_1_1) {sym_name = "mid__cons__buf0"} : memref<2560xi32>
module @chained {
  aie.device(npu2) {
    %s = aie.tile(0, 0)
    %m0 = aie.tile(0, 1)
    %m1 = aie.tile(1, 1)
    %t = aie.tile(0, 2)
    aie.objectfifo @in(%s, {%m0}, [2 : i32, 40 : i32]) : !aie.objectfifo<memref<64xi32>>
    aie.objectfifo @mid(%m0, {%m1}, [2 : i32, 40 : i32]) : !aie.objectfifo<memref<64xi32>>
    aie.objectfifo.link [@in] -> [@mid] ([] [])
    aie.objectfifo @held(%m1, {%t}, 2 : i32) : !aie.objectfifo<memref<64xi32>>
    aie.objectfifo.link [@mid] -> [@held] ([] [])
    aie.core(%t) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c80 = arith.constant 80 : index
      scf.for %i = %c0 to %c80 step %c1 {
        %x = aie.objectfifo.acquire @held(Consume, 1) : memref<64xi32>
        aie.objectfifo.release @held(Consume, 1)
      }
      aie.end
    }
  }
}
