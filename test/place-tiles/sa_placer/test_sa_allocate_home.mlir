//===- test_sa_allocate_home.mlir ------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-place-tiles="placer=sa_placer sa-seed=1" --aie-objectFifo-stateful-transform --aie-assign-buffer-addresses %s | FileCheck %s
// RUN: aie-opt --aie-place-tiles="placer=sa_placer sa-seed=2" --aie-objectFifo-stateful-transform --aie-assign-buffer-addresses %s | FileCheck %s
// RUN: aie-opt --aie-place-tiles="placer=sa_placer sa-seed=3" --aie-objectFifo-stateful-transform --aie-assign-buffer-addresses %s | FileCheck %s

// A linked fifo's allocated objects are the link's pool on the link tile, so
// A on (0,2) must reach the link tile's memory as well as the delegate's: only
// (0,3) does.

// CHECK:     %[[L:.*]] = aie.tile(0, 3)
// CHECK-DAG: aie.buffer(%[[L]]) {{.*}}sym_name = "in0__cons__buf0"} : memref<1024xi32>
// CHECK-DAG: aie.buffer(%[[L]]) {{.*}}sym_name = "in0__cons__buf1"} : memref<1024xi32>
module @link_tile_north {
  aie.device(npu2_2col) {
    %s = aie.logical_tile<ShimNOCTile>(?, ?)
    %a = aie.tile(0, 2)
    %x4 = aie.tile(0, 4)
    %x5 = aie.tile(0, 5)
    %l = aie.logical_tile<CoreTile>(?, ?)
    aie.objectfifo @in0(%a, {%l}, 2 : i32) : !aie.objectfifo<memref<1024xi32>>
    aie.objectfifo @out(%l, {%s}, 2 : i32) : !aie.objectfifo<memref<1024xi32>>
    aie.objectfifo.link [@in0] -> [@out] ([] [])
    aie.objectfifo.allocate @in0 (%a)
    aie.core(%a) {
      %0 = aie.objectfifo.acquire @in0(Produce, 1) : memref<1024xi32>
      aie.objectfifo.release @in0(Produce, 1)
      aie.end
    }
    aie.core(%x4) { aie.end }
    aie.core(%x5) { aie.end }
  }
}
