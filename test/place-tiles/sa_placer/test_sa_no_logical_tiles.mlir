//===- test_sa_no_logical_tiles.mlir ---------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// The SA leaves a design with no logical tile as it is, pinned DMA ends and
// all.

// RUN: aie-opt --aie-place-tiles='placer=sa_placer sa-seed=1' %s | FileCheck %s

// CHECK-LABEL: module @all_physical
// CHECK: aie.objectfifo @a(%{{.*}}, {%{{.*}}}, 2 : i32)
module @all_physical {
  aie.device(npu2) {
    %s = aie.tile(0, 0)
    %c = aie.tile(0, 2)
    aie.objectfifo @a(%s, {%c}, 2 : i32) {prod_port = #aie.end_port<DMA : 0>} : !aie.objectfifo<memref<16xi32>>
  }
}
