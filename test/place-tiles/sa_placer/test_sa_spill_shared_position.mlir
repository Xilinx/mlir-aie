//===- test_sa_spill_shared_position.mlir ----------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-place-tiles="placer=sa_placer sa-seed=1" %s 2>&1 | FileCheck %s

// Two mem tile LTOs pinned to (0, 1) hold 2 KB and 320 KB, which fit its
// 512 KB. The SA weighs each pool by its own objects, whichever LTO it is on.
// CHECK-NOT: warning: SA placer's resource estimate is over capacity
// CHECK-LABEL: @shared_position
// CHECK: aie.objectfifo @a_in(%{{.*}}, {%mem_tile_0_1}
// CHECK: aie.objectfifo @b_in(%{{.*}}, {%mem_tile_0_1}

module @shared_position {
  aie.device(npu2_1col) {
    %core = aie.logical_tile<CoreTile>(?, ?)
    %ma = aie.logical_tile<MemTile>(0, 1)
    %mb = aie.logical_tile<MemTile>(0, 1)
    %shim = aie.logical_tile<ShimNOCTile>(?, ?)
    aie.objectfifo @a_in(%shim, {%ma}, 2 : i32) : !aie.objectfifo<memref<256xi32>>
    aie.objectfifo @a_out(%ma, {%core}, 2 : i32) : !aie.objectfifo<memref<256xi32>>
    aie.objectfifo.link [@a_in] -> [@a_out] ([] [])
    aie.objectfifo @b_in(%shim, {%mb}, 2 : i32) : !aie.objectfifo<memref<40000xi32>>
    aie.objectfifo @b_out(%mb, {%shim}, 2 : i32) : !aie.objectfifo<memref<40000xi32>>
    aie.objectfifo.link [@b_in] -> [@b_out] ([] [])
    aie.core(%core) {
      %0 = aie.objectfifo.acquire @a_out(Consume, 1) : memref<256xi32>
      aie.objectfifo.release @a_out(Consume, 1)
      aie.end
    }
  }
}
