//===- test_sa_walk_channel_short.mlir -------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-place-tiles="placer=sa_placer sa-seed=1" %s 2>&1 | FileCheck %s

// Seven ends each way at (0, 1) need more channels than its six, which no
// walk gives, so no pool is walked: the SA charges the channels alone (12
// over, 20 each), not memory for 32 KB walked buffers that do not fit where
// the split objects do.
// CHECK: warning: SA placer's resource estimate is over capacity (penalty=240)

module @seven_ends {
  aie.device(npu2) {
    %shim = aie.logical_tile<ShimNOCTile>(0, 0)
    %m = aie.logical_tile<MemTile>(0, 1)
    %n = aie.logical_tile<MemTile>(1, 1)
    %fill0 = aie.buffer(%m) {sym_name = "fill0"} : memref<311296xi8>
    %fill1 = aie.buffer(%n) {sym_name = "fill1"} : memref<507904xi8>
    aie.objectfifo @f0(%shim, {%m}, 2 : i32) : !aie.objectfifo<memref<4096xi32>>
    aie.objectfifo @f0o(%m, {%shim}, 2 : i32) : !aie.objectfifo<memref<4096xi32>>
    aie.objectfifo.link [@f0] -> [@f0o] ([] [])
    aie.objectfifo @f1(%shim, {%m}, 2 : i32) : !aie.objectfifo<memref<4096xi32>>
    aie.objectfifo @f1o(%m, {%shim}, 2 : i32) : !aie.objectfifo<memref<4096xi32>>
    aie.objectfifo.link [@f1] -> [@f1o] ([] [])
    aie.objectfifo @f2(%shim, {%m}, 2 : i32) : !aie.objectfifo<memref<4096xi32>>
    aie.objectfifo @f2o(%m, {%shim}, 2 : i32) : !aie.objectfifo<memref<4096xi32>>
    aie.objectfifo.link [@f2] -> [@f2o] ([] [])
    aie.objectfifo @f3(%shim, {%m}, 2 : i32) : !aie.objectfifo<memref<4096xi32>>
    aie.objectfifo @f3o(%m, {%shim}, 2 : i32) : !aie.objectfifo<memref<4096xi32>>
    aie.objectfifo.link [@f3] -> [@f3o] ([] [])
    aie.objectfifo @f4(%shim, {%m}, 2 : i32) : !aie.objectfifo<memref<4096xi32>>
    aie.objectfifo @f4o(%m, {%shim}, 2 : i32) : !aie.objectfifo<memref<4096xi32>>
    aie.objectfifo.link [@f4] -> [@f4o] ([] [])
    aie.objectfifo @f5(%shim, {%m}, 2 : i32) : !aie.objectfifo<memref<4096xi32>>
    aie.objectfifo @f5o(%m, {%shim}, 2 : i32) : !aie.objectfifo<memref<4096xi32>>
    aie.objectfifo.link [@f5] -> [@f5o] ([] [])
    aie.objectfifo @f6(%shim, {%m}, 2 : i32) : !aie.objectfifo<memref<4096xi32>>
    aie.objectfifo @f6o(%m, {%shim}, 2 : i32) : !aie.objectfifo<memref<4096xi32>>
    aie.objectfifo.link [@f6] -> [@f6o] ([] [])
  }
}
