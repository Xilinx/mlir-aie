//===- test_sa_core_data_alignment.mlir ------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-place-tiles="placer=sa_placer sa-seed=1 sa-core-data-sizes=d:0=600" %s 2>&1 | FileCheck %s --check-prefix=BARE
// RUN: aie-opt --aie-place-tiles="placer=sa_placer sa-seed=1 sa-core-data-sizes=d:0=600@1" %s 2>&1 | FileCheck %s --check-prefix=BARE
// RUN: aie-opt --aie-place-tiles="placer=sa_placer sa-seed=1 sa-core-data-sizes=d:0=600@2048" %s 2>&1 | FileCheck %s --check-prefix=ALIGNED
// RUN: not aie-opt --aie-place-tiles="placer=sa_placer sa-core-data-sizes=d:0=600@3" %s 2>&1 | FileCheck %s --check-prefix=BAD
// RUN: not aie-opt --aie-place-tiles="placer=sa_placer sa-core-data-sizes=d:0=600@" %s 2>&1 | FileCheck %s --check-prefix=BAD

// A charge carries the alignment the core's sections start at, as addressing
// reserves them; without one it is 1. The 1 KiB stack, the 63488-byte buffer
// and a 600-byte data region fit the core's 64 KiB unpadded, not with the
// region padded to 2048 bytes.
// BARE-NOT: warning
// BARE: aie.tile
// ALIGNED: warning: SA placer's resource estimate is over capacity
// BAD: error: sa-core-data-sizes entry 'd:0=600@{{3?}}' is not <device>:<core>=<bytes>[@<align>]

module {
  aie.device(npu2) @d {
    %c = aie.logical_tile<CoreTile>(?, ?)
    %b = aie.buffer(%c) {sym_name = "b"} : memref<63488xi8>
    aie.core(%c) {
      aie.end
    }
  }
}
