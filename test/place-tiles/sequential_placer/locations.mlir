// RUN: aie-opt --aie-place-tiles --mlir-print-debuginfo %s | FileCheck %s
// RUN: aie-opt --aie-place-tiles --aie-place-tiles --mlir-print-debuginfo %s | FileCheck %s

// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Keep all source locations when merging into an existing or a new physical
// tile. A singleton retains its location, rather than inheriting the device's.
module {
  aie.device(npu2) {
    %physical = aie.tile(0, 1) loc("tiles.py":10:3)
    %alias = aie.logical_tile<MemTile>(0, 1) loc("tiles.py":20:3)
    %first = aie.logical_tile<MemTile>(1, 1) loc("tiles.py":30:3)
    %second = aie.logical_tile<MemTile>(1, 1) loc("tiles.py":40:3)
    %single = aie.logical_tile<CoreTile>(0, 2) loc("tiles.py":50:3)
  }
}

// CHECK-DAG: aie.tile(0, 1) loc(#[[EXISTING:loc[0-9]+]])
// CHECK-DAG: aie.tile(1, 1) loc(#[[NEW:loc[0-9]+]])
// CHECK-DAG: aie.tile(0, 2) loc(#[[SINGLE:loc[0-9]+]])
// CHECK-DAG: #[[PHYSICAL:loc[0-9]+]] = loc("tiles.py":10:3)
// CHECK-DAG: #[[ALIAS:loc[0-9]+]] = loc("tiles.py":20:3)
// CHECK-DAG: #[[FIRST:loc[0-9]+]] = loc("tiles.py":30:3)
// CHECK-DAG: #[[SECOND:loc[0-9]+]] = loc("tiles.py":40:3)
// CHECK-DAG: #[[SINGLE]] = loc("tiles.py":50:3)
// CHECK-DAG: #[[EXISTING]] = loc(fused[#[[PHYSICAL]], #[[ALIAS]]])
// CHECK-DAG: #[[NEW]] = loc(fused[#[[FIRST]], #[[SECOND]]])
