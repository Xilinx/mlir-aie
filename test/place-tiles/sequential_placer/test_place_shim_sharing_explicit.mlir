// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt --aie-place-tiles --aie-objectfifo-allocate %s | FileCheck %s
// RUN: aie-opt --aie-place-tiles='placer=sa_placer sa-seed=42' --aie-objectfifo-allocate %s | FileCheck %s
// RUN: aie-opt --aie-place-tiles --aie-annotate-shim-sharing --aie-objectfifo-allocate %s | FileCheck %s
// RUN: sed 's/ {aie.shareable_with = \[@b\]}//' %s | not aie-opt --aie-place-tiles 2>&1 | FileCheck %s --check-prefix=DENIED
// RUN: sed 's/ {aie.shareable_with = \[@b\]}//' %s | not aie-opt --aie-place-tiles='placer=sa_placer sa-seed=42' --aie-objectfifo-allocate 2>&1 | FileCheck %s --check-prefix=DENIED

// These explicit permissions are trusted even without runtime lifetime evidence.
// CHECK-DAG: aie.route_endpoint @a(%[[SHIM:[a-zA-Z0-9_]+]]) DMA {{.*}}channelIndex = [[CH:[0-9]]] : i32
// CHECK-DAG: aie.route_endpoint @b(%[[SHIM]]) DMA {{.*}}channelIndex = [[CH]] : i32
// DENIED: error:

module {
  aie.device(npu2_1col) {
    %s0 = aie.logical_tile<ShimNOCTile>(0, 0)
    %s1 = aie.logical_tile<ShimNOCTile>(0, 0)
    %s2 = aie.logical_tile<ShimNOCTile>(0, 0)
    %t2 = aie.tile(0, 2)
    %t3 = aie.tile(0, 3)
    %t4 = aie.tile(0, 4)
    aie.route_endpoint @a(%s0) DMA {aie.shareable_with = [@b]}
    aie.route_endpoint @b(%s1) DMA {aie.shareable_with = [@a]}
    aie.route_endpoint @c(%s2) DMA
    aie.route_endpoint @source_a(%t2) Core {channelIndex = 0 : i32}
    aie.route_endpoint @source_b(%t3) Core {channelIndex = 0 : i32}
    aie.route_endpoint @source_c(%t4) Core {channelIndex = 0 : i32}
    aie.route from @source_a to [@a]
    aie.route from @source_b to [@b]
    aie.route from @source_c to [@c]
  }
}
