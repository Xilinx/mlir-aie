// RUN: aie-opt --split-input-file --aie-objectfifo-allocate --verify-diagnostics %s -o /dev/null

// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Core stream ports are also hardware resources shared by tile aliases.
module @existing_source {
  // expected-remark @+1 {{could not find a spill-aware allocation}}
  aie.device(npu2) {
    %home = aie.tile(0, 2)
    // expected-error @+1 {{number of output Core channels exceeded}}
    %alias = aie.logical_tile<CoreTile>(0, 2)
    %dest = aie.tile(1, 0)
    aie.flow(%home, Core : 0, %dest, DMA : 1)
    aie.route_endpoint @source(%alias) Core {channelIndex = 0 : i32}
    aie.route_endpoint @dest(%dest) DMA
    aie.route from @source to [@dest]
  }
}

// -----

module @existing_dest {
  // expected-remark @+1 {{could not find a spill-aware allocation}}
  aie.device(npu2) {
    %home = aie.tile(0, 2)
    // expected-error @+1 {{number of input Core channels exceeded}}
    %alias = aie.logical_tile<CoreTile>(0, 2)
    %source = aie.tile(1, 0)
    aie.flow(%source, DMA : 1, %home, Core : 0)
    aie.route_endpoint @source(%source) DMA
    aie.route_endpoint @dest(%alias) Core {channelIndex = 0 : i32}
    aie.route from @source to [@dest]
  }
}
