// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt --split-input-file --aie-generate-column-control-overlay="configuration=true adaptive=true" --verify-diagnostics %s -o /dev/null

module {
  // expected-error @+1 {{cannot plan adaptive control overlay: Unable to find a legal routing:}}
  aie.device(npu2_1col) @unroutable {
    %shim = aie.tile(0, 0)
    %mem = aie.tile(0, 1)
    %t2 = aie.tile(0, 2)
    %t3 = aie.tile(0, 3)
    %t4 = aie.tile(0, 4)
    %t5 = aie.tile(0, 5)
    aie.flow(%shim, DMA : 0, %t5, DMA : 0)
    aie.flow(%mem, DMA : 0, %t2, DMA : 0)
    aie.flow(%mem, DMA : 1, %t2, DMA : 1)
    aie.flow(%mem, DMA : 2, %t3, DMA : 0)
    aie.flow(%mem, DMA : 3, %t3, DMA : 1)
    aie.flow(%mem, DMA : 4, %t4, DMA : 0)
    aie.flow(%mem, DMA : 5, %t4, DMA : 1)
  }
}

// -----

// expected-error @+1 {{failed to analyze application resources for adaptive control overlay}}
module {
  // expected-remark @+1 {{could not find a spill-aware allocation}}
  aie.device(npu2_1col) @unallocatable {
    // expected-error @+1 {{requires at least 3 MM2S channels, but capacity is 2}}
    %shim = aie.tile(0, 0)
    %core = aie.tile(0, 2)
    // expected-note @+1 {{pre-existing aie.shim_dma_allocation reserves DMA channel 0}}
    aie.shim_dma_allocation @reserved(%shim, MM2S, 0)
    // expected-note @+1 {{DMA endpoint @a; occupies channel 1}}
    aie.route_endpoint @a(%shim) DMA
    // expected-note @+1 {{DMA endpoint @b}}
    aie.route_endpoint @b(%shim) DMA
    aie.route_endpoint @a_out(%core) Core {channelIndex = 0 : i32}
    aie.route_endpoint @b_out(%core) Core {channelIndex = 1 : i32}
    aie.route from @a to [@a_out]
    aie.route from @b to [@b_out]
  }
}
