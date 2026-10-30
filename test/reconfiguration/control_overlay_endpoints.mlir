// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt --aie-create-control-overlay --aie-assign-packet-ids --aie-objectfifo-allocate %s | FileCheck %s
// RUN: not aie-opt --aie-create-control-overlay="adaptive=false" --aie-assign-packet-ids --aie-objectfifo-allocate %s 2>&1 | FileCheck %s --check-prefix=FIXED

// CHECK-LABEL: @endpoints
// CHECK: aie.shim_dma_allocation @ctrlpkt_col0_mm2s_chan1
// CHECK-SAME: aie.fixed
// CHECK: aie.shim_dma_allocation @input_shim_alloc(%{{.*}}, MM2S, 1, <pkt_id = {{[0-9]+}}>)
// CHECK-LABEL: @ctrl_pkt_overlay
// FIXED: {{already in use|number of output DMA channel exceeded|failed to generate|Unable to find}}

module {
  aie.device(npu2_1col) @endpoints {
    %shim = aie.tile(0, 0)
    %core = aie.tile(0, 2)
    %rtp = aie.buffer(%core) {sym_name = "rtp"} : memref<4xi32>
    aie.shim_dma_allocation @reserved(%shim, MM2S, 0)
    aie.route_endpoint @input(%shim) DMA
    aie.route_endpoint @output(%core) Core {channelIndex = 0 : i32}
    aie.route from @input to [@output]
    aie.runtime_sequence(%arg : memref<16xi32>) {
      %c0 = arith.constant 0 : i32
      aiex.npu.write32(%c0, %c0) {buffer = @rtp} : i32, i32
      aiex.npu.dma_memcpy_nd(%arg[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0, 1]) {metadata = @input, id = 0 : i64} : memref<16xi32>
    }
  }
}
