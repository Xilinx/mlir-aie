// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: not aie-opt --split-input-file --aie-create-control-overlay --aie-assign-packet-ids --aie-objectfifo-allocate %s > %t.out 2> %t.err
// RUN: FileCheck %s < %t.out
// RUN: FileCheck %s --check-prefix=ERR < %t.err

// CHECK-LABEL: module @separate_configurations
// CHECK: aie.shim_dma_allocation @ctrlpkt_col0_mm2s_chan1
// CHECK: aie.shim_dma_allocation @a_shim_alloc(%{{.*}}, MM2S, 1, <pkt_id = {{[0-9]+}}>)
// CHECK: aie.shim_dma_allocation @b_shim_alloc(%{{.*}}, MM2S, 1, <pkt_id = {{[0-9]+}}>)
// ERR: number of output DMA channel exceeded

module @separate_configurations {
  aie.device(npu2_1col) @main {
    aie.runtime_sequence @dispatch(%input : memref<16xi32>) {
      aiex.configure @cfg {
        aiex.run @run_a(%input) : (memref<16xi32>)
      }
      aiex.configure @cfg {
        aiex.run @run_b(%input) : (memref<16xi32>)
      }
    }
  }
  aie.device(npu2_1col) @cfg {
    %shim = aie.tile(0, 0)
    %t2 = aie.tile(0, 2)
    %t3 = aie.tile(0, 3)
    aie.shim_dma_allocation @reserved(%shim, MM2S, 0)
    aie.route_endpoint @a(%shim) DMA
    aie.route_endpoint @a_out(%t2) Core {channelIndex = 0 : i32}
    aie.route from @a to [@a_out]
    aie.runtime_sequence @run_a(%input : memref<16xi32>) {
      aiex.npu.dma_memcpy_nd(%input[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0, 1]) {metadata = @a, id = 0 : i64} : memref<16xi32>
    }
    aie.route_endpoint @b(%shim) DMA
    aie.route_endpoint @b_out(%t3) Core {channelIndex = 0 : i32}
    aie.route from @b to [@b_out]
    aie.runtime_sequence @run_b(%input : memref<16xi32>) {
      aiex.npu.dma_memcpy_nd(%input[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0, 1]) {metadata = @b, id = 0 : i64} : memref<16xi32>
    }
  }
}

// -----

module @one_configuration {
  aie.device(npu2_1col) @main {
    aie.runtime_sequence @dispatch(%input : memref<16xi32>) {
      aiex.configure @cfg {
        aiex.run @run_a(%input) : (memref<16xi32>)
        aiex.run @run_b(%input) : (memref<16xi32>)
      }
    }
  }
  aie.device(npu2_1col) @cfg {
    %shim = aie.tile(0, 0)
    %t2 = aie.tile(0, 2)
    %t3 = aie.tile(0, 3)
    aie.shim_dma_allocation @reserved(%shim, MM2S, 0)
    aie.route_endpoint @a(%shim) DMA
    aie.route_endpoint @a_out(%t2) Core {channelIndex = 0 : i32}
    aie.route from @a to [@a_out]
    aie.runtime_sequence @run_a(%input : memref<16xi32>) {
      aiex.npu.dma_memcpy_nd(%input[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0, 1]) {metadata = @a, id = 0 : i64} : memref<16xi32>
    }
    aie.route_endpoint @b(%shim) DMA
    aie.route_endpoint @b_out(%t3) Core {channelIndex = 0 : i32}
    aie.route from @b to [@b_out]
    aie.runtime_sequence @run_b(%input : memref<16xi32>) {
      aiex.npu.dma_memcpy_nd(%input[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0, 1]) {metadata = @b, id = 0 : i64} : memref<16xi32>
    }
  }
}
