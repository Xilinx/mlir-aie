// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt --split-input-file --aie-generate-column-control-overlay="configuration=true adaptive=false" --aie-import-overlay --aie-annotate-shim-sharing %s | FileCheck %s

// An overlap in any callee prevents sharing, even when other calls are disjoint.
// CHECK-LABEL: module @callee_overlap
// CHECK: aie.route_endpoint @a(
// CHECK-SAME: aie.shareable_with = []
// CHECK: aie.route_endpoint @b(
// CHECK-SAME: aie.shareable_with = []
module @callee_overlap {
  aie.device(npu2_1col) @cfg {
    %shim = aie.tile(0, 0)
    %t2 = aie.tile(0, 2)
    %t3 = aie.tile(0, 3)
    aie.route_endpoint @a(%shim) DMA
    aie.route_endpoint @a_out(%t2) Core {channelIndex = 0 : i32}
    aie.route from @a to [@a_out]
    aie.route_endpoint @b(%shim) DMA
    aie.route_endpoint @b_out(%t3) Core {channelIndex = 0 : i32}
    aie.route from @b to [@b_out]
    aie.runtime_sequence @dispatch(%input : memref<16xi32>) {
      aiex.configure @cfg {
        aiex.run @both(%input) : (memref<16xi32>)
        aiex.run @one(%input) : (memref<16xi32>)
      }
    }
    aie.runtime_sequence @both(%input : memref<16xi32>) {
      aiex.npu.dma_memcpy_nd(%input[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0, 1]) {metadata = @a, id = 0 : i64, issue_token = true} : memref<16xi32>
      aiex.npu.dma_memcpy_nd(%input[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0, 1]) {metadata = @b, id = 0 : i64, issue_token = true} : memref<16xi32>
      aiex.npu.dma_wait {symbol = @a}
      aiex.npu.dma_wait {symbol = @b}
    }
    aie.runtime_sequence @one(%input : memref<16xi32>) {
      aiex.npu.dma_memcpy_nd(%input[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0, 1]) {metadata = @a, id = 0 : i64, issue_token = true} : memref<16xi32>
      aiex.npu.dma_wait {symbol = @a}
    }
  }
}

// -----

// Completing a callee cannot complete the caller's earlier transfer.
// CHECK-LABEL: module @caller_live
// CHECK: aie.route_endpoint @a(
// CHECK-SAME: aie.shareable_with = []
// CHECK: aie.route_endpoint @b(
// CHECK-SAME: aie.shareable_with = []
// CHECK: aie.shim_dma_allocation @ctrlpkt_col0_mm2s_chan0
// CHECK-SAME: aie.shareable_with = [@b]
module @caller_live {
  aie.device(npu2_1col) @cfg {
    %shim = aie.tile(0, 0)
    %t2 = aie.tile(0, 2)
    %t3 = aie.tile(0, 3)
    aie.route_endpoint @a(%shim) DMA
    aie.route_endpoint @a_out(%t2) Core {channelIndex = 0 : i32}
    aie.route from @a to [@a_out]
    aie.route_endpoint @b(%shim) DMA
    aie.route_endpoint @b_out(%t3) Core {channelIndex = 0 : i32}
    aie.route from @b to [@b_out]
    aie.runtime_sequence @dispatch(%input : memref<16xi32>) {
      aiex.npu.dma_memcpy_nd(%input[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0, 1]) {metadata = @a, id = 0 : i64, issue_token = true} : memref<16xi32>
      aiex.configure @cfg {
        aiex.run @one(%input) : (memref<16xi32>)
      }
      aiex.npu.dma_wait {symbol = @a}
    }
    aie.runtime_sequence @one(%input : memref<16xi32>) {
      aiex.npu.dma_memcpy_nd(%input[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0, 1]) {metadata = @b, id = 0 : i64, issue_token = true} : memref<16xi32>
      aiex.npu.dma_wait {symbol = @b}
    }
  }
}

// -----

// A nested call's summary retains endpoint identities in the defining device.
// CHECK-LABEL: module @nested_calls
// CHECK: aie.route_endpoint @a(
// CHECK-SAME: aie.shareable_with = [@b]
// CHECK: aie.route_endpoint @b(
// CHECK-SAME: aie.shareable_with = [@a]
// CHECK: aie.shim_dma_allocation @ctrlpkt_col0_mm2s_chan0
// CHECK-SAME: aie.shareable_with = [@a, @b]
module @nested_calls {
  aie.device(npu2_1col) @cfg {
    %shim = aie.tile(0, 0)
    %t2 = aie.tile(0, 2)
    %t3 = aie.tile(0, 3)
    aie.route_endpoint @a(%shim) DMA
    aie.route_endpoint @a_out(%t2) Core {channelIndex = 0 : i32}
    aie.route from @a to [@a_out]
    aie.route_endpoint @b(%shim) DMA
    aie.route_endpoint @b_out(%t3) Core {channelIndex = 0 : i32}
    aie.route from @b to [@b_out]
    aie.runtime_sequence @dispatch(%input : memref<16xi32>) {
      aiex.configure @cfg {
        aiex.run @wrapper(%input) : (memref<16xi32>)
        aiex.run @one(%input) : (memref<16xi32>)
      }
    }
    aie.runtime_sequence @wrapper(%input : memref<16xi32>) {
      aiex.configure @cfg {
        aiex.run @two(%input) : (memref<16xi32>)
        aiex.run @one(%input) : (memref<16xi32>)
      }
    }
    aie.runtime_sequence @one(%input : memref<16xi32>) {
      aiex.npu.dma_memcpy_nd(%input[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0, 1]) {metadata = @a, id = 0 : i64, issue_token = true} : memref<16xi32>
      aiex.npu.dma_wait {symbol = @a}
    }
    aie.runtime_sequence @two(%input : memref<16xi32>) {
      aiex.npu.dma_memcpy_nd(%input[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0, 1]) {metadata = @b, id = 0 : i64, issue_token = true} : memref<16xi32>
      aiex.npu.dma_wait {symbol = @b}
    }
  }
}

// -----

// A call boundary does not order independent S2MM producers.
// CHECK-LABEL: module @callee_receives
// CHECK: aie.route_endpoint @a(
// CHECK-SAME: aie.shareable_with = []
// CHECK: aie.route_endpoint @b(
// CHECK-SAME: aie.shareable_with = []
module @callee_receives {
  aie.device(npu2_1col) @cfg {
    %shim = aie.tile(0, 0)
    %t2 = aie.tile(0, 2)
    %t3 = aie.tile(0, 3)
    aie.route_endpoint @a(%shim) DMA
    aie.route_endpoint @a_out(%t2) Core {channelIndex = 0 : i32}
    aie.route from @a_out to [@a]
    aie.route_endpoint @b(%shim) DMA
    aie.route_endpoint @b_out(%t3) Core {channelIndex = 0 : i32}
    aie.route from @b_out to [@b]
    aie.runtime_sequence @dispatch(%input : memref<16xi32>) {
      aiex.configure @cfg {
        aiex.run @both(%input) : (memref<16xi32>)
      }
    }
    aie.runtime_sequence @both(%input : memref<16xi32>) {
      aiex.npu.dma_memcpy_nd(%input[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0, 1]) {metadata = @a, id = 0 : i64, issue_token = true} : memref<16xi32>
      aiex.npu.dma_wait {symbol = @a}
      aiex.npu.dma_memcpy_nd(%input[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0, 1]) {metadata = @b, id = 0 : i64, issue_token = true} : memref<16xi32>
      aiex.npu.dma_wait {symbol = @b}
    }
  }
}

// -----

// CHECK-LABEL: module @distinct_devices
// CHECK: aie.device(npu2_1col) @left
// CHECK: aie.route_endpoint @a(
// CHECK-SAME: aie.shareable_with = []
// CHECK: aie.device(npu2_1col) @right
// CHECK: aie.route_endpoint @a(
// CHECK-SAME: aie.shareable_with = [@b]
module @distinct_devices {
  aie.device(npu2_1col) @left {
    %shim = aie.tile(0, 0)
    %t2 = aie.tile(0, 2)
    %t3 = aie.tile(0, 3)
    aie.route_endpoint @a(%shim) DMA
    aie.route_endpoint @a_out(%t2) Core {channelIndex = 0 : i32}
    aie.route from @a to [@a_out]
    aie.route_endpoint @b(%shim) DMA
    aie.route_endpoint @b_out(%t3) Core {channelIndex = 0 : i32}
    aie.route from @b to [@b_out]
    aie.runtime_sequence @dispatch(%input : memref<16xi32>) {
      aiex.configure @left {
        aiex.run @both(%input) : (memref<16xi32>)
        aiex.run @one(%input) : (memref<16xi32>)
      }
    }
    aie.runtime_sequence @both(%input : memref<16xi32>) {
      aiex.npu.dma_memcpy_nd(%input[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0, 1]) {metadata = @a, id = 0 : i64, issue_token = true} : memref<16xi32>
      aiex.npu.dma_memcpy_nd(%input[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0, 1]) {metadata = @b, id = 0 : i64, issue_token = true} : memref<16xi32>
      aiex.npu.dma_wait {symbol = @a}
      aiex.npu.dma_wait {symbol = @b}
    }
    aie.runtime_sequence @one(%input : memref<16xi32>) {
      aiex.npu.dma_memcpy_nd(%input[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0, 1]) {metadata = @a, id = 0 : i64, issue_token = true} : memref<16xi32>
      aiex.npu.dma_wait {symbol = @a}
    }
  }
  aie.device(npu2_1col) @right {
    %shim = aie.tile(0, 0)
    %t2 = aie.tile(0, 2)
    %t3 = aie.tile(0, 3)
    aie.route_endpoint @a(%shim) DMA
    aie.route_endpoint @a_out(%t2) Core {channelIndex = 0 : i32}
    aie.route from @a to [@a_out]
    aie.route_endpoint @b(%shim) DMA
    aie.route_endpoint @b_out(%t3) Core {channelIndex = 0 : i32}
    aie.route from @b to [@b_out]
    aie.runtime_sequence @dispatch(%input : memref<16xi32>) {
      aiex.configure @right {
        aiex.run @wrapper(%input) : (memref<16xi32>)
        aiex.run @one(%input) : (memref<16xi32>)
      }
    }
    aie.runtime_sequence @wrapper(%input : memref<16xi32>) {
      aiex.configure @right {
        aiex.run @two(%input) : (memref<16xi32>)
        aiex.run @one(%input) : (memref<16xi32>)
      }
    }
    aie.runtime_sequence @one(%input : memref<16xi32>) {
      aiex.npu.dma_memcpy_nd(%input[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0, 1]) {metadata = @a, id = 0 : i64, issue_token = true} : memref<16xi32>
      aiex.npu.dma_wait {symbol = @a}
    }
    aie.runtime_sequence @two(%input : memref<16xi32>) {
      aiex.npu.dma_memcpy_nd(%input[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0, 1]) {metadata = @b, id = 0 : i64, issue_token = true} : memref<16xi32>
      aiex.npu.dma_wait {symbol = @b}
    }
  }
}
