//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

// RUN: aie-opt --aie-assign-bd-ids --aie-substitute-shim-dma-allocations \
// RUN:         --aie-assign-runtime-sequence-bd-ids \
// RUN:         --verify-diagnostics --split-input-file %s

// A static shim DMA program claims a BD id (aie-assign-bd-ids) before the
// runtime sequence's own allocator runs, so a copy naming that id is
// refused as "already in use", as a task naming it would be.
module @copy_over_static_bd {
  aie.device(npu2) {
    %s0 = aie.tile(0, 0)
    %sb = aie.external_buffer {sym_name = "sb"} : memref<8xi32>
    aie.shim_dma(%s0) {
        aie.dma_start(MM2S, 0, ^bd0, ^end)
      ^bd0:
        // expected-note@+1 {{the static program's buffer descriptor is here}}
        aie.dma_bd(%sb : memref<8xi32> offset = 0 len = 4)
        aie.next_bd ^end
      ^end:
        aie.end
    }
    aie.shim_dma_allocation @ctrl(%s0, MM2S, 1)
    aie.runtime_sequence @seq(%pkts : memref<?xi32>) {
      // expected-error@+1 {{programs buffer descriptor ID 0, which a static DMA program on tile (0,0) already uses; give the copy another ID}}
      aiex.npu.dma_memcpy_nd(%pkts[0, 0, 0, 0][1, 1, 1, 3][0, 0, 0, 1]) {id = 0 : i64, issue_token = true, metadata = @ctrl} : memref<?xi32>
      aiex.npu.dma_wait {symbol = @ctrl}
    }
  }
}

// -----

// Another id is unaffected.
module @copy_over_static_bd_other_id {
  aie.device(npu2) {
    %s0 = aie.tile(0, 0)
    %sb = aie.external_buffer {sym_name = "sb"} : memref<8xi32>
    aie.shim_dma(%s0) {
        aie.dma_start(MM2S, 0, ^bd0, ^end)
      ^bd0:
        aie.dma_bd(%sb : memref<8xi32> offset = 0 len = 4)
        aie.next_bd ^end
      ^end:
        aie.end
    }
    aie.shim_dma_allocation @ctrl(%s0, MM2S, 1)
    aie.runtime_sequence @seq(%pkts : memref<?xi32>) {
      aiex.npu.dma_memcpy_nd(%pkts[0, 0, 0, 0][1, 1, 1, 3][0, 0, 0, 1]) {id = 1 : i64, issue_token = true, metadata = @ctrl} : memref<?xi32>
      aiex.npu.dma_wait {symbol = @ctrl}
    }
  }
}
