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

// -----

module @configured_after_completion {
  aie.device(npu2) @cfg {
    %shim = aie.tile(0, 0)
    %buffer = aie.external_buffer {sym_name = "buffer"} : memref<8xi32>
    aie.shim_dma(%shim) {
      aie.dma_start(S2MM, 1, ^bd, ^end)
    ^bd:
      // expected-note@+1 {{the static program's buffer descriptor is here}}
      aie.dma_bd(%buffer : memref<8xi32> offset = 0 len = 8) {bd_id = 0 : i32}
      aie.next_bd ^end
    ^end:
      aie.end
    }
  }
  aie.device(npu2) {
    %shim = aie.tile(0, 0)
    aie.shim_dma_allocation @data(%shim, MM2S, 1)
    aie.shim_dma_allocation @ctrl(%shim, MM2S, 0)
    aie.runtime_sequence @seq(%buffer: memref<8xi32>) {
      %earlier = aiex.dma_configure_task_for @data {
        aie.dma_bd(%buffer : memref<8xi32> offset = 0 len = 8) {bd_id = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%earlier)
      aiex.npu.dma_wait {symbol = @data}
      %configure = aiex.dma_configure_task_for @ctrl {
        aie.dma_bd(%buffer : memref<8xi32> offset = 0 len = 8)
        aie.end
      } {issue_token = true, aiex.ctrl_pkt_transfer = @cfg}
      aiex.dma_start_task(%configure)
      aiex.dma_await_task(%configure)
      // expected-error@+1 {{programs buffer descriptor ID 0, which a static DMA program on tile (0,0) already uses; give the copy another ID}}
      aiex.npu.dma_memcpy_nd(%buffer[0, 0, 0, 0][1, 1, 1, 8][0, 0, 0, 1]) {id = 0 : i64, issue_token = true, metadata = @data} : memref<8xi32>
    }
  }
}
