//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

// RUN: aie-opt --aie-assign-bd-ids --aie-substitute-shim-dma-allocations \
// RUN:         --aie-assign-runtime-sequence-bd-ids \
// RUN:         --verify-diagnostics --split-input-file %s

// A static shim DMA program claims a BD id (aie-assign-bd-ids) before the
// runtime sequence's own allocator ever runs, so a copy naming that id gets
// the same "already in use" refusal a task would: there is no earlier task
// to release, but it is the same unresolvable collision, not a copy-over-
// copy or copy-over-task overlap with any escape through this pass's model.
module @copy_over_static_bd {
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
      %c0 = arith.constant 0 : i32
      %c1 = arith.constant 1 : i32
      // expected-error@+1 {{Specified buffer descriptor ID 0 is already in use. Release the earlier task first: aiex.dma_await_task waits for hardware completion before its BDs are reusable. aiex.dma_free_task also releases them but does NOT wait, so it is only safe when some other synchronization already guarantees that task has finished (see programming_guide/section-2/section-2d/DMATasks.md).}}
      aiex.npu.dma_memcpy_nd(%pkts[0, 0, 0, 0][1, 1, 1, 3][0, 0, 0, 1]) {id = 0 : i64, issue_token = true, metadata = @ctrl} : memref<?xi32>
      aiex.npu.dma_wait {symbol = @ctrl}
    }
  }
}

// -----

// The same id on a different channel is unaffected.
// CHECK-LABEL: @copy_over_static_bd_different_channel
module @copy_over_static_bd_different_channel {
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
      %c0 = arith.constant 0 : i32
      %c1 = arith.constant 1 : i32
      aiex.npu.dma_memcpy_nd(%pkts[0, 0, 0, 0][1, 1, 1, 3][0, 0, 0, 1]) {id = 1 : i64, issue_token = true, metadata = @ctrl} : memref<?xi32>
      aiex.npu.dma_wait {symbol = @ctrl}
    }
  }
}
