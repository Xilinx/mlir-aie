// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// A core arming its own tile's DMA channel.
//
// The receiving channel runs once: its descriptor chain does not loop, so it
// takes one buffer and stops. The core waits on a lock only that channel
// releases, arms the channel again, and waits a second time. The second wait
// returns only if the arming worked, and the token the core then sends is the
// only way the host hears about it.

module {
  aie.device(NPUDEVICE) {
    %shim = aie.tile(0, 0)
    %core_tile = aie.tile(0, 2)

    %in_buf = aie.buffer(%core_tile) {sym_name = "in_buf"} : memref<8xi32>
    %tok_buf = aie.buffer(%core_tile) {sym_name = "tok_buf"} : memref<8xi32>

    // in_prod says the buffer is free, in_cons that the channel filled it.
    // Only the DMA releases in_cons.
    %in_prod = aie.lock(%core_tile, 0) {init = 2 : i32, sym_name = "in_prod"}
    %in_cons = aie.lock(%core_tile, 1) {init = 0 : i32, sym_name = "in_cons"}
    %tok_prod = aie.lock(%core_tile, 2) {init = 1 : i32, sym_name = "tok_prod"}
    %tok_cons = aie.lock(%core_tile, 3) {init = 0 : i32, sym_name = "tok_cons"}

    aie.flow(%shim, DMA : 0, %core_tile, DMA : 0)
    aie.flow(%core_tile, DMA : 0, %shim, DMA : 0)

    %core = aie.core(%core_tile) {
      %c0 = arith.constant 0 : index
      %c1_i32 = arith.constant 1 : i32
      %c42_i32 = arith.constant 42 : i32

      // The channel was armed once, so this is the first buffer.
      aie.use_lock(%in_cons, AcquireGreaterEqual, %c1_i32)

      // Arm it again, for the buffer the host sent after the first.
      aie.dma_rearm(S2MM : 0) bd_id 0 {repeat_count = 0 : i32}

      // Only an armed channel releases this a second time.
      aie.use_lock(%in_cons, AcquireGreaterEqual, %c1_i32)

      aie.use_lock(%tok_prod, AcquireGreaterEqual, %c1_i32)
      memref.store %c42_i32, %tok_buf[%c0] : memref<8xi32>
      aie.use_lock(%tok_cons, Release, %c1_i32)
      aie.end
    } {aie.processor_bus}

    aie.mem(%core_tile) {
      // One descriptor, and the chain does not come round: the channel takes
      // one buffer per arming.
      %0 = aie.dma(S2MM, 0) {loop = false, repeat_count = 0 : i32} [{
        %c1_a = arith.constant 1 : i32
        aie.use_lock(%in_prod, AcquireGreaterEqual, %c1_a)
        aie.dma_bd(%in_buf : memref<8xi32>) {bd_id = 0 : i32}
        aie.use_lock(%in_cons, Release, %c1_a)
      }]
      %1 = aie.dma(MM2S, 0) [{
        %c1_b = arith.constant 1 : i32
        aie.use_lock(%tok_cons, AcquireGreaterEqual, %c1_b)
        aie.dma_bd(%tok_buf : memref<8xi32>) {bd_id = 1 : i32}
        aie.use_lock(%tok_prod, Release, %c1_b)
      }]
      aie.end
    }

    aie.shim_dma_allocation @send(%shim, MM2S, 0)
    aie.shim_dma_allocation @recv(%shim, S2MM, 0)

    aie.runtime_sequence(%src : memref<16xi32>, %tok : memref<8xi32>) {
      %t_recv = aiex.dma_configure_task_for @recv {
        aie.dma_bd(%tok : memref<8xi32>)
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t_recv)

      // Two buffers: the second waits on the arming the core issues.
      %t0 = aiex.dma_configure_task_for @send {
        aie.dma_bd(%src : memref<16xi32> offset = 0 len = 8)
        aie.end
      }
      aiex.dma_start_task(%t0)
      %t1 = aiex.dma_configure_task_for @send {
        aie.dma_bd(%src : memref<16xi32> offset = 8 len = 8)
        aie.end
      }
      aiex.dma_start_task(%t1)

      aiex.dma_await_task(%t_recv)
    }
  }
}
