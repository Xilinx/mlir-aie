//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// DMA channel reset via the aiex.dma_channel_reset op -- op-based analog of
// ../dma. Same falsifiable two-BD design (bad BD 0 [900..907], good BD 1
// [100..107], both gated on cons): each dispatch enqueues the bad BD, resets the
// channel to flush it, enqueues the good BD, then arms the lock. Drop the reset
// and the bad BD reaches the host. See README.md.

module {
  aie.device(NPUDEVICE) {
    %t00 = aie.tile(0, 0)
    %t02 = aie.tile(0, 2)

    %cons = aie.lock(%t02, 0) {init = 0 : i32, sym_name = "cons"}
    %prod = aie.lock(%t02, 1) {init = 0 : i32, sym_name = "prod"}
    %good = aie.buffer(%t02) { initial_value = dense<[100,101,102,103,104,105,106,107]> : tensor<8xi32>, sym_name = "good" } : memref<8xi32>
    %bad  = aie.buffer(%t02) { initial_value = dense<[900,901,902,903,904,905,906,907]> : tensor<8xi32>, sym_name = "bad" } : memref<8xi32>

    aie.flow(%t02, DMA : 0, %t00, DMA : 0)

    // Both BDs gate on cons (init 0), so nothing sends until the runtime sequence
    // arms the lock. Chaining bad -> good configures both descriptors at load.
    %mem = aie.mem(%t02) {
      %s = aie.dma_start(MM2S, 0, ^bd0, ^end)
    ^bd0:
      %o0 = arith.constant 1 : i32
      aie.use_lock(%cons, AcquireGreaterEqual, %o0)
      aie.dma_bd(%bad : memref<8xi32> offset = 0 len = 8) {bd_id = 0 : i32}
      aie.use_lock(%prod, Release, %o0)
      aie.next_bd ^bd1
    ^bd1:
      %o1 = arith.constant 1 : i32
      aie.use_lock(%cons, AcquireGreaterEqual, %o1)
      aie.dma_bd(%good : memref<8xi32> offset = 0 len = 8) {bd_id = 1 : i32}
      aie.use_lock(%prod, Release, %o1)
      aie.next_bd ^end
    ^end:
      aie.end
    }

    aie.shim_dma_allocation @out0 (%t00, S2MM, 0)

    aie.runtime_sequence @seq(%arg0: memref<8xi32>) {
      // Enqueue the bad BD (0); it cannot run yet (cons is 0).
      %bd_bad = arith.constant 0 : i32
      %rc = arith.constant 0 : i32
      aiex.npu.push_queue (0, 2, MM2S:0) bd_id %bd_bad repeat %rc {issue_token = false} : i32, i32

      // Flush the queue with the merged op: it lowers to the same masked reset
      // pulse on DMA_MM2S_0_Ctrl (bit 1) that ../dma issues directly, mirroring
      // aie-rt's XAie_DmaChannelReset. Reset-only, so the enqueue + lock arm below
      // are still needed.
      aiex.dma_channel_reset(%t02, MM2S, 0)

      // Enqueue the good BD (1) on the flushed queue (aie-rt
      // XAie_DmaChannelPushBdToQueue -- DMA_MM2S_0_Start_Queue, 0x1DE14).
      %bd_good = arith.constant 1 : i32
      aiex.npu.push_queue (0, 2, MM2S:0) bd_id %bd_good repeat %rc {issue_token = true} : i32, i32

      // Arm cons so the good BD runs (aie-rt XAie_LockSetValue via aiex.set_lock).
      aiex.set_lock(%cons, 1)

      aiex.npu.dma_memcpy_nd(%arg0[0,0,0,0][1,1,1,8][0,0,0,1]) {id=0:i64, issue_token=true, metadata=@out0} : memref<8xi32>
      aiex.npu.dma_wait {symbol=@out0}
    }
  }
}
