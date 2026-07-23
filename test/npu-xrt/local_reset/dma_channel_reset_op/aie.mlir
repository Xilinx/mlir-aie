//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// DMA channel reset via the aiex.dma_channel_reset op (op-based analog of ../dma):
// reset a stalled run-forever channel, re-push BD 0, re-arm the lock. See README.md.

module {
  aie.device(NPUDEVICE) {
    %t00 = aie.tile(0, 0)
    %t02 = aie.tile(0, 2)

    %cons = aie.lock(%t02, 0) {init = 0 : i32, sym_name = "cons"}
    %prod = aie.lock(%t02, 1) {init = 0 : i32, sym_name = "prod"}
    %buf = aie.buffer(%t02) { initial_value = dense<[100,101,102,103,104,105,106,107]> : tensor<8xi32>, sym_name = "buf" } : memref<8xi32>

    aie.flow(%t02, DMA : 0, %t00, DMA : 0)

    %mem = aie.mem(%t02) {
      %s = aie.dma_start(MM2S, 0, ^bd0, ^end)
    ^bd0:
      %o = arith.constant 1 : i32
      aie.use_lock(%cons, AcquireGreaterEqual, %o)
      aie.dma_bd(%buf : memref<8xi32> offset = 0 len = 8)
      aie.use_lock(%prod, Release, %o)
      aie.next_bd ^bd0
    ^end:
      aie.end
    }

    aie.shim_dma_allocation @out0 (%t00, S2MM, 0)

    aie.runtime_sequence @seq(%arg0: memref<8xi32>) {
      // Reset MM2S channel 0 via the merged runtime-sequence op. It lowers to a
      // mask-preserving reset pulse on the channel-control register (assert bit 1,
      // then clear it), preserving the other CTRL fields -- the op-based analog of
      // ../dma's raw disable->reset->deassert->enable. Valid because the channel is
      // stalled on the cons lock acquire. The op is reset-only, so the re-push and
      // lock re-arm below are still needed to make the channel run again.
      aiex.dma_channel_reset(%t02, MM2S, 0)

      // Re-push BD 0 to the (now flushed) queue so the channel runs again.
      %bd = arith.constant 0 : i32
      %rc = arith.constant 0 : i32
      aiex.npu.push_queue (0, 2, MM2S:0) bd_id %bd repeat %rc {issue_token = true} : i32, i32

      // Re-arm the cons lock (LOCK0_VALUE, tile-local 0x1F000 = 126976; same
      // offset on AIE-ML/npu1 and AIE2P/npu2) so the send proceeds.
      %la = arith.constant 126976 : i32
      %one = arith.constant 1 : i32
      aiex.npu.write32(%la, %one) {column = 0 : i32, row = 2 : i32} : i32, i32

      aiex.npu.dma_memcpy_nd(%arg0[0,0,0,0][1,1,1,8][0,0,0,1]) {id=0:i64, issue_token=true, metadata=@out0} : memref<8xi32>
      aiex.npu.dma_wait {symbol=@out0}
    }
  }
}
