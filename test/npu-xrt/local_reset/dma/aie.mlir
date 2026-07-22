//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// DMA channel reset: reset a stalled run-forever channel, re-push BD 0, re-arm
// the lock. See README.md.

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
      // Reset MM2S channel 0 (channel-control register, tile-local 0x1DE10 =
      // 122384): disable -> assert reset (bit 1) -> deassert -> enable (bit 0).
      // Valid because the channel is stalled on the cons lock acquire.
      %cc = arith.constant 122384 : i32
      %disable = arith.constant 0 : i32
      %reset = arith.constant 2 : i32
      %enable = arith.constant 1 : i32
      aiex.npu.write32(%cc, %disable) {column = 0 : i32, row = 2 : i32} : i32, i32
      aiex.npu.write32(%cc, %reset) {column = 0 : i32, row = 2 : i32} : i32, i32
      aiex.npu.write32(%cc, %disable) {column = 0 : i32, row = 2 : i32} : i32, i32
      aiex.npu.write32(%cc, %enable) {column = 0 : i32, row = 2 : i32} : i32, i32

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
