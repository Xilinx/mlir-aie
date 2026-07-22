//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Stream-switch connection reset: config-disable then re-enable the DMA:0 slave
// port, re-arm the lock. See README.md.

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
      // Config-disable then re-enable the DMA:0 slave port
      // (Stream_Switch_Slave_DMA_0_Config, 0x3F104). Only this port is touched
      // (the master South config is left resident), so the reset is isolated to
      // the switch connection. Valid because the channel is stalled on the cons
      // lock, so the route is idle.
      %slave = arith.constant 258308 : i32
      %disable = arith.constant 0 : i32
      %enable = arith.constant -2147483648 : i32
      aiex.npu.write32(%slave, %disable) {column = 0 : i32, row = 2 : i32} : i32, i32
      aiex.npu.write32(%slave, %enable) {column = 0 : i32, row = 2 : i32} : i32, i32

      // Re-arm the cons lock (LOCK0_VALUE, 0x1F000 = 126976) so the send
      // proceeds.
      %la = arith.constant 126976 : i32
      %one = arith.constant 1 : i32
      aiex.npu.write32(%la, %one) {column = 0 : i32, row = 2 : i32} : i32, i32

      aiex.npu.dma_memcpy_nd(%arg0[0,0,0,0][1,1,1,8][0,0,0,1]) {id=0:i64, issue_token=true, metadata=@out0} : memref<8xi32>
      aiex.npu.dma_wait {symbol=@out0}
    }
  }
}
