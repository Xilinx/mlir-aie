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
      // Re-enable the DMA:0 slave port (Stream_Switch_Slave_DMA_0_Config,
      // 0x3F104, Slave_Enable at bit 31 = 0x80000000), re-establishing the route
      // the previous dispatch tore down at its end (see the disable below). On
      // the first dispatch the port is still enabled from CDO load, so this is a
      // no-op; from the second dispatch on it genuinely restores a disabled port
      // -- omit it and the MM2S has nowhere to stream and the collect hangs.
      //
      // Protocol: aie-rt XAie_StrmConnCctEnable / XAie_StrmConnCctDisable (both a
      // full-word write32, which this test uses; see driver/src/stream_switch/xaie_ss.c).
      // DEVIATION: those routines rewrite BOTH the master-port and slave-port
      // config registers; this test writes only the slave register, because the
      // master (South) route is resident from the CDO load, so rewriting it to the
      // same value would be an idempotent no-op. Touching only the slave port also
      // isolates any failure to that one port.
      %slave = arith.constant 258308 : i32
      %enable = arith.constant -2147483648 : i32
      aiex.npu.write32(%slave, %enable) {column = 0 : i32, row = 2 : i32} : i32, i32

      // Re-arm the cons lock (LOCK0_VALUE, 0x1F000 = 126976; aie-rt
      // XAie_LockSetValue, a full-word write32) so the send proceeds.
      %la = arith.constant 126976 : i32
      %one = arith.constant 1 : i32
      aiex.npu.write32(%la, %one) {column = 0 : i32, row = 2 : i32} : i32, i32

      aiex.npu.dma_memcpy_nd(%arg0[0,0,0,0][1,1,1,8][0,0,0,1]) {id=0:i64, issue_token=true, metadata=@out0} : memref<8xi32>
      aiex.npu.dma_wait {symbol=@out0}

      // Tear the DMA:0 slave port down (Slave_Enable -> 0; XAie_StrmConnCctDisable,
      // slave register only, per the deviation noted above) now that the send is
      // done and the channel is quiescent again (stalled on cons). The config
      // register is sticky, so the port stays disabled until the next dispatch's
      // re-enable above -- which is what makes that re-enable load-bearing rather
      // than a no-op net of an adjacent disable.
      %disable = arith.constant 0 : i32
      aiex.npu.write32(%slave, %disable) {column = 0 : i32, row = 2 : i32} : i32, i32
    }
  }
}
