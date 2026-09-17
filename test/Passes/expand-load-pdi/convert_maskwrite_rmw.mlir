//===- convert_maskwrite_rmw.mlir -------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Regression guard for the control-packet maskwrite read-modify-write fold
// (emitControlPacketOps / seedRegState in AIEToConfiguration.cpp). A control
// packet writes the WHOLE 32-bit word -- the hardware has no masked
// (read-modify-write) control packet -- so a MASKWRITE txn op must be folded to
// a full write of the resulting value, resolved against the register state left
// by prior control packets in the same runtime-sequence block. The DMA-channel
// self-clear teardown is the production trigger: the config writes the channel
// CTRL register full (enable bit set), then the reset pulse maskwrites only the
// reset bit. Folding the maskwrite against the seeded state must PRESERVE the
// enable bit through the pulse; dropping the mask (emitting the raw masked
// value) would zero it and corrupt on-device behavior while leaving the emitted
// register set value-identical to a plain write. This test pins the folded
// values so a regression that drops the mask fails a CHECK below.
//
// Minimal design: one MM2S DMA channel on a core-less compute tile, no
// switchbox, so the only register touched more than once is the channel CTRL
// (2219536) -- the config enable write followed by the reset assert/deassert
// maskwrite pulse.

// RUN: aie-opt --aie-expand-load-pdi="ctrl-pkt=true self-clear=true" %s | FileCheck %s

module {
  aie.device(npu2_1col) @ctrl_pkt_overlay {
    %o = aie.tile(0, 2)
  }
  aie.device(npu2_1col) @cfg {
    %compute = aie.tile(0, 2)
    %buf = aie.buffer(%compute) {address = 1024 : i32, sym_name = "buf"} : memref<16xi32>
    %lock = aie.lock(%compute, 0) {init = 1 : i32}
    %mem = aie.mem(%compute) {
      %0 = aie.dma(MM2S, 0) [{
        %c1 = arith.constant 1 : i32
        aie.use_lock(%lock, AcquireGreaterEqual, %c1)
        aie.dma_bd(%buf : memref<16xi32> offset = 0 len = 16) {bd_id = 0 : i32}
        aie.use_lock(%lock, Release, %c1)
      }]
      aie.end
    }
  }
  aie.device(npu2_1col) @main {
    aie.runtime_sequence(%arg0: memref<16xi32>) {
      // Config: MM2S_0 channel CTRL (2219536) written full with the enable bit
      // (bit 0 = 1). This full write seeds the running register state.
      // CHECK: aiex.control_packet {address = 2219536 : ui32, data = array<i32: 1>
      //
      // Self-clear DMA reset pulse on the SAME register, folded via
      // read-modify-write against the seed:
      //   assert   = (cur=1 & ~mask=2) | (val=2 & mask=2) = 1 | 2 = 3  (enable bit preserved)
      //   deassert = (cur=3 & ~mask=2) | (val=0 & mask=2) = 1 | 0 = 1  (enable bit preserved)
      // A mask-dropping regression would emit 2 then 0 here, zeroing the enable
      // bit -- these CHECKs fail if that happens.
      // CHECK: aiex.control_packet {address = 2219536 : ui32, data = array<i32: 3>
      // CHECK: aiex.control_packet {address = 2219536 : ui32, data = array<i32: 1>
      aiex.npu.load_pdi { device_ref = @cfg }
    }
  }
}
