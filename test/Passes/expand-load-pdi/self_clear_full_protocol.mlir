//===- self_clear_full_protocol.mlir ---------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Folds the intents of the now-deleted self_clear_dma.mlir / self_clear_dma_dmaop.mlir /
// self_clear_dma_scope.mlir: self-clear (now unconditional for ctrlpkt/write32) used to emit only
// the packet-switch port teardown, with the circuit-connect and DMA-channel-reset teardowns gated
// behind separate sub-flags. Those sub-flags are gone: self-clear=true alone now emits the
// COMPLETE per-config teardown protocol
// (switch + circuit + DMA) in one pass. This config exercises all three at once -- a packet-switch
// master/slave port (Core master, South slave, like write32-no-overlay.mlir), a circuit-switch
// connect (DMA source to North, like switchbox_config.mlir), and an MM2S DMA channel expressed via
// the region form (aie.dma / DMAOp, like self_clear_dma_dmaop.mlir) on a core-less compute tile --
// so a regression that drops any one of the three teardowns fails a CHECK below.

// RUN: aie-opt --aie-expand-load-pdi="ctrl-pkt=true self-clear=true" %s | FileCheck %s

module {
  aie.device(npu2_1col) @ctrl_pkt_overlay {
    // No switchbox at all: the overlay owns no ports, so every port @cfg's
    // switchbox enables is exclusively-data and self-clear must disable all
    // of it (excludePorts stays empty).
    %o = aie.tile(0, 2)
  }
  aie.device(npu2_1col) @cfg {
    %shim = aie.tile(0, 0)
    %compute = aie.tile(0, 2)
    aie.switchbox(%compute) {
      // Packet-switch master/slave port pair (Core master, South slave).
      %amsel = aie.amsel<0> (0)
      %ms = aie.masterset(Core : 0, %amsel)
      aie.packet_rules(South : 0) {
        aie.rule(31, 0, %amsel)
      }
      // Circuit-switch connect (DMA source to North dest): a distinct
      // register pair from the packet-switch port above, so its disable is
      // independently observable.
      aie.connect<DMA : 0, North : 0>
    }
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
      // The resident-overlay preload, then @cfg's own configuration (DMA BD
      // setup + channel enable, circuit connect enable, packet master/slave
      // port + arbiter/slot enable) as control packets.
      // CHECK: aiex.npu.load_pdi {device_ref = @ctrl_pkt_overlay
      // MM2S_0 channel enable (2219536).
      // CHECK: aiex.control_packet {address = 2219536 : ui32, data = array<i32: 1>
      // Circuit connect enable: source-side (2355252) and dest-side (2355460).
      // CHECK: aiex.control_packet {address = 2355252 : ui32
      // CHECK: aiex.control_packet {address = 2355460 : ui32
      // Packet master/slave port enable: masterset (2355200) and packet rule (2355476).
      // CHECK: aiex.control_packet {address = 2355200 : ui32, data = array<i32: -1073741816>
      // CHECK: aiex.control_packet {address = 2355476 : ui32, data = array<i32: -1073741824>

      // Self-clear teardown, all three classes, each demand-scoped to exactly
      // the resource @cfg used above (order: switch disable -- packet ports
      // then circuit connects, mirroring disableDataSwitches -- then DMA
      // channel reset):
      //
      // Switch teardown: the packet master/slave port disabled (reset to 0),
      // proving the packet-switch class of the merged protocol fires.
      // CHECK-DAG: aiex.control_packet {address = 2355200 : ui32, data = array<i32: 0>
      // CHECK-DAG: aiex.control_packet {address = 2355476 : ui32, data = array<i32: 0>
      //
      // Circuit teardown (now part of unconditional self-clear for ctrlpkt/write32):
      // the circuit connect disabled (reset to 0) at its own address pair, distinct
      // from the packet port above.
      // CHECK-DAG: aiex.control_packet {address = 2355252 : ui32, data = array<i32: 0>
      // CHECK-DAG: aiex.control_packet {address = 2355460 : ui32, data = array<i32: 0>
      //
      // DMA teardown (now part of unconditional self-clear for ctrlpkt/write32):
      // MM2S_0 (2219536) reset assert then deassert, folded via read-modify-write
      // against the channel-enable write above (data:1). The reset bit (mask 2)
      // is pulsed while the channel enable bit (1) is preserved, not zeroed:
      //   assert   = (cur=1 & ~2) | (2 & 2) = 3
      //   deassert = (cur=3 & ~2) | (0 & 2) = 1
      // CHECK: aiex.control_packet {address = 2219536 : ui32, data = array<i32: 3>
      // CHECK: aiex.control_packet {address = 2219536 : ui32, data = array<i32: 1>
      aiex.npu.load_pdi { device_ref = @cfg }
    }
  }
}
