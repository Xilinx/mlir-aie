//===- write32-no-overlay.mlir ----------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Tests that reset-free (out-of-band direct-write reconfiguration)
// no longer depends on a resident @ctrl_pkt_overlay device: it now resets to
// @empty (like plain write32) instead of preloading the overlay, self-clear
// runs even though useOverlay is false, and the self-clear switch-teardown
// tolerates a module with no @ctrl_pkt_overlay device at all (excludePorts is
// then just empty, so every packet-switch port this config's own connect
// enabled gets disabled -- there is no persistent overlay to carve out of the
// exclude set). Drives with-reset=true to restore the @empty preload this test
// asserts: the reset-free arm defaults to no preload at all, see
// write32-reset-free.mlir.

// RUN: aie-opt --aie-expand-load-pdi="reset-free=true with-reset=true self-clear=true" %s | FileCheck %s

// CHECK-NOT: ctrl_pkt_overlay

module {
  aie.device(npu2_1col) @cfg {
    %shim = aie.tile(0, 0)
    %compute = aie.tile(0, 2)
    // Packet-switched (the device oracle rung, 08-full-reconfig, is
    // packet-switched): one master/slave pair, tagged with no
    // is_ctrl_pkt_overlay attribute since there is no resident overlay to
    // share this port with.
    aie.switchbox(%compute) {
      %amsel = aie.amsel<0> (0)
      %ms = aie.masterset(Core : 0, %amsel)
      aie.packet_rules(South : 0) {
        aie.rule(31, 0, %amsel)
      }
    }
    %buf = aie.buffer(%compute) {address = 1024 : i32, sym_name = "buf"} : memref<16xi32>
    %mem = aie.mem(%compute) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb2)
    ^bb1:
      aie.dma_bd(%buf : memref<16xi32> offset = 0 len = 16) {bd_id = 0 : i32, next_bd_id = 0 : i32}
      aie.next_bd ^bb1
    ^bb2:
      aie.end
    }
  }

  aie.device(npu2_1col) @main {
    aie.runtime_sequence(%arg0: memref<16xi32>) {
      // The preload resets to @empty (the overlay-free reset), not a
      // @ctrl_pkt_overlay load.
      // CHECK: aiex.npu.load_pdi {device_ref = @empty_0, expand_mode = 0 : i32}
      // The config itself lowers to direct write32s/blockwrite (Transaction
      // output), same as plain write32 mode: the DMA BD, then MM2S_0's
      // channel-enable write (2219536).
      // CHECK: aiex.npu.blockwrite
      // CHECK-DAG: %[[V0:.*]] = arith.constant 0 : i32
      // CHECK-DAG: %[[A0:.*]] = arith.constant 2219540 : i32
      // CHECK: aiex.npu.write32(%[[A0]], %[[V0]]) : i32, i32
      // CHECK-DAG: %[[V1:.*]] = arith.constant 1 : i32
      // CHECK-DAG: %[[A1:.*]] = arith.constant 2219536 : i32
      // CHECK: aiex.npu.write32(%[[A1]], %[[V1]]) : i32, i32
      // Packet-switch master/slave port config: masterset (2355200), the
      // packet rule (2355476), and its arbiter/slot config (2355792).
      // CHECK-DAG: %[[V2:.*]] = arith.constant -1073741816 : i32
      // CHECK-DAG: %[[A2:.*]] = arith.constant 2355200 : i32
      // CHECK: aiex.npu.write32(%[[A2]], %[[V2]]) : i32, i32
      // CHECK-DAG: %[[V3:.*]] = arith.constant -1073741824 : i32
      // CHECK-DAG: %[[A3:.*]] = arith.constant 2355476 : i32
      // CHECK: aiex.npu.write32(%[[A3]], %[[V3]]) : i32, i32
      // CHECK-DAG: %[[V4:.*]] = arith.constant 2031872 : i32
      // CHECK-DAG: %[[A4:.*]] = arith.constant 2355792 : i32
      // CHECK: aiex.npu.write32(%[[A4]], %[[V4]]) : i32, i32
      // Self-clear switch-teardown: with no overlay to exclude, the config's
      // own master/slave port is disabled (reset to 0) once its DMA transfer
      // completes -- the same two addresses (2355200, 2355476) written again.
      // CHECK-DAG: %[[V5:.*]] = arith.constant 0 : i32
      // CHECK-DAG: %[[A5:.*]] = arith.constant 2355200 : i32
      // CHECK: aiex.npu.write32(%[[A5]], %[[V5]]) : i32, i32
      // CHECK-DAG: %[[V6:.*]] = arith.constant 0 : i32
      // CHECK-DAG: %[[A6:.*]] = arith.constant 2355476 : i32
      // CHECK: aiex.npu.write32(%[[A6]], %[[V6]]) : i32, i32
      // Self-clear DMA teardown: MM2S_0 (2219536) reset assert (value=2,
      // mask=2 -- both operands are the literal 2, so left unbound here) then
      // deassert (value=0, mask=2).
      // CHECK-DAG: %[[A7:.*]] = arith.constant 2219536 : i32
      // CHECK: aiex.npu.maskwrite32(%[[A7]], %{{.*}}, %{{.*}}) : i32, i32, i32
      // CHECK-DAG: %[[V8:.*]] = arith.constant 0 : i32
      // CHECK-DAG: %[[M8:.*]] = arith.constant 2 : i32
      // CHECK-DAG: %[[A8:.*]] = arith.constant 2219536 : i32
      // CHECK: aiex.npu.maskwrite32(%[[A8]], %[[V8]], %[[M8]]) : i32, i32, i32
      aiex.npu.load_pdi { device_ref = @cfg }
    }
  }
}
