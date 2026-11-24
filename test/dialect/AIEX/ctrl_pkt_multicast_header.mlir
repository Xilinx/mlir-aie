//===- ctrl_pkt_multicast_header.mlir --------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: split-file %s %t
// RUN: aie-opt %t/routes.mlir -aie-import-overlay -aie-ctrl-packet-to-dma="parallel-shims=true chain-bds=true" | FileCheck %s
// RUN: aie-translate --aie-ctrlpkt-to-bin -aie-output-binary=false %t/routes.mlir | FileCheck %s --check-prefix=BIN
// RUN: aie-opt %t/invalid.mlir --split-input-file -aie-ctrl-packet-to-dma --verify-diagnostics -o /dev/null

//--- routes.mlir

// CHECK:       aiex.dma_configure_task_for @ctrlpkt_col0_mm2s_chan1
// CHECK:       aie.dma_bd({{.*}} offset = 0 len = 6)
// CHECK:       aiex.dma_await_task({{.*}}){{$}}
// CHECK:       aiex.dma_configure_task_for @ctrlpkt_col0_mm2s_chan1
// CHECK:       aie.dma_bd({{.*}} offset = 6 len = 3)
// CHECK:       aiex.dma_await_task({{.*}}){{$}}
// CHECK:       aiex.dma_configure_task_for @ctrlpkt_col0_mm2s_chan0
// CHECK:       aie.dma_bd({{.*}} offset = 9 len = 3)
// CHECK:       aiex.dma_await_task({{.*}}){{$}}
// CHECK:       aiex.dma_configure_task_for @ctrlpkt_col0_mm2s_chan0
// CHECK:       aie.dma_bd({{.*}} offset = 12 len = 3)
// CHECK:       aiex.dma_await_task({{.*}}){{$}}
// CHECK-NOT:   aiex.dma_configure_task_for

// BIN:      0000001C
// BIN-NEXT: 0002C000
// BIN-NEXT: 00000001
// BIN-NEXT: 0000001C
// BIN-NEXT: 8002C004
// BIN-NEXT: 00000002
// BIN-NEXT: 8000001D
// BIN-NEXT: 0002C000
// BIN-NEXT: 00000004
// BIN-NEXT: 8000001E
// BIN-NEXT: 0002C000
// BIN-NEXT: 00000005
// BIN-NEXT: 8000001B
// BIN-NEXT: 0003F004
// BIN-NEXT: 00000003
module {
  aie.device(npu2_1col) @main {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 27>}
    aie.shim_dma_allocation @ctrlpkt_col0_mm2s_chan0 (%tile_0_0, MM2S, 0) {aie.is_ctrl_pkt_overlay}
    aie.runtime_sequence() {
      aiex.set_device @main
      aiex.control_packet_barrier {independent}
      aiex.control_packet {address = 2277376 : ui32, data = array<i32: 1>, multicast_id = 0 : i32, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 2277380 : ui32, data = array<i32: 2>, multicast_id = 0 : i32, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 2277376 : ui32, data = array<i32: 4>, multicast_id = 1 : i32, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 2277376 : ui32, data = array<i32: 5>, multicast_id = 2 : i32, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 2355204 : ui32, data = array<i32: 3>, opcode = 0 : i32, stream_id = 0 : i32}
    }
  } {overlay = @overlay}
  aie.device(npu2_1col) @overlay {
    %shim = aie.tile(0, 0)
    aie.shim_dma_allocation @ctrlpkt_col0_mm2s_chan0 (%shim, MM2S, 0) {aie.is_ctrl_pkt_overlay}
    aie.shim_dma_allocation @ctrlpkt_col0_mm2s_chan1 (%shim, MM2S, 1) {aie.is_ctrl_pkt_overlay}
  } {aie.ctrl_pkt_multicast = [{allocation = @ctrlpkt_col0_mm2s_chan1, packet_id = 28 : i32, tiles = [#aie.tile<0, 2>, #aie.tile<0, 3>]}, {allocation = @ctrlpkt_col0_mm2s_chan1, packet_id = 29 : i32, tiles = [#aie.tile<0, 2>, #aie.tile<0, 4>]}, {allocation = @ctrlpkt_col0_mm2s_chan0, packet_id = 30 : i32, tiles = [#aie.tile<0, 2>, #aie.tile<0, 5>]}], aie.is_ctrl_pkt_overlay}
}

//--- invalid.mlir
module {
  aie.device(npu2_1col) @main {
    aie.runtime_sequence() {
      aiex.set_device @main
      aiex.control_packet {address = 2228224 : ui32, data = array<i32: 1>, multicast_id = 0 : i32, opcode = 0 : i32, stream_id = 0 : i32}
    }
  } {overlay = @overlay}
  // expected-error@+1 {{allocation @missing must reference an MM2S shim DMA allocation in this overlay}}
  aie.device(npu2_1col) @overlay {
  } {aie.ctrl_pkt_multicast = [{allocation = @missing, packet_id = 28 : i32, tiles = [#aie.tile<0, 2>, #aie.tile<0, 3>]}], aie.is_ctrl_pkt_overlay}
}

// -----

module {
  aie.device(npu2_1col) @main {
    aie.runtime_sequence() {
      aiex.set_device @main
      aiex.control_packet {address = 2228224 : ui32, data = array<i32: 1>, multicast_id = 0 : i32, opcode = 0 : i32, stream_id = 0 : i32}
    }
  } {overlay = @overlay}
  // expected-error@+1 {{allocation @receive must reference an MM2S shim DMA allocation in this overlay}}
  aie.device(npu2_1col) @overlay {
    %shim = aie.tile(0, 0)
    aie.shim_dma_allocation @receive(%shim, S2MM, 0) {aie.is_ctrl_pkt_overlay}
  } {aie.ctrl_pkt_multicast = [{allocation = @receive, packet_id = 28 : i32, tiles = [#aie.tile<0, 2>, #aie.tile<0, 3>]}], aie.is_ctrl_pkt_overlay}
}

// -----

module {
  aie.device(npu2_1col) @main {
    aie.runtime_sequence() {
      aiex.set_device @main
      // expected-error@+1 {{multicast allocation @control requires the matching imported shim DMA allocation}}
      aiex.control_packet {address = 2228224 : ui32, data = array<i32: 1>, multicast_id = 0 : i32, opcode = 0 : i32, stream_id = 0 : i32}
    }
  } {overlay = @overlay}
  aie.device(npu2_1col) @overlay {
    %shim = aie.tile(0, 0)
    aie.shim_dma_allocation @control(%shim, MM2S, 0) {aie.is_ctrl_pkt_overlay}
  } {aie.ctrl_pkt_multicast = [{allocation = @control, packet_id = 28 : i32, tiles = [#aie.tile<0, 2>, #aie.tile<0, 3>]}], aie.is_ctrl_pkt_overlay}
}

// -----

module {
  aie.device(npu2_1col) @main {
    aie.runtime_sequence() {
      // expected-error@+1 {{multicast_id 0 has no declaration in the active overlay}}
      aiex.control_packet {address = 2228224 : ui32, data = array<i32: 1>, multicast_id = 0 : i32, opcode = 0 : i32, stream_id = 0 : i32}
    }
  }
}

// -----

module {
  aie.device(npu2_1col) @main {
    aie.runtime_sequence() {
      aiex.set_device @main
      aiex.control_packet {address = 2228224 : ui32, data = array<i32: 1>, multicast_id = 0 : i32, opcode = 0 : i32, stream_id = 0 : i32}
    }
  } {overlay = @overlay}
  // expected-error@+1 {{aie.ctrl_pkt_multicast entry 0 requires allocation, a valid packet_id, and tiles}}
  aie.device(npu2_1col) @overlay {
    %shim = aie.tile(0, 0)
    aie.shim_dma_allocation @control(%shim, MM2S, 0) {aie.is_ctrl_pkt_overlay}
  } {aie.ctrl_pkt_multicast = [{allocation = @control, packet_id = 32 : i32, tiles = [#aie.tile<0, 2>, #aie.tile<0, 3>]}], aie.is_ctrl_pkt_overlay}
}
