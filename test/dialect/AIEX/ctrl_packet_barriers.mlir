// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt %s -aie-reorder-ctrl-packets-by-tile -aie-dedup-ctrl-packet-multicast -aie-coalesce-control-packets | FileCheck %s
// RUN: aie-opt %s -aie-reorder-ctrl-packets-by-tile -aie-dedup-ctrl-packet-multicast -aie-dedup-ctrl-packet-multicast -aie-coalesce-control-packets | FileCheck %s
// RUN: aie-opt %s -aie-reorder-ctrl-packets-by-tile="address-order=false" | FileCheck %s --check-prefix=DSE
// RUN: aie-opt %s -aie-reorder-ctrl-packets-by-tile="eliminate-dead-writes=false" | FileCheck %s --check-prefix=SORT
// RUN: aie-opt %s -aie-ctrl-packet-to-dma="parallel-shims=true chain-bds=true" | FileCheck %s --check-prefix=DMA
// RUN: aie-opt %s -aie-reorder-ctrl-packets-by-tile="address-order=false eliminate-dead-writes=false" | FileCheck %s --check-prefix=EXACT

module {
  aie.device(npu2) @overlay {
  } {aie.ctrl_pkt_multicast = [{column = 0 : i32, id = 28 : i32, tiles = [#aie.tile<0, 2>, #aie.tile<0, 3>]}], aie.is_ctrl_pkt_overlay}
  aie.device(npu2) @cfg {
  } {overlay = @overlay}
  aie.device(npu2) @main {
    %shim0 = aie.tile(0, 0)
    %shim1 = aie.tile(1, 0)
    aie.shim_dma_allocation @ctrl0(%shim0, MM2S, 0) {aie.is_ctrl_pkt_overlay}
    aie.shim_dma_allocation @ctrl1(%shim1, MM2S, 0) {aie.is_ctrl_pkt_overlay}

    // CHECK-LABEL: @storage
    // CHECK: address = 2228224 : ui32, data = array<i32: 1, 7, 8, 4>
    // CHECK-NEXT: aiex.control_packet_barrier
    // CHECK-NEXT: aiex.control_packet {address = 2228224 : ui32, data = array<i32: 10>
    // DSE-LABEL: @storage
    // DSE: address = 2228236 : ui32, data = array<i32: 4>
    // DSE-NEXT: aiex.control_packet {address = 2228224 : ui32, data = array<i32: 1>
    // DSE-NEXT: aiex.control_packet {address = 2228228 : ui32, data = array<i32: 7, 8>
    // SORT-LABEL: @storage
    // SORT: address = 2228224 : ui32, data = array<i32: 1, 2, 3>
    // SORT-NEXT: aiex.control_packet {address = 2228236 : ui32, data = array<i32: 4>
    // SORT-NEXT: aiex.control_packet {address = 2228228 : ui32, data = array<i32: 7, 8>
    aie.runtime_sequence @storage() {
      aiex.control_packet_barrier {compiler_only}
      aiex.control_packet {address = 2228236 : ui32, data = array<i32: 4>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 2228224 : ui32, data = array<i32: 1, 2, 3>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 2228228 : ui32, data = array<i32: 7, 8>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet_barrier
      aiex.control_packet {address = 2228224 : ui32, data = array<i32: 9>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 2228224 : ui32, data = array<i32: 10>, opcode = 0 : i32, stream_id = 0 : i32}
    }

    // CHECK-LABEL: @actions
    // CHECK: address = 2228224 : ui32, data = array<i32: 2>
    // CHECK-NEXT: aiex.control_packet {address = 2301952 : ui32, data = array<i32: 1>
    // CHECK-NEXT: aiex.control_packet {address = 2301952 : ui32, data = array<i32: 1>
    // CHECK-NEXT: aiex.control_packet {address = 2228224 : ui32, data = array<i32: 4>
    aie.runtime_sequence @actions() {
      aiex.control_packet_barrier {compiler_only}
      aiex.control_packet {address = 2228224 : ui32, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 2228224 : ui32, data = array<i32: 2>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 2301952 : ui32, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 2301952 : ui32, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 2228224 : ui32, data = array<i32: 3>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 2228224 : ui32, data = array<i32: 4>, opcode = 0 : i32, stream_id = 0 : i32}
    }

    // CHECK-LABEL: @multicast
    // CHECK: address = 2228224 : ui32, data = array<i32: 7, 9>, multicast_id = 28 : i32
    // CHECK-NEXT: aiex.control_packet_barrier
    // CHECK-NEXT: aiex.control_packet {address = 2228224 : ui32, data = array<i32: 2>, multicast_id = 28 : i32
    aie.runtime_sequence @multicast() {
      aiex.set_device @cfg
      aiex.control_packet_barrier {compiler_only}
      aiex.control_packet {address = 2228224 : ui32, data = array<i32: 7>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 2228228 : ui32, data = array<i32: 9>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 3276800 : ui32, data = array<i32: 7>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 3276804 : ui32, data = array<i32: 9>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet_barrier
      aiex.control_packet {address = 2228224 : ui32, data = array<i32: 2>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 3276800 : ui32, data = array<i32: 2>, opcode = 0 : i32, stream_id = 0 : i32}
    }

    // DMA-LABEL: @multicast_disjoint_destinations
    // DMA: aiex.dma_await_task({{.*}}) {defer_completion}
    // DMA: aiex.dma_await_task({{.*}}){{$}}
    // DMA: aiex.dma_await_task({{.*}}){{$}}
    // DMA: aiex.dma_await_task({{.*}}){{$}}
    aie.runtime_sequence @multicast_disjoint_destinations() {
      aiex.set_device @cfg
      aiex.control_packet_barrier
      aiex.control_packet {address = 2228224 : ui32, data = array<i32: 7>, multicast_id = 28 : i32, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 35782656 : ui32, data = array<i32: 9>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet_barrier
      aiex.control_packet {address = 2228224 : ui32, data = array<i32: 7>, multicast_id = 28 : i32, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 3276800 : ui32, data = array<i32: 9>, opcode = 0 : i32, stream_id = 0 : i32}
    }

    // DMA-LABEL: @ordered_destinations
    // DMA: aiex.dma_await_task({{.*}}){{$}}
    // DMA: aiex.dma_await_task({{.*}}){{$}}
    aie.runtime_sequence @ordered_destinations() {
      aiex.control_packet {address = 2301952 : ui32, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 3350528 : ui32, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
    }

    // CHECK-LABEL: @multicast_actions
    // CHECK: address = 2301952 : ui32, data = array<i32: 2>, multicast_id = 28 : i32
    // CHECK-NEXT: aiex.control_packet {address = 2301952 : ui32, data = array<i32: 0>, multicast_id = 28 : i32
    // CHECK-NEXT: {{^ *}}}
    aie.runtime_sequence @multicast_actions() {
      aiex.set_device @cfg
      aiex.control_packet_barrier {independent}
      aiex.control_packet {address = 2301952 : ui32, data = array<i32: 2>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 2301952 : ui32, data = array<i32: 0>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 3350528 : ui32, data = array<i32: 2>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 3350528 : ui32, data = array<i32: 0>, opcode = 0 : i32, stream_id = 0 : i32}
    }

    // CHECK-LABEL: @independent
    // CHECK: address = 2301952 : ui32, data = array<i32: 1>
    // CHECK-NEXT: aiex.control_packet {address = 2301952 : ui32, data = array<i32: 0>
    // CHECK-NEXT: aiex.control_packet {address = 35856384 : ui32, data = array<i32: 1>
    // CHECK-NEXT: aiex.control_packet {address = 35856384 : ui32, data = array<i32: 0>
    // DMA-LABEL: @independent
    // DMA: aiex.dma_await_task({{.*}}) {defer_completion}
    // DMA: aiex.dma_await_task({{.*}}) {defer_completion}
    // DMA: aiex.dma_await_task({{.*}}) {defer_completion}
    // DMA: aiex.dma_await_task({{.*}}){{$}}
    // DMA: aiex.dma_await_task({{.*}}){{$}}
    // DMA: aiex.dma_await_task({{.*}}){{$}}
    aie.runtime_sequence @independent() {
      aiex.control_packet_barrier {independent}
      aiex.control_packet {address = 35856384 : ui32, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 2301952 : ui32, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 35856384 : ui32, data = array<i32: 0>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 2301952 : ui32, data = array<i32: 0>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet_barrier
      aiex.control_packet {address = 2301952 : ui32, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 35856384 : ui32, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
    }

    // CHECK-LABEL: @ordinary_writes
    // CHECK: aiex.control_packet {address = 2228224 : ui32, data = array<i32: 3, 2>
    // CHECK-NEXT: aiex.control_packet {address = 2301952 : ui32, data = array<i32: 1>
    // CHECK-NEXT: aiex.control_packet {address = 2228224 : ui32, data = array<i32: 5>
    // CHECK-NEXT: aiex.control_packet {address = 2301952 : ui32, data = array<i32: 1>
    // CHECK-NEXT: aiex.control_packet {address = 2228224 : ui32, data = array<i32: 6>
    // CHECK-NEXT: {{^ *}}}
    // EXACT-LABEL: @ordinary_writes
    // EXACT-NEXT: aiex.control_packet {address = 2228228 : ui32, data = array<i32: 2>
    // EXACT-NEXT: aiex.control_packet {address = 2228224 : ui32, data = array<i32: 1>
    // EXACT-NEXT: aiex.control_packet {address = 2228224 : ui32, data = array<i32: 3>
    // EXACT-NEXT: aiex.control_packet {address = 2301952 : ui32, data = array<i32: 1>
    // EXACT-NEXT: aiex.control_packet {address = 2228224 : ui32, data = array<i32: 4>
    // EXACT-NEXT: aiex.control_packet {address = 2228224 : ui32, data = array<i32: 5>
    // EXACT-NEXT: aiex.control_packet {address = 2301952 : ui32, data = array<i32: 1>
    // EXACT-NEXT: aiex.control_packet {address = 2228224 : ui32, data = array<i32: 6>
    // EXACT-NEXT: {{^ *}}}
    aie.runtime_sequence @ordinary_writes() {
      aiex.control_packet {address = 2228228 : ui32, data = array<i32: 2>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 2228224 : ui32, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 2228224 : ui32, data = array<i32: 3>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 2301952 : ui32, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 2228224 : ui32, data = array<i32: 4>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 2228224 : ui32, data = array<i32: 5>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 2301952 : ui32, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 2228224 : ui32, data = array<i32: 6>, opcode = 0 : i32, stream_id = 0 : i32}
    }

    // DMA-LABEL: @ordinary_action_boundary
    // DMA: aiex.dma_await_task({{.*}}){{$}}
    // DMA: aiex.dma_await_task({{.*}}){{$}}
    // DMA: aiex.dma_await_task({{.*}}){{$}}
    aie.runtime_sequence @ordinary_action_boundary() {
      aiex.control_packet {address = 35782656 : ui32, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 2301952 : ui32, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 35782656 : ui32, data = array<i32: 2>, opcode = 0 : i32, stream_id = 0 : i32}
    }

    // DMA-LABEL: @same_destination_action
    // DMA: aie.dma_bd({{.*}} offset = 0 len = 9)
    // DMA: aiex.dma_start_task
    // DMA-NEXT: aiex.dma_await_task({{.*}}){{$}}
    // DMA-NEXT: {{^ *}}}
    aie.runtime_sequence @same_destination_action() {
      aiex.control_packet_barrier {independent}
      aiex.control_packet {address = 2228224 : ui32, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 2301952 : ui32, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 2228224 : ui32, data = array<i32: 2>, opcode = 0 : i32, stream_id = 0 : i32}
    }

    // DMA-LABEL: @mixed_batch_boundary
    // DMA: aie.dma_bd({{.*}} offset = 0 len = 3)
    // DMA: aiex.dma_start_task
    // DMA-NEXT: aiex.dma_await_task({{.*}}){{$}}
    // DMA: aie.dma_bd({{.*}} offset = 3 len = 6)
    // DMA: aiex.dma_start_task
    // DMA-NEXT: aiex.dma_await_task({{.*}}){{$}}
    // DMA-NEXT: {{^ *}}}
    aie.runtime_sequence @mixed_batch_boundary() {
      aiex.control_packet {address = 2228224 : ui32, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 35782656 : ui32, data = array<i32: 2>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 35856384 : ui32, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
    }
  }
}