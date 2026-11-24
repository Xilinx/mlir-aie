//===- test_place_fifo_channel_demand.mlir ---------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// The sequential placer charges a fifo end a DMA channel only where the split
// gives it a DMA end (plannedDmaEnd), as the SA does: not a core stream port,
// and once per consumer end however many links on one tile read the fifo.

// RUN: aie-opt --split-input-file --aie-place-tiles %s | FileCheck %s
// RUN: aie-opt --split-input-file --aie-place-tiles --aie-objectfifo-split --aie-objectfifo-verify --aie-assign-packet-ids --aie-objectfifo-allocate %s -o /dev/null

// @c reaches the core's stream port, so the core needs S2MM channels only
// for @a and @b, which it has.
// CHECK-LABEL: module @stream_port_end
// CHECK:       aie.objectfifo @c({{.*}}cons_ports = [#aie.end_port<Core : 0>]
module @stream_port_end {
  aie.device(npu2_2col) {
    %s = aie.logical_tile<ShimNOCTile>(?, ?)
    %s2 = aie.logical_tile<ShimNOCTile>(?, ?)
    %t = aie.logical_tile<CoreTile>(?, ?)
    aie.objectfifo @a(%s, {%t}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @b(%s, {%t}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @c(%s2, {%t}, 2 : i32) {cons_ports = [#aie.end_port<Core : 0>]} : !aie.objectfifo<memref<16xi32>>
    aie.core(%t) {
      %x = aie.objectfifo.acquire @a(Consume, 1) : memref<16xi32>
      %y = aie.objectfifo.acquire @b(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @a(Consume, 1)
      aie.objectfifo.release @b(Consume, 1)
      aie.end
    }
  }
}

// -----

// Three links on one mem tile each read one of @in's three consumer ends:
// three S2MM channels there, not nine.
// CHECK-LABEL: module @three_links_read_one_fifo
// CHECK:       %[[M:.*]] = aie.tile({{[0-9]+}}, 1)
// CHECK:       aie.objectfifo @in(%{{.*}}, {%[[M]], %[[M]], %[[M]]}
module @three_links_read_one_fifo {
  aie.device(npu2) {
    %s = aie.logical_tile<ShimNOCTile>(?, ?)
    %m = aie.logical_tile<MemTile>(?, ?)
    %t0 = aie.logical_tile<CoreTile>(?, ?)
    %t1 = aie.logical_tile<CoreTile>(?, ?)
    %t2 = aie.logical_tile<CoreTile>(?, ?)
    aie.objectfifo @in(%s, {%m, %m, %m}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @o0(%m, {%t0}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @o1(%m, {%t1}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @o2(%m, {%t2}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo.link [@in] -> [@o0] ([] [])
    aie.objectfifo.link [@in] -> [@o1] ([] [])
    aie.objectfifo.link [@in] -> [@o2] ([] [])
    aie.core(%t0) {
      %x = aie.objectfifo.acquire @o0(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @o0(Consume, 1)
      aie.end
    }
    aie.core(%t1) {
      %x = aie.objectfifo.acquire @o1(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @o1(Consume, 1)
      aie.end
    }
    aie.core(%t2) {
      %x = aie.objectfifo.acquire @o2(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @o2(Consume, 1)
      aie.end
    }
  }
}
