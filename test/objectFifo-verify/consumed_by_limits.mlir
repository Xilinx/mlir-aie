//===- consumed_by_limits.mlir ----------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Where the bytes an output receives are not all the input's, or not as many
// as the pools' objects hold, the proof must say no more than the hardware
// guarantees.

// RUN: aie-opt --split-input-file --aie-objectfifo-split --aie-objectfifo-verify %s | FileCheck %s

// Two packet routes merge into @host, so what it receives may all be @in2's,
// relayed through the memtile: neither input is proven consumed.

// CHECK-LABEL: module @merged_output
// CHECK-NOT: consumed_by
module @merged_output {
  aie.device(npu2) {
    %shim = aie.tile(0, 0)
    %shim1 = aie.tile(1, 0)
    %mt = aie.tile(0, 1)
    aie.route_endpoint @in1(%shim) DMA
    aie.route_endpoint @in2(%shim) DMA
    aie.route_endpoint @host(%shim1) DMA
    aie.objectfifo.pool @p(%mt) {depth = 2 : i32} : memref<16xi32> {
      aie.objectfifo.segment @s0 {offset = 0 : i32, size = 16 : i32}
    }
    aie.objectfifo.dma_endpoint @fill(%mt) fills @p
    aie.objectfifo.dma_endpoint @drain(%mt) drains @p
    aie.route from @in2 to [@fill]
    aie.route from @in1 to [@host] {packet = #aie.packet_info<pkt_id = 1>}
    aie.route from @drain to [@host] {packet = #aie.packet_info<pkt_id = 2>}
  }
}

// -----

// The output keeps each packet's header, so it receives more bytes than the
// input sent.

// CHECK-LABEL: module @kept_header
// CHECK-NOT: consumed_by
module @kept_header {
  aie.device(npu2) {
    %shim = aie.tile(0, 0)
    %shim1 = aie.tile(1, 0)
    aie.route_endpoint @in(%shim) DMA
    aie.route_endpoint @host(%shim1) DMA
    aie.route from @in to [@host] {keep_pkt_header, packet = #aie.packet_info<pkt_id = 1>}
  }
}

