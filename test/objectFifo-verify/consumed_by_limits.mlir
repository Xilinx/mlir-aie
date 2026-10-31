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

// -----

// The core tile's DMA fills half of each `in` object, so the input sends 32
// bytes for every 64 the output receives.

// CHECK-LABEL: module @fill_half
// CHECK: aie.route_endpoint @in_prod_dma({{.*}}) DMA {consumed_by = [#aie.consumed_by<output = @out_cons_dma, input_bytes = 1, output_bytes = 2>], fifoName = "in"}
module @fill_half {
  aie.device(npu2_1col) {
    %shim = aie.tile(0, 0)
    %core = aie.tile(0, 2)
    aie.objectfifo @in(%shim, {%core dimensionsFromStream [<size = 8, stride = 1>]}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @out(%core, {%shim}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.core(%core) {
      scf.while : () -> () {
        %true = arith.constant true
        scf.condition(%true)
      } do {
        %i = aie.objectfifo.acquire @in(Consume, 1) : memref<16xi32>
        %o = aie.objectfifo.acquire @out(Produce, 1) : memref<16xi32>
        aie.objectfifo.release @in(Consume, 1)
        aie.objectfifo.release @out(Produce, 1)
        scf.yield
      }
      aie.end
    }
  }
}

// -----

// The core tile's DMA sends each `out` object twice, so the output receives
// 128 bytes for every 64 the input sends.

// CHECK-LABEL: module @drain_twice
// CHECK: aie.route_endpoint @in_prod_dma({{.*}}) DMA {consumed_by = [#aie.consumed_by<output = @out_cons_dma, input_bytes = 1, output_bytes = 2>], fifoName = "in"}
module @drain_twice {
  aie.device(npu2_1col) {
    %shim = aie.tile(0, 0)
    %core = aie.tile(0, 2)
    aie.objectfifo @in(%shim, {%core}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @out(%core dimensionsToStream [<size = 2, stride = 0>, <size = 16, stride = 1>], {%shim}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.core(%core) {
      scf.while : () -> () {
        %true = arith.constant true
        scf.condition(%true)
      } do {
        %i = aie.objectfifo.acquire @in(Consume, 1) : memref<16xi32>
        %o = aie.objectfifo.acquire @out(Produce, 1) : memref<16xi32>
        aie.objectfifo.release @in(Consume, 1)
        aie.objectfifo.release @out(Produce, 1)
        scf.yield
      }
      aie.end
    }
  }
}

