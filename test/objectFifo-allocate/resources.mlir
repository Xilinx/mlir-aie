// RUN: aie-opt --aie-objectfifo-allocate %s | FileCheck %s
// RUN: aie-opt --aie-objectfifo-allocate --mlir-print-debuginfo %s | FileCheck %s --check-prefix=LOC --implicit-check-not='loc(unknown)'
// RUN: aie-opt --aie-objectfifo-allocate %s -o %t1.mlir
// RUN: aie-opt --aie-objectfifo-allocate %t1.mlir -o %t2.mlir
// RUN: diff %t1.mlir %t2.mlir

// Copyright (C) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Buffers for each object, a counting lock pair for each segment, a DMA channel
// for each endpoint, and the flow between them.

module @resources {
  aie.device(xcve2302) {
    %tile12 = aie.tile(1, 2)
    %tile33 = aie.tile(3, 3)

    aie.objectfifo.pool @prod__pool(%tile12) {
      depth = 2 : i32
    } : memref<16xi32> {
      aie.objectfifo.segment @s0 {offset = 0 : i32, size = 16 : i32}
    } loc("fifo_user.py":10:2)
    aie.objectfifo.dma_endpoint @prod_dma(%tile12) drains @prod__pool

    aie.objectfifo.pool @cons__pool(%tile33) {
      depth = 3 : i32
    } : memref<16xi32> {
      aie.objectfifo.segment @s0 {offset = 0 : i32, size = 16 : i32}
    }
    aie.objectfifo.dma_endpoint @cons_dma(%tile33) fills @cons__pool

    aie.route from @prod_dma to [@cons_dma] loc("fifo_user.py":20:4)
  }
}

// CHECK-LABEL: @resources
// CHECK-DAG:   %[[T12:.*]] = aie.tile(1, 2)
// CHECK-DAG:   %[[T33:.*]] = aie.tile(3, 3)
// CHECK-DAG:   aie.buffer(%[[T12]]) {sym_name = "prod__buf0"} : memref<16xi32>
// CHECK-DAG:   aie.buffer(%[[T12]]) {sym_name = "prod__buf1"} : memref<16xi32>
// CHECK-DAG:   aie.buffer(%[[T33]]) {sym_name = "cons__buf0"} : memref<16xi32>
// CHECK-DAG:   aie.buffer(%[[T33]]) {sym_name = "cons__buf1"} : memref<16xi32>
// CHECK-DAG:   aie.buffer(%[[T33]]) {sym_name = "cons__buf2"} : memref<16xi32>

// The producer's lock counts free objects, the consumer's counts full ones.
// CHECK-DAG:   aie.lock(%[[T12]]) {init = 2 : i32, sym_name = "prod__prodlock0"}
// CHECK-DAG:   aie.lock(%[[T12]]) {init = 0 : i32, sym_name = "prod__conslock0"}
// CHECK-DAG:   aie.lock(%[[T33]]) {init = 3 : i32, sym_name = "cons__prodlock0"}
// CHECK-DAG:   aie.lock(%[[T33]]) {init = 0 : i32, sym_name = "cons__conslock0"}

// CHECK:   aie.objectfifo.pool @prod__pool({{.*}}) {buffers = [@prod__buf0, @prod__buf1], depth = 2 : i32}
// CHECK: aie.objectfifo.segment @s0 {consumeLock = @prod__conslock0, offset = 0 : i32, produceLock = @prod__prodlock0, size = 16 : i32}
// CHECK:   aie.objectfifo.dma_endpoint @prod_dma({{.*}}) drains @prod__pool {channelIndex = 0 : i32}
// CHECK:   aie.objectfifo.pool @cons__pool({{.*}}) {buffers = [@cons__buf0, @cons__buf1, @cons__buf2], depth = 3 : i32}
// CHECK: aie.objectfifo.segment @s0 {consumeLock = @cons__conslock0, offset = 0 : i32, produceLock = @cons__prodlock0, size = 16 : i32}
// CHECK:   aie.objectfifo.dma_endpoint @cons_dma({{.*}}) fills @cons__pool {channelIndex = 0 : i32}
// CHECK:   aie.flow(%[[T12]], DMA : 0, %[[T33]], DMA : 0)
// CHECK-NOT: aie.route

// The modular allocation pass must retain pool/route provenance, not replace it
// with a tile or device location. The implicit negative check covers every op.
// LOC-DAG: aie.buffer({{.*}}) {sym_name = "prod__buf0"} : memref<16xi32> loc(#[[POOL:loc[0-9]*]])
// LOC-DAG: aie.buffer({{.*}}) {sym_name = "prod__buf1"} : memref<16xi32> loc(#[[POOL]])
// LOC-DAG: aie.lock({{.*}}) {init = 2 : i32, sym_name = "prod__prodlock0"} loc(#[[POOL]])
// LOC-DAG: aie.flow({{.*}}) loc(#[[ROUTE:loc[0-9]*]])
// LOC-DAG: #[[POOL]] = loc("fifo_user.py":10:2)
// LOC-DAG: #[[ROUTE]] = loc("fifo_user.py":20:4)
