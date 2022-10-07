// RUN: phy-opt --convert-layout-to-physical="device=aie" %s | FileCheck %s
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

func.func private @kernel(%Q1: !spatial.queue<memref<1024xi32>>, %Q2: !spatial.queue<memref<1024xi32>>) {
  cf.br ^bb
^bb:
  cf.br ^bb
}

%Q1 = spatial.queue<1>(): !spatial.queue<memref<1024xi32>>
%Q2 = spatial.queue<1>(): !spatial.queue<memref<1024xi32>>
%node = spatial.node @kernel(%Q1, %Q2): (!spatial.queue<memref<1024xi32>>, !spatial.queue<memref<1024xi32>>) -> !spatial.node

layout.platform<"vck190"> {
  layout.device<"aie"> {

    // CHECK-DAG: %[[OSTREAM:.*]], %[[ISTREAM:.*]] = physical.stream<[0, 1]> () {aie.id = "0", aie.port = "DMA.I", aie.tile = "6.3"} : (!physical.ostream<i32>, !physical.istream<i32>)

    // CHECK-DAG: %[[BUFFER_1:.*]] = physical.buffer() {aie.id = "1", aie.tile = "6.3"} : memref<1024xi32>
    // CHECK-DAG: %[[BUFFER_2:.*]] = physical.buffer() {aie.id = "2", aie.tile = "6.3"} : memref<1024xi32>

    // CHECK-DAG: %[[LOCK_1:.*]] = physical.lock<0> () {aie.id = "1", aie.tile = "6.3"}
    // CHECK-DAG: %[[LOCK_2:.*]] = physical.lock<0> () {aie.id = "2", aie.tile = "6.3"}

    // CHECK-DAG: physical.stream_dma(%[[ISTREAM]] : !physical.istream<i32>) {
    // CHECK-DAG:   %[[CONNECT1:.*]] = physical.stream_dma_connect (%[[LOCK_1]][0 -> 1], %[[BUFFER_1]][0 : 1024] : memref<1024xi32>, %[[CONNECT2:.*]]) 
    // CHECK-DAG:   %[[CONNECT2]] = physical.stream_dma_connect (%[[LOCK_2]][0 -> 1], %[[BUFFER_2]][0 : 1024] : memref<1024xi32>, %[[CONNECT1]]) 
    // CHECK-DAG: } {aie.engine = "S2MM", aie.id = "0", aie.tile = "6.3"}

    layout.route<["tile/6.3/port/DMA.I/id/0/stream",
                  "tile/6.3/engine/S2MM/id/0/stream_dma",
                  "tile/6.3/id/1/buffer,tile/6.3/id/1/lock"]>
                  (%Q1: !spatial.queue<memref<1024xi32>> -> %node: !spatial.node)

    layout.route<["tile/6.3/port/DMA.I/id/0/stream",
                  "tile/6.3/engine/S2MM/id/0/stream_dma",
                  "tile/6.3/id/2/buffer,tile/6.3/id/2/lock"]>
                  (%Q2: !spatial.queue<memref<1024xi32>> -> %node: !spatial.node)

  }
}
