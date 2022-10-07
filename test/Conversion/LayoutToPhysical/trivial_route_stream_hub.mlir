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
 
    // CHECK-DAG: %[[OSTREAM0:.*]], %[[ISTREAM0:.*]] = physical.stream<[0, 1]> () {aie.id = "0", aie.port = "DMA.O", aie.tile = "6.2"} : (!physical.ostream<i32>, !physical.istream<i32>)
    // CHECK-DAG: %[[OSTREAM1:.*]], %[[ISTREAM1:.*]] = physical.stream<[0, 1]> () {aie.id = "0", aie.port = "DMA.I", aie.tile = "7.3"} : (!physical.ostream<i32>, !physical.istream<i32>)
    // CHECK-DAG: %0 = physical.stream_hub(%[[ISTREAM0]], %[[OSTREAM1]]) {aie.impl = "broadcast_packet"} : (!physical.istream<i32>, !physical.ostream<i32>) -> !physical.stream_hub<i32>

    layout.route<["tile/6.2/port/DMA.O/id/0/stream",
                  "impl/broadcast_packet/stream_hub",
                  "tile/7.3/port/DMA.I/id/0/stream"]>
                  (%Q1: !spatial.queue<memref<1024xi32>> -> %node: !spatial.node)
    layout.route<["tile/6.2/port/DMA.O/id/0/stream",
                  "impl/broadcast_packet/stream_hub",
                  "tile/7.3/port/DMA.I/id/0/stream"]>
                  (%Q2: !spatial.queue<memref<1024xi32>> -> %node: !spatial.node)

  }
}
