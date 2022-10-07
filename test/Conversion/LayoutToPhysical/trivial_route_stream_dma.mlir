// RUN: phy-opt --convert-layout-to-physical="device=aie" %s | FileCheck %s
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

func.func private @kernel(%Q: !spatial.queue<memref<1024xi32>>) {
  cf.br ^bb
^bb:
  cf.br ^bb
}

%Q = spatial.queue<1>(): !spatial.queue<memref<1024xi32>>
%node = spatial.node @kernel(%Q): (!spatial.queue<memref<1024xi32>>) -> !spatial.node

layout.platform<"vck190"> {
  layout.device<"aie"> {

    // CHECK: %[[OSTREAM:.*]], %[[ISTREAM:.*]] = physical.stream () {aie.id = "0", aie.port = "DMA.I", aie.tile = "6.3"} : (!physical.ostream<i32>, !physical.istream<i32>)
    // CHECK: physical.stream_dma(%[[ISTREAM]] : !physical.istream<i32>) {
    // CHECK:   %[[CONNECT:.*]] = physical.stream_dma_connect (%0[0 -> 1], %1[0 : 1024] : memref<1024xi32>, %[[CONNECT]]) 
    // CHECK: } {aie.engine = "S2MM", aie.id = "0", aie.tile = "6.3"}

    layout.route<["tile/6.3/port/DMA.I/id/0/stream",
                  "tile/6.3/engine/S2MM/id/0/stream_dma",
                  "tile/6.3/id/1/buffer,tile/6.3/id/1/lock"]>
                  (%Q: !spatial.queue<memref<1024xi32>> -> %node: !spatial.node)

  }
}
