// REQUIRES: aie_found
// RUN: phy-opt --convert-physical-to-aie %s | FileCheck %s
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// CHECK: %[[Tile7:.*]] = AIE.tile(7, 0)
// CHECK: %[[Tile6:.*]] = AIE.tile(6, 0)

// CHECK: %[[Lock:.*]] = AIE.lock(%[[Tile6]], 0)
// CHECK: %[[Buffer:.*]] = AIE.external_buffer

%0:2 = physical.stream<[0, 1]>(){ aie.tile = "6.0", aie.port = "DMA.O", aie.id = "0" }: (!physical.ostream<i32>, !physical.istream<i32>)
%1:2 = physical.stream<[0, 1]>(){ aie.tile = "6.0", aie.port = "DMA.I", aie.id = "1" }: (!physical.ostream<i32>, !physical.istream<i32>)
%2:2 = physical.stream<[0, 1]>(){ aie.tile = "7.0", aie.port = "DMA.O", aie.id = "1" }: (!physical.ostream<i32>, !physical.istream<i32>)
%l   = physical.lock<0>() { aie.tile = "6.0", aie.id = "0" }
%b   = physical.buffer() { aie.external_address = "2203318222848" }: memref<1024xi32>

// CHECK: AIE.shimDMA(%[[Tile6]]) {
// CHECK:   AIE.dmaStart(S2MM1, ^[[S2MM1:.*]], ^[[MM2SBD:.*]])
// CHECK: ^[[S2MM1]]:
// CHECK:   AIE.useLock(%[[Lock]], Acquire, 0)
// CHECK:   AIE.dmaBd(<%[[Buffer]] : memref<1024xi32>, 0, 1024>, 0)
// CHECK:   AIE.useLock(%[[Lock]], Release, 1)
// CHECK:   cf.br ^[[End:.*]]
// CHECK: ^[[MM2SBD]]:
// CHECK:   AIE.dmaStart(MM2S0, ^[[MM2S0:.*]], ^[[End]])
// CHECK: ^[[MM2S0]]:
// CHECK:   AIE.useLock(%[[Lock]], Acquire, 1)
// CHECK:   AIE.dmaBdPacket(0, 0)
// CHECK:   AIE.dmaBd(<%[[Buffer]] : memref<1024xi32>, 0, 1024>, 0)
// CHECK:   AIE.useLock(%[[Lock]], Release, 0)
// CHECK:   cf.br ^[[End:.*]]
// CHECK: ^[[End]]:
// CHECK:   AIE.end
// CHECK: }

physical.stream_dma(%0#0: !physical.ostream<i32>) {
  %m = physical.stream_dma_connect<0>(%l[1->0], %b[0:1024]: memref<1024xi32>)
} { aie.tile = "6.0", aie.engine = "MM2S", aie.id = "0" }

physical.stream_dma(%1#0: !physical.ostream<i32>) {
  %m = physical.stream_dma_connect(%l[0->1], %b[0:1024]: memref<1024xi32>)
} { aie.tile = "6.0", aie.engine = "S2MM", aie.id = "1" }

// CHECK: AIE.shimDMA(%[[Tile7]]) {
// CHECK:   AIE.dmaStart(MM2S1, ^[[MM2S1A:.*]], ^[[End2:.*]])
// CHECK: ^[[MM2S1A]]:
// CHECK:   AIE.useLock(%[[Lock]], Acquire, 1)
// CHECK:   AIE.dmaBd(<%[[Buffer]] : memref<1024xi32>, 0, 1024>, 0)
// CHECK:   AIE.useLock(%[[Lock]], Release, 0)
// CHECK:   cf.br ^[[MM2S1B:.*]]
// CHECK: ^[[MM2S1B]]:
// CHECK:   AIE.useLock(%[[Lock]], Acquire, 1)
// CHECK:   AIE.dmaBdPacket(0, 0)
// CHECK:   AIE.dmaBd(<%[[Buffer]] : memref<1024xi32>, 0, 1024>, 0)
// CHECK:   AIE.useLock(%[[Lock]], Release, 0)
// CHECK:   cf.br ^[[MM2S1B:.*]]
// CHECK: ^[[End2]]:
// CHECK:   AIE.end
// CHECK: }

physical.stream_dma(%2#0: !physical.ostream<i32>) {
  %m0 = physical.stream_dma_connect(%l[1->0], %b[0:1024]: memref<1024xi32>, %m1)
  %m1 = physical.stream_dma_connect<0>(%l[1->0], %b[0:1024]: memref<1024xi32>, %m1)
} { aie.tile = "7.0", aie.engine = "MM2S", aie.id = "1" }
