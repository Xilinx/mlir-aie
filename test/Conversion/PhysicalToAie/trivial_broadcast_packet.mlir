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

// CHECK: %[[Tile4:.*]] = AIE.tile(6, 4)
// CHECK: %[[Tile3:.*]] = AIE.tile(6, 3)
// CHECK: %[[Tile0:.*]] = AIE.tile(6, 0)
%0:2 = physical.stream<[0, 1]>(){ aie.tile = "6.0", aie.port = "DMA.O",  aie.id = "0" }: (!physical.ostream<i32>, !physical.istream<i32>)
%1:2 = physical.stream<[0]>()   { aie.tile = "6.3", aie.port = "Core.I", aie.id = "0" }: (!physical.ostream<i32>, !physical.istream<i32>)
%2:2 = physical.stream<[1]>()   { aie.tile = "6.4", aie.port = "DMA.I",  aie.id = "2" }: (!physical.ostream<i32>, !physical.istream<i32>)
%3:2 = physical.stream<[2]>()   { aie.tile = "6.4", aie.port = "Core.I", aie.id = "1" }: (!physical.ostream<i32>, !physical.istream<i32>)


// CHECK: AIE.broadcast_packet(%[[Tile0]], DMA : 0) {
// CHECK:   AIE.bp_id(0) {
// CHECK:     AIE.bp_dest<%[[Tile3]], Core : 0>
// CHECK:   }
// CHECK:   AIE.bp_id(1) {
// CHECK:     AIE.bp_dest<%[[Tile4]], DMA : 2>
// CHECK:   }
// CHECK: }
physical.stream_hub(%0#1, %1#0, %2#0, %3#0)
  { aie.impl = "broadcast_packet" }
  : (!physical.istream<i32>, !physical.ostream<i32>, !physical.ostream<i32>, !physical.ostream<i32>)
  -> !physical.stream_hub<i32>