//===- test_herd_routing2.mlir ---------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// REQUIRES: stephenn
// RUN: aie-opt --aie-herd-routing %s | FileCheck %s

// CHECK-LABEL: module @test_herd_routing2 {
// CHECK: }

// This models the connectivity between some herds from one of Phil's examples (sixteen_tiles_plane_daa)
// Herd "pp" streams data to the first column of Herd "compute"
// Herd "ifm" streams data to Herd "compute" column-by-column
module @test_herd_routing2 {
 aie.device(xcvc1902) {
  %0 = aie.herd[4][1] { sym_name = "pp" }        // herd ping-pong
  %1 = aie.herd[4][1] { sym_name = "ifm" }       // herd input-feature-map
  %2 = aie.herd[4][4] { sym_name = "compute" }   // herd compute

  aie.place(%0, %2, 0, 2)
  aie.place(%1, %2, 0, 1)

  %i0 = aie.iter(0, 1, 1) // 0
  %i1 = aie.iter(0, 4, 1) // 0, 1, 2, 3
  %i2 = aie.iter(0, 4, 1) // 0, 1, 2, 3

  %3 = aie.select(%0, %i1, %i0) // (0, 0), (1, 0), (2, 0), (3, 0)
                                //   |       |       |       |
                                //   v       v       v       v
  %4 = aie.select(%2, %i0, %i1) // (0, 0), (0, 1), (0, 2), (0, 3)
  aie.route(<%3, DMA: 0>, <%4, DMA: 0>)

  %5 = aie.select(%1, %i1, %i0) // (0, 0), (1, 0), (2, 0), (3, 0)
                                //   |       |       |       |
                                //   v       v       v       v
  %6 = aie.select(%2, %i1, %i2) // (0, 0), (1, 0), (2, 0), (3, 0)
                                // (0, 1), (1, 1), (2, 1), (3, 1)
                                // (0, 2), (1, 2), (2, 2), (3, 2)
                                // (0, 3), (1, 3), (2, 3), (3, 3)
  aie.route(<%5, DMA: 0>, <%6, DMA: 1>)
 }
}
