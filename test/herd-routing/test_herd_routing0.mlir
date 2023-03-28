//===- test_herd_routing0.mlir ---------------------------------*- MLIR -*-===//
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

// CHECK-LABEL: module @test_herd_routing0 {
// CHECK:   %0 = AIE.herd[1] [1] {sym_name = "t"}
// CHECK:   %1 = AIE.herd[1] [1] {sym_name = "s"}
// CHECK:   %2 = AIE.iter(0, 1, 1)
// CHECK:   %3 = AIE.select(%0, %2, %2)
// CHECK:   %4 = AIE.select(%1, %2, %2)
// CHECK:   %5 = AIE.iter(3, 4, 1)
// CHECK:   %6 = AIE.iter(1, 2, 1)
// CHECK:   %7 = AIE.select(%0, %5, %6)
// CHECK:   %8 = AIE.switchbox(%7) {
// CHECK:     AIE.connect<South : 0, North : 0>
// CHECK:   }
// CHECK:   %9 = AIE.iter(1, 2, 1)
// CHECK:   %10 = AIE.iter(0, 1, 1)
// CHECK:   %11 = AIE.select(%0, %9, %10)
// CHECK:   %12 = AIE.switchbox(%11) {
// CHECK:     AIE.connect<West : 0, East : 0>
// CHECK:   }
// CHECK:   %13 = AIE.iter(3, 4, 1)
// CHECK:   %14 = AIE.iter(2, 3, 1)
// CHECK:   %15 = AIE.select(%0, %13, %14)
// CHECK:   %16 = AIE.switchbox(%15) {
// CHECK:     AIE.connect<South : 0, North : 0>
// CHECK:   }
// CHECK:   %17 = AIE.iter(0, 1, 1)
// CHECK:   %18 = AIE.iter(0, 1, 1)
// CHECK:   %19 = AIE.select(%0, %17, %18)
// CHECK:   %20 = AIE.switchbox(%19) {
// CHECK:     AIE.connect<DMA : 0, East : 0>
// CHECK:   }
// CHECK:   %21 = AIE.iter(3, 4, 1)
// CHECK:   %22 = AIE.iter(3, 4, 1)
// CHECK:   %23 = AIE.select(%0, %21, %22)
// CHECK:   %24 = AIE.switchbox(%23) {
// CHECK:     AIE.connect<South : 0, DMA : 0>
// CHECK:   }
// CHECK:   %25 = AIE.iter(2, 3, 1)
// CHECK:   %26 = AIE.iter(0, 1, 1)
// CHECK:   %27 = AIE.select(%0, %25, %26)
// CHECK:   %28 = AIE.switchbox(%27) {
// CHECK:     AIE.connect<West : 0, East : 0>
// CHECK:   }
// CHECK:   %29 = AIE.iter(3, 4, 1)
// CHECK:   %30 = AIE.iter(0, 1, 1)
// CHECK:   %31 = AIE.select(%0, %29, %30)
// CHECK:   %32 = AIE.switchbox(%31) {
// CHECK:     AIE.connect<West : 0, North : 0>
// CHECK:   }
// CHECK: }

// * * * * * * * * *
// * * * * * s * * *
// * * * * * * * * *
// * * * * * * * * *
// * * t * * * * * *
//
// t[0][0] copies to s[0][0]
module @test_herd_routing0 {
 AIE.device(xcvc1902) {
  %0 = AIE.herd[1][1] { sym_name = "t" } // herd t
  %1 = AIE.herd[1][1] { sym_name = "s" } // herd s

  %i0 = AIE.iter(0, 1, 1)

  %2 = AIE.select(%0, %i0, %i0)
  %3 = AIE.select(%1, %i0, %i0)
  AIE.place(%0, %1, 3, 3) // herd t[0][0] and herd s[0][0] are spaced by 3-horizontally and 3-vertically
  AIE.route(<%2, DMA: 0>, <%3, DMA: 0>)
 }
}
