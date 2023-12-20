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
// CHECK:   %0 = aie.herd[1] [1] {sym_name = "t"}
// CHECK:   %1 = aie.herd[1] [1] {sym_name = "s"}
// CHECK:   %2 = aie.iter(0, 1, 1)
// CHECK:   %3 = aie.select(%0, %2, %2)
// CHECK:   %4 = aie.select(%1, %2, %2)
// CHECK:   %5 = aie.iter(3, 4, 1)
// CHECK:   %6 = aie.iter(1, 2, 1)
// CHECK:   %7 = aie.select(%0, %5, %6)
// CHECK:   %8 = aie.switchbox(%7) {
// CHECK:     aie.connect<South : 0, North : 0>
// CHECK:   }
// CHECK:   %9 = aie.iter(1, 2, 1)
// CHECK:   %10 = aie.iter(0, 1, 1)
// CHECK:   %11 = aie.select(%0, %9, %10)
// CHECK:   %12 = aie.switchbox(%11) {
// CHECK:     aie.connect<West : 0, East : 0>
// CHECK:   }
// CHECK:   %13 = aie.iter(3, 4, 1)
// CHECK:   %14 = aie.iter(2, 3, 1)
// CHECK:   %15 = aie.select(%0, %13, %14)
// CHECK:   %16 = aie.switchbox(%15) {
// CHECK:     aie.connect<South : 0, North : 0>
// CHECK:   }
// CHECK:   %17 = aie.iter(0, 1, 1)
// CHECK:   %18 = aie.iter(0, 1, 1)
// CHECK:   %19 = aie.select(%0, %17, %18)
// CHECK:   %20 = aie.switchbox(%19) {
// CHECK:     aie.connect<DMA : 0, East : 0>
// CHECK:   }
// CHECK:   %21 = aie.iter(3, 4, 1)
// CHECK:   %22 = aie.iter(3, 4, 1)
// CHECK:   %23 = aie.select(%0, %21, %22)
// CHECK:   %24 = aie.switchbox(%23) {
// CHECK:     aie.connect<South : 0, DMA : 0>
// CHECK:   }
// CHECK:   %25 = aie.iter(2, 3, 1)
// CHECK:   %26 = aie.iter(0, 1, 1)
// CHECK:   %27 = aie.select(%0, %25, %26)
// CHECK:   %28 = aie.switchbox(%27) {
// CHECK:     aie.connect<West : 0, East : 0>
// CHECK:   }
// CHECK:   %29 = aie.iter(3, 4, 1)
// CHECK:   %30 = aie.iter(0, 1, 1)
// CHECK:   %31 = aie.select(%0, %29, %30)
// CHECK:   %32 = aie.switchbox(%31) {
// CHECK:     aie.connect<West : 0, North : 0>
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
 aie.device(xcvc1902) {
  %0 = aie.herd[1][1] { sym_name = "t" } // herd t
  %1 = aie.herd[1][1] { sym_name = "s" } // herd s

  %i0 = aie.iter(0, 1, 1)

  %2 = aie.select(%0, %i0, %i0)
  %3 = aie.select(%1, %i0, %i0)
  aie.place(%0, %1, 3, 3) // herd t[0][0] and herd s[0][0] are spaced by 3-horizontally and 3-vertically
  aie.route(<%2, DMA: 0>, <%3, DMA: 0>)
 }
}
