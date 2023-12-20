//===- test_herd_routing1.mlir ---------------------------------*- MLIR -*-===//
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

// CHECK-LABEL: module @test_herd_routing1 {
// CHECK:   %0 = aie.herd[4] [1] {sym_name = "t"}
// CHECK:   %1 = aie.herd[4] [4] {sym_name = "s"}
// CHECK:   %2 = aie.iter(0, 1, 1)
// CHECK:   %3 = aie.iter(0, 4, 1)
// CHECK:   %4 = aie.iter(0, 4, 1)
// CHECK:   %5 = aie.select(%0, %3, %2)
// CHECK:   %6 = aie.select(%1, %3, %4)
// CHECK:   %7 = aie.iter(1, 2, 1)
// CHECK:   %8 = aie.iter(4, 5, 1)
// CHECK:   %9 = aie.select(%0, %7, %8)
// CHECK:   %10 = aie.switchbox(%9) {
// CHECK:     aie.connect<South : 0, DMA : 0>
// CHECK:   }
// CHECK:   %11 = aie.iter(1, 2, 1)
// CHECK:   %12 = aie.iter(3, 4, 1)
// CHECK:   %13 = aie.select(%0, %11, %12)
// CHECK:   %14 = aie.switchbox(%13) {
// CHECK:     aie.connect<South : 0, DMA : 0>
// CHECK:     aie.connect<South : 0, North : 0>
// CHECK:   }
// CHECK:   %15 = aie.iter(0, 1, 1)
// CHECK:   %16 = aie.iter(4, 5, 1)
// CHECK:   %17 = aie.select(%0, %15, %16)
// CHECK:   %18 = aie.switchbox(%17) {
// CHECK:     aie.connect<South : 0, DMA : 0>
// CHECK:   }
// CHECK:   %19 = aie.iter(1, 2, 1)
// CHECK:   %20 = aie.iter(0, 1, 1)
// CHECK:   %21 = aie.select(%0, %19, %20)
// CHECK:   %22 = aie.switchbox(%21) {
// CHECK:     aie.connect<DMA : 0, North : 0>
// CHECK:   }
// CHECK:   %23 = aie.iter(3, 4, 1)
// CHECK:   %24 = aie.iter(2, 3, 1)
// CHECK:   %25 = aie.select(%0, %23, %24)
// CHECK:   %26 = aie.switchbox(%25) {
// CHECK:     aie.connect<South : 0, DMA : 0>
// CHECK:     aie.connect<South : 0, North : 0>
// CHECK:   }
// CHECK:   %27 = aie.iter(1, 2, 1)
// CHECK:   %28 = aie.iter(2, 3, 1)
// CHECK:   %29 = aie.select(%0, %27, %28)
// CHECK:   %30 = aie.switchbox(%29) {
// CHECK:     aie.connect<South : 0, DMA : 0>
// CHECK:     aie.connect<South : 0, North : 0>
// CHECK:   }
// CHECK:   %31 = aie.iter(2, 3, 1)
// CHECK:   %32 = aie.iter(2, 3, 1)
// CHECK:   %33 = aie.select(%0, %31, %32)
// CHECK:   %34 = aie.switchbox(%33) {
// CHECK:     aie.connect<South : 0, DMA : 0>
// CHECK:     aie.connect<South : 0, North : 0>
// CHECK:   }
// CHECK:   %35 = aie.iter(0, 1, 1)
// CHECK:   %36 = aie.iter(0, 1, 1)
// CHECK:   %37 = aie.select(%0, %35, %36)
// CHECK:   %38 = aie.switchbox(%37) {
// CHECK:     aie.connect<DMA : 0, North : 0>
// CHECK:   }
// CHECK:   %39 = aie.iter(3, 4, 1)
// CHECK:   %40 = aie.iter(3, 4, 1)
// CHECK:   %41 = aie.select(%0, %39, %40)
// CHECK:   %42 = aie.switchbox(%41) {
// CHECK:     aie.connect<South : 0, DMA : 0>
// CHECK:     aie.connect<South : 0, North : 0>
// CHECK:   }
// CHECK:   %43 = aie.iter(3, 4, 1)
// CHECK:   %44 = aie.iter(0, 1, 1)
// CHECK:   %45 = aie.select(%0, %43, %44)
// CHECK:   %46 = aie.switchbox(%45) {
// CHECK:     aie.connect<DMA : 0, North : 0>
// CHECK:   }
// CHECK:   %47 = aie.iter(2, 3, 1)
// CHECK:   %48 = aie.iter(3, 4, 1)
// CHECK:   %49 = aie.select(%0, %47, %48)
// CHECK:   %50 = aie.switchbox(%49) {
// CHECK:     aie.connect<South : 0, DMA : 0>
// CHECK:     aie.connect<South : 0, North : 0>
// CHECK:   }
// CHECK:   %51 = aie.iter(0, 1, 1)
// CHECK:   %52 = aie.iter(3, 4, 1)
// CHECK:   %53 = aie.select(%0, %51, %52)
// CHECK:   %54 = aie.switchbox(%53) {
// CHECK:     aie.connect<South : 0, DMA : 0>
// CHECK:     aie.connect<South : 0, North : 0>
// CHECK:   }
// CHECK:   %55 = aie.iter(2, 3, 1)
// CHECK:   %56 = aie.iter(4, 5, 1)
// CHECK:   %57 = aie.select(%0, %55, %56)
// CHECK:   %58 = aie.switchbox(%57) {
// CHECK:     aie.connect<South : 0, DMA : 0>
// CHECK:   }
// CHECK:   %59 = aie.iter(2, 3, 1)
// CHECK:   %60 = aie.iter(0, 1, 1)
// CHECK:   %61 = aie.select(%0, %59, %60)
// CHECK:   %62 = aie.switchbox(%61) {
// CHECK:     aie.connect<DMA : 0, North : 0>
// CHECK:   }
// CHECK:   %63 = aie.iter(3, 4, 1)
// CHECK:   %64 = aie.iter(1, 2, 1)
// CHECK:   %65 = aie.select(%0, %63, %64)
// CHECK:   %66 = aie.switchbox(%65) {
// CHECK:     aie.connect<South : 0, DMA : 0>
// CHECK:     aie.connect<South : 0, North : 0>
// CHECK:   }
// CHECK:   %67 = aie.iter(3, 4, 1)
// CHECK:   %68 = aie.iter(4, 5, 1)
// CHECK:   %69 = aie.select(%0, %67, %68)
// CHECK:   %70 = aie.switchbox(%69) {
// CHECK:     aie.connect<South : 0, DMA : 0>
// CHECK:   }
// CHECK:   %71 = aie.iter(1, 2, 1)
// CHECK:   %72 = aie.iter(1, 2, 1)
// CHECK:   %73 = aie.select(%0, %71, %72)
// CHECK:   %74 = aie.switchbox(%73) {
// CHECK:     aie.connect<South : 0, DMA : 0>
// CHECK:     aie.connect<South : 0, North : 0>
// CHECK:   }
// CHECK:   %75 = aie.iter(2, 3, 1)
// CHECK:   %76 = aie.iter(1, 2, 1)
// CHECK:   %77 = aie.select(%0, %75, %76)
// CHECK:   %78 = aie.switchbox(%77) {
// CHECK:     aie.connect<South : 0, DMA : 0>
// CHECK:     aie.connect<South : 0, North : 0>
// CHECK:   }
// CHECK:   %79 = aie.iter(0, 1, 1)
// CHECK:   %80 = aie.iter(2, 3, 1)
// CHECK:   %81 = aie.select(%0, %79, %80)
// CHECK:   %82 = aie.switchbox(%81) {
// CHECK:     aie.connect<South : 0, DMA : 0>
// CHECK:     aie.connect<South : 0, North : 0>
// CHECK:   }
// CHECK:   %83 = aie.iter(0, 1, 1)
// CHECK:   %84 = aie.iter(1, 2, 1)
// CHECK:   %85 = aie.select(%0, %83, %84)
// CHECK:   %86 = aie.switchbox(%85) {
// CHECK:     aie.connect<South : 0, DMA : 0>
// CHECK:     aie.connect<South : 0, North : 0>
// CHECK:   }
// CHECK: }

// * * s s s s * * *
// * * s s s s * * *
// * * s s s s * * *
// * * s s s s * * *
// * * t t t t * * *
//
// t[0][0] broadcasts to s[0][0], s[0][1], s[0][2], s[0][3]
// t[1][0] broadcasts to s[1][0], s[1][1], s[1][2], s[1][3]
// t[2][0] broadcasts to s[2][0], s[2][1], s[2][2], s[2][3]
// t[3][0] broadcasts to s[3][0], s[3][1], s[3][2], s[3][3]
module @test_herd_routing1 {
 aie.device(xcvc1902) {
  %0 = aie.herd[4][1] { sym_name = "t" } // herd t
  %1 = aie.herd[4][4] { sym_name = "s" } // herd s

  %i0 = aie.iter(0, 1, 1)
  %i1 = aie.iter(0, 4, 1)
  %i2 = aie.iter(0, 4, 1)

  %2 = aie.select(%0, %i1, %i0)
  %3 = aie.select(%1, %i1, %i2)
  aie.place(%0, %1, 0, 1) // herd t[0][0] and herd s[0][0] are spaced by 0-horizontally and 1-vertically
  aie.route(<%2, DMA: 0>, <%3, DMA: 0>)
 }
}
