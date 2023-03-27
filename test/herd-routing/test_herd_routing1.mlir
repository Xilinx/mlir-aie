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
// CHECK:   %0 = AIE.herd[4] [1] {sym_name = "t"}
// CHECK:   %1 = AIE.herd[4] [4] {sym_name = "s"}
// CHECK:   %2 = AIE.iter(0, 1, 1)
// CHECK:   %3 = AIE.iter(0, 4, 1)
// CHECK:   %4 = AIE.iter(0, 4, 1)
// CHECK:   %5 = AIE.select(%0, %3, %2)
// CHECK:   %6 = AIE.select(%1, %3, %4)
// CHECK:   %7 = AIE.iter(1, 2, 1)
// CHECK:   %8 = AIE.iter(4, 5, 1)
// CHECK:   %9 = AIE.select(%0, %7, %8)
// CHECK:   %10 = AIE.switchbox(%9) {
// CHECK:     AIE.connect<South : 0, DMA : 0>
// CHECK:   }
// CHECK:   %11 = AIE.iter(1, 2, 1)
// CHECK:   %12 = AIE.iter(3, 4, 1)
// CHECK:   %13 = AIE.select(%0, %11, %12)
// CHECK:   %14 = AIE.switchbox(%13) {
// CHECK:     AIE.connect<South : 0, DMA : 0>
// CHECK:     AIE.connect<South : 0, North : 0>
// CHECK:   }
// CHECK:   %15 = AIE.iter(0, 1, 1)
// CHECK:   %16 = AIE.iter(4, 5, 1)
// CHECK:   %17 = AIE.select(%0, %15, %16)
// CHECK:   %18 = AIE.switchbox(%17) {
// CHECK:     AIE.connect<South : 0, DMA : 0>
// CHECK:   }
// CHECK:   %19 = AIE.iter(1, 2, 1)
// CHECK:   %20 = AIE.iter(0, 1, 1)
// CHECK:   %21 = AIE.select(%0, %19, %20)
// CHECK:   %22 = AIE.switchbox(%21) {
// CHECK:     AIE.connect<DMA : 0, North : 0>
// CHECK:   }
// CHECK:   %23 = AIE.iter(3, 4, 1)
// CHECK:   %24 = AIE.iter(2, 3, 1)
// CHECK:   %25 = AIE.select(%0, %23, %24)
// CHECK:   %26 = AIE.switchbox(%25) {
// CHECK:     AIE.connect<South : 0, DMA : 0>
// CHECK:     AIE.connect<South : 0, North : 0>
// CHECK:   }
// CHECK:   %27 = AIE.iter(1, 2, 1)
// CHECK:   %28 = AIE.iter(2, 3, 1)
// CHECK:   %29 = AIE.select(%0, %27, %28)
// CHECK:   %30 = AIE.switchbox(%29) {
// CHECK:     AIE.connect<South : 0, DMA : 0>
// CHECK:     AIE.connect<South : 0, North : 0>
// CHECK:   }
// CHECK:   %31 = AIE.iter(2, 3, 1)
// CHECK:   %32 = AIE.iter(2, 3, 1)
// CHECK:   %33 = AIE.select(%0, %31, %32)
// CHECK:   %34 = AIE.switchbox(%33) {
// CHECK:     AIE.connect<South : 0, DMA : 0>
// CHECK:     AIE.connect<South : 0, North : 0>
// CHECK:   }
// CHECK:   %35 = AIE.iter(0, 1, 1)
// CHECK:   %36 = AIE.iter(0, 1, 1)
// CHECK:   %37 = AIE.select(%0, %35, %36)
// CHECK:   %38 = AIE.switchbox(%37) {
// CHECK:     AIE.connect<DMA : 0, North : 0>
// CHECK:   }
// CHECK:   %39 = AIE.iter(3, 4, 1)
// CHECK:   %40 = AIE.iter(3, 4, 1)
// CHECK:   %41 = AIE.select(%0, %39, %40)
// CHECK:   %42 = AIE.switchbox(%41) {
// CHECK:     AIE.connect<South : 0, DMA : 0>
// CHECK:     AIE.connect<South : 0, North : 0>
// CHECK:   }
// CHECK:   %43 = AIE.iter(3, 4, 1)
// CHECK:   %44 = AIE.iter(0, 1, 1)
// CHECK:   %45 = AIE.select(%0, %43, %44)
// CHECK:   %46 = AIE.switchbox(%45) {
// CHECK:     AIE.connect<DMA : 0, North : 0>
// CHECK:   }
// CHECK:   %47 = AIE.iter(2, 3, 1)
// CHECK:   %48 = AIE.iter(3, 4, 1)
// CHECK:   %49 = AIE.select(%0, %47, %48)
// CHECK:   %50 = AIE.switchbox(%49) {
// CHECK:     AIE.connect<South : 0, DMA : 0>
// CHECK:     AIE.connect<South : 0, North : 0>
// CHECK:   }
// CHECK:   %51 = AIE.iter(0, 1, 1)
// CHECK:   %52 = AIE.iter(3, 4, 1)
// CHECK:   %53 = AIE.select(%0, %51, %52)
// CHECK:   %54 = AIE.switchbox(%53) {
// CHECK:     AIE.connect<South : 0, DMA : 0>
// CHECK:     AIE.connect<South : 0, North : 0>
// CHECK:   }
// CHECK:   %55 = AIE.iter(2, 3, 1)
// CHECK:   %56 = AIE.iter(4, 5, 1)
// CHECK:   %57 = AIE.select(%0, %55, %56)
// CHECK:   %58 = AIE.switchbox(%57) {
// CHECK:     AIE.connect<South : 0, DMA : 0>
// CHECK:   }
// CHECK:   %59 = AIE.iter(2, 3, 1)
// CHECK:   %60 = AIE.iter(0, 1, 1)
// CHECK:   %61 = AIE.select(%0, %59, %60)
// CHECK:   %62 = AIE.switchbox(%61) {
// CHECK:     AIE.connect<DMA : 0, North : 0>
// CHECK:   }
// CHECK:   %63 = AIE.iter(3, 4, 1)
// CHECK:   %64 = AIE.iter(1, 2, 1)
// CHECK:   %65 = AIE.select(%0, %63, %64)
// CHECK:   %66 = AIE.switchbox(%65) {
// CHECK:     AIE.connect<South : 0, DMA : 0>
// CHECK:     AIE.connect<South : 0, North : 0>
// CHECK:   }
// CHECK:   %67 = AIE.iter(3, 4, 1)
// CHECK:   %68 = AIE.iter(4, 5, 1)
// CHECK:   %69 = AIE.select(%0, %67, %68)
// CHECK:   %70 = AIE.switchbox(%69) {
// CHECK:     AIE.connect<South : 0, DMA : 0>
// CHECK:   }
// CHECK:   %71 = AIE.iter(1, 2, 1)
// CHECK:   %72 = AIE.iter(1, 2, 1)
// CHECK:   %73 = AIE.select(%0, %71, %72)
// CHECK:   %74 = AIE.switchbox(%73) {
// CHECK:     AIE.connect<South : 0, DMA : 0>
// CHECK:     AIE.connect<South : 0, North : 0>
// CHECK:   }
// CHECK:   %75 = AIE.iter(2, 3, 1)
// CHECK:   %76 = AIE.iter(1, 2, 1)
// CHECK:   %77 = AIE.select(%0, %75, %76)
// CHECK:   %78 = AIE.switchbox(%77) {
// CHECK:     AIE.connect<South : 0, DMA : 0>
// CHECK:     AIE.connect<South : 0, North : 0>
// CHECK:   }
// CHECK:   %79 = AIE.iter(0, 1, 1)
// CHECK:   %80 = AIE.iter(2, 3, 1)
// CHECK:   %81 = AIE.select(%0, %79, %80)
// CHECK:   %82 = AIE.switchbox(%81) {
// CHECK:     AIE.connect<South : 0, DMA : 0>
// CHECK:     AIE.connect<South : 0, North : 0>
// CHECK:   }
// CHECK:   %83 = AIE.iter(0, 1, 1)
// CHECK:   %84 = AIE.iter(1, 2, 1)
// CHECK:   %85 = AIE.select(%0, %83, %84)
// CHECK:   %86 = AIE.switchbox(%85) {
// CHECK:     AIE.connect<South : 0, DMA : 0>
// CHECK:     AIE.connect<South : 0, North : 0>
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
 AIE.device(xcvc1902) {
  %0 = AIE.herd[4][1] { sym_name = "t" } // herd t
  %1 = AIE.herd[4][4] { sym_name = "s" } // herd s

  %i0 = AIE.iter(0, 1, 1)
  %i1 = AIE.iter(0, 4, 1)
  %i2 = AIE.iter(0, 4, 1)

  %2 = AIE.select(%0, %i1, %i0)
  %3 = AIE.select(%1, %i1, %i2)
  AIE.place(%0, %1, 0, 1) // herd t[0][0] and herd s[0][0] are spaced by 0-horizontally and 1-vertically
  AIE.route(<%2, DMA: 0>, <%3, DMA: 0>)
 }
}
