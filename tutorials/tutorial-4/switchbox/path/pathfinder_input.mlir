//===- pathfinder_input.mlir -----------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-canonicalize-device %s | aie-opt --aie-create-pathfinder-flows | FileCheck %s
// CHECK: %15 = aie.switchbox(%3) {
// CHECK:      aie.connect<DMA : 0, North : 0>
// CHECK:      aie.connect<East : 0, DMA : 0>
// CHECK:    }
// CHECK:    %16 = aie.switchbox(%4) {
// CHECK:      aie.connect<South : 0, East : 0>
// CHECK:    }
// CHECK:    %17 = aie.switchbox(%7) {
// CHECK:      aie.connect<West : 0, East : 0>
// CHECK:      aie.connect<East : 0, South : 0>
// CHECK:    }
// CHECK:    %18 = aie.switchbox(%10) {
// CHECK:      aie.connect<West : 0, East : 0>
// CHECK:      aie.connect<East : 0, West : 0>
// CHECK:      aie.connect<South : 0, East : 1>
// CHECK:      aie.connect<East : 1, South : 0>
// CHECK:    }
// CHECK:    %19 = aie.switchbox(%13) {
// CHECK:      aie.connect<West : 0, DMA : 0>
// CHECK:      aie.connect<DMA : 0, West : 0>
// CHECK:      aie.connect<West : 1, North : 0>
// CHECK:      aie.connect<North : 0, West : 1>
// CHECK:    }
// CHECK:    %20 = aie.switchbox(%6) {
// CHECK:      aie.connect<North : 0, West : 0>
// CHECK:    }
// CHECK:    %21 = aie.switchbox(%9) {
// CHECK:      aie.connect<DMA : 0, North : 0>
// CHECK:      aie.connect<North : 0, DMA : 0>
// CHECK:    }
// CHECK:    %22 = aie.switchbox(%14) {
// CHECK:      aie.connect<South : 0, DMA : 0>
// CHECK:      aie.connect<DMA : 0, South : 0>
// CHECK:    }

module @pathfinder{
%t01 = aie.tile(0, 1)
%t02 = aie.tile(0, 2)
%t03 = aie.tile(0, 3)
%t11 = aie.tile(1, 1)
%t12 = aie.tile(1, 2)
%t13 = aie.tile(1, 3)
%t21 = aie.tile(2, 1)
%t22 = aie.tile(2, 2)
%t23 = aie.tile(2, 3)
%t31 = aie.tile(3, 1)
%t32 = aie.tile(3, 2)
%t33 = aie.tile(3, 3)
%t41 = aie.tile(4, 1)
%t42 = aie.tile(4, 2)
%t43 = aie.tile(4, 3)

aie.flow(%t11, DMA : 0, %t42, DMA : 0)
aie.flow(%t42, DMA : 0, %t11, DMA : 0)
aie.flow(%t31, DMA : 0, %t43, DMA : 0)
aie.flow(%t43, DMA : 0, %t31, DMA : 0)

//aie.flow(%t03, DMA : 0, %t41, DMA : 0)
}

