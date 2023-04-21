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
// CHECK: %15 = AIE.switchbox(%3) {
// CHECK:      AIE.connect<DMA : 0, North : 0>
// CHECK:      AIE.connect<East : 0, DMA : 0>
// CHECK:    }
// CHECK:    %16 = AIE.switchbox(%4) {
// CHECK:      AIE.connect<South : 0, East : 0>
// CHECK:    }
// CHECK:    %17 = AIE.switchbox(%7) {
// CHECK:      AIE.connect<West : 0, East : 0>
// CHECK:      AIE.connect<East : 0, South : 0>
// CHECK:    }
// CHECK:    %18 = AIE.switchbox(%10) {
// CHECK:      AIE.connect<West : 0, East : 0>
// CHECK:      AIE.connect<East : 0, West : 0>
// CHECK:      AIE.connect<South : 0, East : 1>
// CHECK:      AIE.connect<East : 1, South : 0>
// CHECK:    }
// CHECK:    %19 = AIE.switchbox(%13) {
// CHECK:      AIE.connect<West : 0, DMA : 0>
// CHECK:      AIE.connect<DMA : 0, West : 0>
// CHECK:      AIE.connect<West : 1, North : 0>
// CHECK:      AIE.connect<North : 0, West : 1>
// CHECK:    }
// CHECK:    %20 = AIE.switchbox(%6) {
// CHECK:      AIE.connect<North : 0, West : 0>
// CHECK:    }
// CHECK:    %21 = AIE.switchbox(%9) {
// CHECK:      AIE.connect<DMA : 0, North : 0>
// CHECK:      AIE.connect<North : 0, DMA : 0>
// CHECK:    }
// CHECK:    %22 = AIE.switchbox(%14) {
// CHECK:      AIE.connect<South : 0, DMA : 0>
// CHECK:      AIE.connect<DMA : 0, South : 0>
// CHECK:    }

module @pathfinder{
%t01 = AIE.tile(0, 1)
%t02 = AIE.tile(0, 2)
%t03 = AIE.tile(0, 3)
%t11 = AIE.tile(1, 1)
%t12 = AIE.tile(1, 2)
%t13 = AIE.tile(1, 3)
%t21 = AIE.tile(2, 1)
%t22 = AIE.tile(2, 2)
%t23 = AIE.tile(2, 3)
%t31 = AIE.tile(3, 1)
%t32 = AIE.tile(3, 2)
%t33 = AIE.tile(3, 3)
%t41 = AIE.tile(4, 1)
%t42 = AIE.tile(4, 2)
%t43 = AIE.tile(4, 3)

AIE.flow(%t11, DMA : 0, %t42, DMA : 0)
AIE.flow(%t42, DMA : 0, %t11, DMA : 0)
AIE.flow(%t31, DMA : 0, %t43, DMA : 0)
AIE.flow(%t43, DMA : 0, %t31, DMA : 0)

//AIE.flow(%t03, DMA : 0, %t41, DMA : 0)
}

