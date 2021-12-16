//===- hello_world.mlir ----------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-create-cores --aie-lower-memcpy %s | FileCheck %s

// CHECK-LABEL: module @hello_world {
// CHECK:   %0 = AIE.tile(3, 3)
// CHECK:   %1 = AIE.buffer(%0) : memref<512xi32>
// CHECK:   %2 = AIE.mem(%0) {
// CHECK:     %10 = AIE.dmaStart(MM2S0, ^bb1, ^bb2)
// CHECK:   ^bb1:
// CHECK:     AIE.useToken @token0(Acquire, 1)
// CHECK:     AIE.dmaBd(<%1 : memref<512xi32>, 0, 512>, 0)
// CHECK:     AIE.useToken @token0(Release, 2)
// CHECK:     br ^bb2
// CHECK:   ^bb2:
// CHECK:     AIE.end
// CHECK:   }
// CHECK:   %3 = AIE.tile(4, 4)
// CHECK:   %4 = AIE.buffer(%3) : memref<512xi32>
// CHECK:   %5 = AIE.mem(%3) {
// CHECK:     %10 = AIE.dmaStart(S2MM0, ^bb1, ^bb2)
// CHECK:   ^bb1:
// CHECK:     AIE.useToken @token0(Acquire, 1)
// CHECK:     AIE.dmaBd(<%4 : memref<512xi32>, 0, 512>, 0)
// CHECK:     AIE.useToken @token0(Release, 2)
// CHECK:     br ^bb2
// CHECK:   ^bb2:
// CHECK:     AIE.end
// CHECK:   }
// CHECK:   %6 = memref.alloc() : memref<512xi32>
// CHECK:   %7 = memref.alloc() : memref<512xi32>
// CHECK:   AIE.token(0) {sym_name = "token0"}
// CHECK:   %8 = AIE.core(%0) {
// CHECK:     AIE.useToken @token0(Acquire, 0)
// CHECK:     %c16 = arith.constant 16 : index
// CHECK:     %c1_i32 = arith.constant 1 : i32
// CHECK:     memref.store %c1_i32, %1[%c16] : memref<512xi32>
// CHECK:     AIE.useToken @token0(Release, 1)
// CHECK:     AIE.end
// CHECK:   }
// CHECK:   %9 = AIE.core(%3) {
// CHECK:     AIE.useToken @token0(Acquire, 2)
// CHECK:     %c16 = arith.constant 16 : index
// CHECK:     %10 = memref.load %4[%c16] : memref<512xi32>
// CHECK:     AIE.useToken @token0(Release, 3)
// CHECK:     AIE.end
// CHECK:   }
// CHECK:   AIE.flow(%0, DMA : 0, %3, DMA : 0)
// CHECK: }

module @hello_world {

  %tile33 = AIE.tile(3, 3)
  %tile44 = AIE.tile(4, 4)

  %buf0 = memref.alloc() : memref<512xi32>
  %buf1 = memref.alloc() : memref<512xi32>

  AIE.token(0) { sym_name="token0" }

  func @producer(%arg0: memref<512xi32>) -> () {
    AIE.useToken @token0(Acquire, 0)
    %i = arith.constant 16 : index
    %val = arith.constant 1 : i32
    memref.store %val, %arg0[%i] : memref<512xi32>
    AIE.useToken @token0(Release, 1)
    return
  }

  func @consumer(%arg0: memref<512xi32>) -> () {
    AIE.useToken @token0(Acquire, 2)
    %i = arith.constant 16 : index
    %val = memref.load %arg0[%i] : memref<512xi32>
    AIE.useToken @token0(Release, 3)
    return
  }

  call @producer(%buf0) { aie.x = 3, aie.y = 3 } : (memref<512xi32>) -> () // write 1 to buf[16]
  call @consumer(%buf1) { aie.x = 4, aie.y = 4 } : (memref<512xi32>) -> () // read buf[16]

  AIE.memcpy @token0(1, 2) (%tile33 : <%buf0, 0, 512>, %tile44 : <%buf1, 0, 512>) : (memref<512xi32>, memref<512xi32>)
}
