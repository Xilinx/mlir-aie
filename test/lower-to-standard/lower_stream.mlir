//===- lower_stream.mlir ---------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-standard-lowering="tilecol=1 tilerow=1" %s | FileCheck --check-prefix=CHECK11 %s
// RUN: aie-opt --aie-standard-lowering="tilecol=2 tilerow=1" %s | FileCheck --check-prefix=CHECK21 %s

//CHECK11:  func.func @core_1_1() {
//CHECK11:    %c0_i32 = arith.constant 0 : i32
//CHECK11:    %c1_i32 = arith.constant 1 : i32
//CHECK11:    %c16_i32 = arith.constant 16 : i32
//CHECK11:    %c32_i128 = arith.constant 32 : i128
//CHECK11:    call @llvm.aie.put.ms(%c0_i32, %c16_i32) : (i32, i32) -> ()
//CHECK11:    call @llvm.aie.put.wms(%c1_i32, %c32_i128) : (i32, i128) -> ()
//CHECK11:    %c64_i384 = arith.constant 64 : i384
//CHECK11:    call @llvm.aie.put.mcd(%c64_i384) : (i384) -> ()
//CHECK11:    return
//CHECK11:  }

//CHECK21:  func.func @core_2_1() {
//CHECK21:    %c0_i32 = arith.constant 0 : i32
//CHECK21:    %c1_i32 = arith.constant 1 : i32
//CHECK21:    %0 = call @llvm.aie.get.ss(%c0_i32) : (i32) -> i32
//CHECK21:    %1 = call @llvm.aie.get.ss(%c1_i32) : (i32) -> i32
//CHECK21:    %2 = arith.addi %0, %1 : i32
//CHECK21:    %3 = call @llvm.aie.get.scd() : () -> i384
//CHECK21:    return
//CHECK21:  }

// Test LLVM lowering to some AIE scalar intrinsic functions (streams, cascades)
// Each core's region is lowered to LLVM Dialect
module @test_core_llvm0 {
 AIE.device(xcvc1902) {
  %tile11 = AIE.tile(1, 1)
  %tile21 = AIE.tile(2, 1)

  %core11 = AIE.core(%tile11) {
    %0 = arith.constant 0 : i32
    %1 = arith.constant 1 : i32
    %val0 = arith.constant 16 : i32
    %val1 = arith.constant 32 : i128
    AIE.putStream(%val0 : i32,  %0 : i32)
    AIE.putStream(%val1 : i128, %1 : i32)
    %val2 = arith.constant 64 : i384
    AIE.putCascade(%val2 : i384)
    AIE.end
  }

  %core21 = AIE.core(%tile21) {
    %0 = arith.constant 0 : i32
    %1 = arith.constant 1 : i32
    //%val0 = AIE.getStream(0) : i32
    %val0 = AIE.getStream(%0 : i32) : i32
    %val1 = AIE.getStream(%1 : i32) : i32
    %2 = arith.addi %val0, %val1 : i32
    %3 = AIE.getCascade() : i384
    AIE.end
  }

 }
}
