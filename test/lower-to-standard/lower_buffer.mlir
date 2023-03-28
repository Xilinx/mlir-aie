//===- lower_buffer.mlir ---------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-standard-lowering="tilecol=3 tilerow=3" %s | FileCheck --check-prefixes=CHECKALL,CHECK33 %s
// RUN: aie-opt --aie-standard-lowering="tilecol=4 tilerow=3" %s | FileCheck --check-prefixes=CHECKALL,CHECK43 %s
// RUN: aie-opt --aie-standard-lowering %s | FileCheck --check-prefixes=CHECKALL,CHECK33,CHECK43 %s

// CHECKALL:    memref.global "public" @a : memref<4xi32>
// CHECK43-LABEL:  func.func @core_4_3() {
// CHECK43:    %c0 = arith.constant 0 : index
// CHECK43:    %0 = memref.get_global @a : memref<4xi32>
// CHECK43:    %1 = memref.load %0[%c0] : memref<4xi32>
// CHECK43:    return
// CHECK43:  }

// CHECK33-LABEL:  func.func @core_3_3() {
// CHECK33:    %c0 = arith.constant 0 : index
// CHECK33:    %c377_i32 = arith.constant 377 : i32
// CHECK33:    %0 = memref.get_global @a : memref<4xi32>
// CHECK33:    memref.store %c377_i32, %0[%c0] : memref<4xi32>
// CHECK33:    return
// CHECK33:  }

module @codegen1 {
 AIE.device(xcvc1902) {
  %t33 = AIE.tile(3, 3)
  %a = AIE.buffer(%t33) { sym_name = "a" } : memref<4xi32>
  %core33 = AIE.core(%t33) {
    %0 = arith.constant 0 : index
    %377 = arith.constant 377 : i32
    memref.store %377, %a[%0] : memref<4xi32>
    AIE.end
  }
  %t34 = AIE.tile(4, 3)

  %core34 = AIE.core(%t34) {
    %0 = arith.constant 0 : index
    %1 = memref.load %a[%0] : memref<4xi32>
//    AIE.debug(%1 : i32)
    AIE.end
  }
 }
}
