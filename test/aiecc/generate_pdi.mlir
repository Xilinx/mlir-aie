//===- generate_pdi.mlir ---------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2025 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// REQUIRES: chess
// REQUIRES: peano

// RUN: %python aiecc.py -v --xchesscc --xbridge --aie-generate-pdi --pdi-name=MlirAie0.pdi %s | FileCheck %s --check-prefix=XCHESSCC
// RUN: %python aiecc.py -v --no-xchesscc --no-xbridge --aie-generate-pdi --pdi-name=MlirAie1.pdi %s | FileCheck %s --check-prefix=PEANO

// RUN: ls | grep MlirAie | FileCheck %s --check-prefix=CHECK-FILE

// XCHESSCC: bootgen {{.*}} MlirAie0.pdi
// XCHESSCC-NOT: xclbinutil

// PEANO: bootgen {{.*}} MlirAie1.pdi
// PEANO-NOT: xclbinutil

// CHECK-FILE: MlirAie0.pdi
// CHECK-FILE: MlirAie1.pdi

module {
  aie.device(npu1) {
  %12 = aie.tile(1, 2)
  %buf = aie.buffer(%12) : memref<256xi32>
  %4 = aie.core(%12)  {
    %0 = arith.constant 0 : i32
    %1 = arith.constant 0 : index
    memref.store %0, %buf[%1] : memref<256xi32>
    aie.end
  }
  }
}
