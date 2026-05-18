//===- objectfifo_array_depth_init_values_roundtrip.mlir -------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Round-trip an aie.objectfifo with array-valued $elemNumber (per-endpoint
// depths) AND initValues. Verifies printObjectFifoInitValues handles both
// the IntegerAttr (scalar) and ArrayAttr (list) shapes of $elemNumber.

// RUN: aie-opt --canonicalize %s | FileCheck %s

// CHECK-LABEL: aie.device(npu2)
// CHECK: aie.objectfifo @of_array_depth(%{{.*}}, {%{{.*}}}, [2 : i32, 1 : i32]) : !aie.objectfifo<memref<16xi32>> = [dense<0> : memref<16xi32>, dense<1> : memref<16xi32>]

module @objectfifo_array_depth_initvalues_roundtrip {
  aie.device(npu2) {
    %mem = aie.tile(0, 1)
    %compute = aie.tile(0, 2)
    aie.objectfifo @of_array_depth(%mem, {%compute}, [2 : i32, 1 : i32])
      : !aie.objectfifo<memref<16xi32>>
      = [dense<0> : memref<16xi32>, dense<1> : memref<16xi32>]
  }
}
