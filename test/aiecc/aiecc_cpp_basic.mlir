//===- aiecc_cpp_basic.mlir ------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2025 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// Test basic functionality of C++ aiecc tool

// RUN: aiecc --version | FileCheck %s --check-prefix=VERSION
// RUN: aiecc --verbose -n %s | FileCheck %s --check-prefix=VERBOSE

// VERSION: aiecc (C++ version)
// VERBOSE: Successfully parsed input file
// VERBOSE: Using temporary directory

module {
  aie.device(xcvc1902) {
    %tile12 = aie.tile(1, 2)
    %buf = aie.buffer(%tile12) : memref<256xi32>
    %core = aie.core(%tile12) {
      %c0 = arith.constant 0 : i32
      %idx = arith.constant 0 : index
      memref.store %c0, %buf[%idx] : memref<256xi32>
      aie.end
    }
  }
}
