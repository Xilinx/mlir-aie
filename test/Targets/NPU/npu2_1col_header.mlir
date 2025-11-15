//===- npu2_instgen.mlir ---------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2025 Advanced Micro Devices, Inc. or its affiliates
//
//===----------------------------------------------------------------------===//

// RUN: aie-translate --aie-npu-to-binary -aie-output-binary=false %s | FileCheck %s
module {
  aie.device(npu2_1col) {
    aie.runtime_sequence() {

      // TXN header 0.1

      // CHECK: 06040100
      //         ^ ^ ^ ^
      //         | | | |
      //         | | | txn bin verison maj
      //         | | txn bin version min
      //         | device generation
      //         num rows

      // CHECK: 00000101
      //             ^ ^
      //             | |
      //             | num columns
      //             num mem tile rows

      // CHECK: 00000000
      //        number of instructions in this transaction binary

      // CHECK: 00000010
      //        number of bytes of the whole transaction binary, including header

      aie.end
    }
  }
}
