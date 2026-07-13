//===- test_lock0.mlir -----------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2021-2022 Xilinx, Inc.
// Copyright (C) 2022-2023 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-create-locks %s | FileCheck %s

// CHECK-LABEL: module @test_lock0 {
// CHECK:         aie.device(xcvc1902) {
// CHECK:           %[[VAL_0:.*]] = aie.tile(3, 3)
// CHECK:           %[[VAL_1:.*]] = aie.tile(2, 3)
// CHECK:           %[[VAL_2:.*]] = aie.lock(%[[VAL_1]], 0) {init = 0 : i32}
// CHECK:           aiex.token(0) {sym_name = "token0"}
// CHECK:           %[[VAL_3:.*]] = aie.mem(%[[VAL_0]]) {
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_4:.*]] = aie.mem(%[[VAL_1]]) {
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_5:.*]] = aie.core(%[[VAL_0]]) {
// CHECK:             aiex.useToken @token0(Acquire, 0)
// CHECK:             aiex.useToken @token0(Release, 1)
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_6:.*]] = aie.core(%[[VAL_1]]) {
// CHECK:             aiex.useToken @token0(Acquire, 1)
// CHECK:             aiex.useToken @token0(Release, 2)
// CHECK:             aie.end
// CHECK:           }
// CHECK:         }
// CHECK:       }

module @test_lock0 {
 aie.device(xcvc1902) {
  %t33 = aie.tile(3, 3)
  %t23 = aie.tile(2, 3)

  aiex.token(0) {sym_name = "token0"}

  %m33 = aie.mem(%t33) {
      aie.end
  }

  %m23 = aie.mem(%t23) {
      aie.end
  }

  %c33 = aie.core(%t33) {
    aiex.useToken @token0(Acquire, 0)
    aiex.useToken @token0(Release, 1)
    aie.end
  }

  %c23 = aie.core(%t23) {
    aiex.useToken @token0(Acquire, 1)
    aiex.useToken @token0(Release, 2)
    aie.end
  }
 }
}
