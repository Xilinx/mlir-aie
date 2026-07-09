//===- locks-ve2802.mlir ---------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2023 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-localize-locks %s | FileCheck %s

// CHECK-LABEL:   aie.device(xcve2802) {
// CHECK:           %[[VAL_0:.*]] = aie.tile(1, 3)
// CHECK:           %[[VAL_1:.*]] = aie.tile(3, 5)
// CHECK:           %[[VAL_2:.*]] = aie.tile(3, 3)
// CHECK:           %[[VAL_3:.*]] = aie.tile(3, 4)
// CHECK:           %[[VAL_4:.*]] = aie.tile(4, 4)
// CHECK:           %[[VAL_5:.*]] = aie.lock(%[[VAL_0]], 0)
// CHECK:           %[[VAL_6:.*]] = aie.lock(%[[VAL_3]], 8)
// CHECK:           %[[VAL_7:.*]] = aie.lock(%[[VAL_4]], 8)
// CHECK:           %[[VAL_8:.*]] = aie.core(%[[VAL_0]]) {
// CHECK:             %[[VAL_9:.*]] = arith.constant 48 : index
// CHECK:             %{{.*}} = arith.constant 0 : i32
// CHECK:             aie.use_lock(%[[VAL_9]], Acquire, %{{.*}})
// CHECK:             %{{.*}} = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_9]], Release, %{{.*}})
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_10:.*]] = aie.core(%[[VAL_1]]) {
// CHECK:             %[[VAL_11:.*]] = arith.constant 8 : index
// CHECK:             %{{.*}} = arith.constant 0 : i32
// CHECK:             aie.use_lock(%[[VAL_11]], Acquire, %{{.*}})
// CHECK:             %{{.*}} = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_11]], Release, %{{.*}})
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_12:.*]] = aie.core(%[[VAL_2]]) {
// CHECK:             %[[VAL_13:.*]] = arith.constant 40 : index
// CHECK:             %{{.*}} = arith.constant 0 : i32
// CHECK:             aie.use_lock(%[[VAL_13]], Acquire, %{{.*}})
// CHECK:             %{{.*}} = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_13]], Release, %{{.*}})
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_14:.*]] = aie.core(%[[VAL_3]]) {
// CHECK:             %[[VAL_15:.*]] = arith.constant 56 : index
// CHECK:             %{{.*}} = arith.constant 0 : i32
// CHECK:             aie.use_lock(%[[VAL_15]], Acquire, %{{.*}})
// CHECK:             %{{.*}} = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_15]], Release, %{{.*}})
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_16:.*]] = aie.core(%[[VAL_4]]) {
// CHECK:             %[[VAL_17:.*]] = arith.constant 56 : index
// CHECK:             %[[VAL_18:.*]] = arith.constant 24 : index
// CHECK:             %{{.*}} = arith.constant 0 : i32
// CHECK:             aie.use_lock(%[[VAL_18]], Acquire, %{{.*}})
// CHECK:             %{{.*}} = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_18]], Release, %{{.*}})
// CHECK:             %{{.*}} = arith.constant 0 : i32
// CHECK:             aie.use_lock(%[[VAL_17]], Acquire, %{{.*}})
// CHECK:             %{{.*}} = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_17]], Release, %{{.*}})
// CHECK:             aie.end
// CHECK:           }
// CHECK:         }

module @test_xaie0 {
 aie.device(xcve2802) {
  %t13 = aie.tile(1, 3)
  %t35 = aie.tile(3, 5)
  %t33 = aie.tile(3, 3)
  %t34 = aie.tile(3, 4)
  %t44 = aie.tile(4, 4)

  %l11_8 = aie.lock(%t13, 0)
  %l33_8 = aie.lock(%t34, 8)
  %l43_8 = aie.lock(%t44, 8)

  aie.core(%t13) {
    %c0_ul0 = arith.constant 0 : i32
    aie.use_lock(%l11_8, Acquire, %c0_ul0)
    %c1_ul1 = arith.constant 1 : i32
    aie.use_lock(%l11_8, Release, %c1_ul1)
    aie.end
  }
  aie.core(%t35) {
    %c0_ul2 = arith.constant 0 : i32
    aie.use_lock(%l33_8, Acquire, %c0_ul2)
    %c1_ul3 = arith.constant 1 : i32
    aie.use_lock(%l33_8, Release, %c1_ul3)
    aie.end
  }
  aie.core(%t33) {
    %c0_ul4 = arith.constant 0 : i32
    aie.use_lock(%l33_8, Acquire, %c0_ul4)
    %c1_ul5 = arith.constant 1 : i32
    aie.use_lock(%l33_8, Release, %c1_ul5)
    aie.end
  }
  aie.core(%t34) {
    %c0_ul6 = arith.constant 0 : i32
    aie.use_lock(%l33_8, Acquire, %c0_ul6)
    %c1_ul7 = arith.constant 1 : i32
    aie.use_lock(%l33_8, Release, %c1_ul7)
    aie.end
  }
  aie.core(%t44) {
    %c0_ul8 = arith.constant 0 : i32
    aie.use_lock(%l33_8, Acquire, %c0_ul8)
    %c1_ul9 = arith.constant 1 : i32
    aie.use_lock(%l33_8, Release, %c1_ul9)
    %c0_ul10 = arith.constant 0 : i32
    aie.use_lock(%l43_8, Acquire, %c0_ul10)
    %c1_ul11 = arith.constant 1 : i32
    aie.use_lock(%l43_8, Release, %c1_ul11)
    aie.end
  }
 }
}
