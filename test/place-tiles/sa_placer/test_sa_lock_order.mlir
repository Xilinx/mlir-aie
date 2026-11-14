//===- test_sa_lock_order.mlir ---------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: %python %S/../../../utils/generate_self_loop_cores.py %s > %t.mlir
// RUN: %python %S/../../../utils/component_checks/sa_lock_check.py --aie-opt aie-opt --seed 1 %t.mlir | FileCheck %s

// The SA lists the lock groups in fifo order, allocation in its structural
// order. The 6x4 grid's pools of 1 to 7 locks take 376 of its 384, and how
// many steps the exact packing takes depends on the order it searches in.
// The packing searches in an order of its own, so the SA and allocation
// reach the same verdict: the locks fit.
// CHECK: agree=1 mismatch=0 skipped=0 agreeing_lock_failures=0 agreeing_lock_stops=0

module @lock_order {
  aie.device(xcvc1902) {
    %ls = aie.logical_tile<ShimNOCTile>(?, ?)
    %t0 = aie.tile(5, 1)
    %t1 = aie.tile(5, 2)
    %t2 = aie.tile(5, 3)
    %t3 = aie.tile(5, 4)
    %t4 = aie.tile(6, 1)
    %t5 = aie.tile(6, 2)
    %t6 = aie.tile(6, 3)
    %t7 = aie.tile(6, 4)
    %t8 = aie.tile(7, 1)
    %t9 = aie.tile(7, 2)
    %t10 = aie.tile(7, 3)
    %t11 = aie.tile(7, 4)
    %t12 = aie.tile(8, 1)
    %t13 = aie.tile(8, 2)
    %t14 = aie.tile(8, 3)
    %t15 = aie.tile(8, 4)
    %t16 = aie.tile(9, 1)
    %t17 = aie.tile(9, 2)
    %t18 = aie.tile(9, 3)
    %t19 = aie.tile(9, 4)
    %t20 = aie.tile(10, 1)
    %t21 = aie.tile(10, 2)
    %t22 = aie.tile(10, 3)
    %t23 = aie.tile(10, 4)
    aie.objectfifo @f0(%t11, {%t11}, 4 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f1(%t4, {%t4}, 6 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f2(%t22, {%t22}, 4 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f3(%t2, {%t2}, 3 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f4(%t7, {%t7}, 4 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f5(%t6, {%t6}, 7 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f6(%t20, {%t20}, 6 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f7(%t14, {%t14}, 3 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f8(%t14, {%t14}, 6 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f9(%t15, {%t15}, 6 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f10(%t6, {%t6}, 7 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f11(%t2, {%t2}, 6 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f12(%t7, {%t7}, 6 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f13(%t22, {%t22}, 3 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f14(%t16, {%t16}, 5 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f15(%t15, {%t15}, 6 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f16(%t23, {%t23}, 6 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f17(%t21, {%t21}, 3 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f18(%t3, {%t3}, 5 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f19(%t12, {%t12}, 3 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f20(%t12, {%t12}, 7 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f21(%t21, {%t21}, 3 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f22(%t10, {%t10}, 3 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f23(%t22, {%t22}, 4 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f24(%t15, {%t15}, 3 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f25(%t6, {%t6}, 7 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f26(%t4, {%t4}, 7 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f27(%t2, {%t2}, 7 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f28(%t1, {%t1}, 7 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f29(%t22, {%t22}, 6 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f30(%t4, {%t4}, 4 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f31(%t14, {%t14}, 7 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f32(%t14, {%t14}, 7 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f33(%t17, {%t17}, 5 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f34(%t13, {%t13}, 5 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f35(%t5, {%t5}, 4 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f36(%t3, {%t3}, 7 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f37(%t22, {%t22}, 7 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f38(%t22, {%t22}, 5 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f39(%t20, {%t20}, 5 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f40(%t16, {%t16}, 6 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f41(%t6, {%t6}, 7 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f42(%t4, {%t4}, 5 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f43(%t16, {%t16}, 5 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f44(%t16, {%t16}, 5 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f45(%t16, {%t16}, 3 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f46(%t7, {%t7}, 7 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f47(%t7, {%t7}, 5 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f48(%t9, {%t9}, 3 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f49(%t7, {%t7}, 5 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f50(%t1, {%t1}, 5 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f51(%t8, {%t8}, 6 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f52(%t9, {%t9}, 6 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f53(%t5, {%t5}, 6 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f54(%t3, {%t3}, 6 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f55(%t21, {%t21}, 4 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f56(%t6, {%t6}, 3 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f57(%t9, {%t9}, 4 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f58(%t0, {%t0}, 3 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f59(%t10, {%t10}, 6 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f60(%t18, {%t18}, 4 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f61(%t7, {%t7}, 6 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f62(%t13, {%t13}, 4 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f63(%t14, {%t14}, 7 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f64(%t13, {%t13}, 6 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f65(%t18, {%t18}, 3 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f66(%t9, {%t9}, 3 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f67(%t1, {%t1}, 3 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f68(%t21, {%t21}, 3 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f69(%t16, {%t16}, 3 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f70(%t11, {%t11}, 7 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f71(%t16, {%t16}, 4 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f72(%t22, {%t22}, 6 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f73(%t19, {%t19}, 4 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f74(%t6, {%t6}, 3 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f75(%t4, {%t4}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
  }
}
