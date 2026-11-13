//===- test_sa_lock_canonical.mlir -----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: %python %S/../../../utils/generate_self_loop_cores.py %s > %t.mlir
// RUN: %python %S/../../../utils/component_checks/sa_lock_check.py --aie-opt aie-opt --seed 1 %t.mlir | FileCheck %s

// An 8x4 grid of 110 pools of 3 to 7 locks whose fifo order is a packing.
// The SA and allocation place first fit in one order taken from the lock
// groups alone, not fifo order, so the SA fits only where allocation does:
// here first fit misses, the exact packing runs out of steps, and both report
// it stopped. The unused logical shim makes this an SA input; the SA records
// the stopped packing, so its moves do not repeat the search.
// CHECK: agree=1 mismatch=0 skipped=0 agreeing_lock_failures=0 agreeing_lock_stops=1

module @lock_canonical {
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
    %t24 = aie.tile(11, 1)
    %t25 = aie.tile(11, 2)
    %t26 = aie.tile(11, 3)
    %t27 = aie.tile(11, 4)
    %t28 = aie.tile(12, 1)
    %t29 = aie.tile(12, 2)
    %t30 = aie.tile(12, 3)
    %t31 = aie.tile(12, 4)
    aie.objectfifo @f0(%t0, {%t0}, 3 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f1(%t0, {%t0}, 3 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f2(%t0, {%t0}, 5 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f3(%t0, {%t0}, 4 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f4(%t3, {%t3}, 4 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f5(%t3, {%t3}, 5 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f6(%t4, {%t4}, 5 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f7(%t4, {%t4}, 7 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f8(%t4, {%t4}, 3 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f9(%t8, {%t8}, 3 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f10(%t8, {%t8}, 7 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f11(%t8, {%t8}, 3 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f12(%t8, {%t8}, 3 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f13(%t10, {%t10}, 3 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f14(%t11, {%t11}, 6 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f15(%t12, {%t12}, 7 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f16(%t12, {%t12}, 3 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f17(%t12, {%t12}, 4 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f18(%t15, {%t15}, 4 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f19(%t16, {%t16}, 5 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f20(%t16, {%t16}, 6 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f21(%t16, {%t16}, 5 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f22(%t17, {%t17}, 5 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f23(%t20, {%t20}, 5 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f24(%t20, {%t20}, 4 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f25(%t20, {%t20}, 3 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f26(%t20, {%t20}, 3 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f27(%t24, {%t24}, 3 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f28(%t24, {%t24}, 4 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f29(%t24, {%t24}, 4 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f30(%t24, {%t24}, 3 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f31(%t25, {%t25}, 5 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f32(%t27, {%t27}, 4 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f33(%t28, {%t28}, 7 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f34(%t28, {%t28}, 4 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f35(%t28, {%t28}, 3 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f36(%t0, {%t0}, 6 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f37(%t0, {%t0}, 4 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f38(%t0, {%t0}, 5 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f39(%t1, {%t1}, 3 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f40(%t1, {%t1}, 6 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f41(%t1, {%t1}, 3 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f42(%t1, {%t1}, 4 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f43(%t2, {%t2}, 5 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f44(%t4, {%t4}, 6 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f45(%t1, {%t1}, 7 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f46(%t1, {%t1}, 3 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f47(%t5, {%t5}, 5 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f48(%t5, {%t5}, 3 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f49(%t5, {%t5}, 6 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f50(%t6, {%t6}, 5 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f51(%t6, {%t6}, 5 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f52(%t6, {%t6}, 4 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f53(%t5, {%t5}, 5 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f54(%t5, {%t5}, 6 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f55(%t8, {%t8}, 5 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f56(%t9, {%t9}, 3 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f57(%t9, {%t9}, 3 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f58(%t9, {%t9}, 3 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f59(%t9, {%t9}, 3 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f60(%t7, {%t7}, 5 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f61(%t7, {%t7}, 5 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f62(%t9, {%t9}, 3 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f63(%t12, {%t12}, 6 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f64(%t9, {%t9}, 3 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f65(%t12, {%t12}, 3 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f66(%t13, {%t13}, 5 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f67(%t13, {%t13}, 5 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f68(%t13, {%t13}, 5 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f69(%t11, {%t11}, 3 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f70(%t14, {%t14}, 4 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f71(%t14, {%t14}, 5 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f72(%t16, {%t16}, 4 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f73(%t13, {%t13}, 4 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f74(%t16, {%t16}, 3 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f75(%t17, {%t17}, 5 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f76(%t17, {%t17}, 7 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f77(%t17, {%t17}, 3 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f78(%t18, {%t18}, 3 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f79(%t18, {%t18}, 6 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f80(%t18, {%t18}, 4 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f81(%t18, {%t18}, 3 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f82(%t20, {%t20}, 6 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f83(%t17, {%t17}, 5 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f84(%t20, {%t20}, 3 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f85(%t21, {%t21}, 7 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f86(%t21, {%t21}, 3 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f87(%t21, {%t21}, 6 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f88(%t19, {%t19}, 7 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f89(%t22, {%t22}, 6 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f90(%t19, {%t19}, 3 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f91(%t21, {%t21}, 5 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f92(%t21, {%t21}, 4 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f93(%t25, {%t25}, 3 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f94(%t25, {%t25}, 5 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f95(%t25, {%t25}, 5 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f96(%t25, {%t25}, 3 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f97(%t23, {%t23}, 3 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f98(%t23, {%t23}, 4 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f99(%t23, {%t23}, 4 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f100(%t25, {%t25}, 5 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f101(%t25, {%t25}, 5 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f102(%t28, {%t28}, 4 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f103(%t29, {%t29}, 6 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f104(%t29, {%t29}, 7 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f105(%t29, {%t29}, 3 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f106(%t30, {%t30}, 5 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f107(%t30, {%t30}, 4 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f108(%t30, {%t30}, 4 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f109(%t27, {%t27}, 3 : i32) : !aie.objectfifo<memref<16xi32>>
  }
}
