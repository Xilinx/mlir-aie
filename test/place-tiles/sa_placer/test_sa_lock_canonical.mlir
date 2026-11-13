//===- test_sa_lock_canonical.mlir -----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: %python %S/../../../utils/component_checks/sa_lock_check.py --aie-opt aie-opt --seed 1 %s | FileCheck %s

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
    aie.core(%t0) {
      %p0 = aie.objectfifo.acquire @f0(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f0(Produce, 1)
      %c0 = aie.objectfifo.acquire @f0(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f0(Consume, 1)
      %p1 = aie.objectfifo.acquire @f1(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f1(Produce, 1)
      %c1 = aie.objectfifo.acquire @f1(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f1(Consume, 1)
      %p2 = aie.objectfifo.acquire @f2(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f2(Produce, 1)
      %c2 = aie.objectfifo.acquire @f2(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f2(Consume, 1)
      %p3 = aie.objectfifo.acquire @f3(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f3(Produce, 1)
      %c3 = aie.objectfifo.acquire @f3(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f3(Consume, 1)
      %p36 = aie.objectfifo.acquire @f36(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f36(Produce, 1)
      %c36 = aie.objectfifo.acquire @f36(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f36(Consume, 1)
      %p37 = aie.objectfifo.acquire @f37(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f37(Produce, 1)
      %c37 = aie.objectfifo.acquire @f37(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f37(Consume, 1)
      %p38 = aie.objectfifo.acquire @f38(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f38(Produce, 1)
      %c38 = aie.objectfifo.acquire @f38(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f38(Consume, 1)
      aie.end
    }
    aie.core(%t1) {
      %p39 = aie.objectfifo.acquire @f39(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f39(Produce, 1)
      %c39 = aie.objectfifo.acquire @f39(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f39(Consume, 1)
      %p40 = aie.objectfifo.acquire @f40(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f40(Produce, 1)
      %c40 = aie.objectfifo.acquire @f40(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f40(Consume, 1)
      %p41 = aie.objectfifo.acquire @f41(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f41(Produce, 1)
      %c41 = aie.objectfifo.acquire @f41(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f41(Consume, 1)
      %p42 = aie.objectfifo.acquire @f42(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f42(Produce, 1)
      %c42 = aie.objectfifo.acquire @f42(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f42(Consume, 1)
      %p45 = aie.objectfifo.acquire @f45(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f45(Produce, 1)
      %c45 = aie.objectfifo.acquire @f45(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f45(Consume, 1)
      %p46 = aie.objectfifo.acquire @f46(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f46(Produce, 1)
      %c46 = aie.objectfifo.acquire @f46(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f46(Consume, 1)
      aie.end
    }
    aie.core(%t2) {
      %p43 = aie.objectfifo.acquire @f43(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f43(Produce, 1)
      %c43 = aie.objectfifo.acquire @f43(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f43(Consume, 1)
      aie.end
    }
    aie.core(%t3) {
      %p4 = aie.objectfifo.acquire @f4(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f4(Produce, 1)
      %c4 = aie.objectfifo.acquire @f4(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f4(Consume, 1)
      %p5 = aie.objectfifo.acquire @f5(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f5(Produce, 1)
      %c5 = aie.objectfifo.acquire @f5(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f5(Consume, 1)
      aie.end
    }
    aie.core(%t4) {
      %p6 = aie.objectfifo.acquire @f6(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f6(Produce, 1)
      %c6 = aie.objectfifo.acquire @f6(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f6(Consume, 1)
      %p7 = aie.objectfifo.acquire @f7(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f7(Produce, 1)
      %c7 = aie.objectfifo.acquire @f7(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f7(Consume, 1)
      %p8 = aie.objectfifo.acquire @f8(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f8(Produce, 1)
      %c8 = aie.objectfifo.acquire @f8(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f8(Consume, 1)
      %p44 = aie.objectfifo.acquire @f44(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f44(Produce, 1)
      %c44 = aie.objectfifo.acquire @f44(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f44(Consume, 1)
      aie.end
    }
    aie.core(%t5) {
      %p47 = aie.objectfifo.acquire @f47(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f47(Produce, 1)
      %c47 = aie.objectfifo.acquire @f47(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f47(Consume, 1)
      %p48 = aie.objectfifo.acquire @f48(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f48(Produce, 1)
      %c48 = aie.objectfifo.acquire @f48(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f48(Consume, 1)
      %p49 = aie.objectfifo.acquire @f49(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f49(Produce, 1)
      %c49 = aie.objectfifo.acquire @f49(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f49(Consume, 1)
      %p53 = aie.objectfifo.acquire @f53(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f53(Produce, 1)
      %c53 = aie.objectfifo.acquire @f53(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f53(Consume, 1)
      %p54 = aie.objectfifo.acquire @f54(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f54(Produce, 1)
      %c54 = aie.objectfifo.acquire @f54(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f54(Consume, 1)
      aie.end
    }
    aie.core(%t6) {
      %p50 = aie.objectfifo.acquire @f50(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f50(Produce, 1)
      %c50 = aie.objectfifo.acquire @f50(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f50(Consume, 1)
      %p51 = aie.objectfifo.acquire @f51(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f51(Produce, 1)
      %c51 = aie.objectfifo.acquire @f51(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f51(Consume, 1)
      %p52 = aie.objectfifo.acquire @f52(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f52(Produce, 1)
      %c52 = aie.objectfifo.acquire @f52(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f52(Consume, 1)
      aie.end
    }
    aie.core(%t7) {
      %p60 = aie.objectfifo.acquire @f60(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f60(Produce, 1)
      %c60 = aie.objectfifo.acquire @f60(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f60(Consume, 1)
      %p61 = aie.objectfifo.acquire @f61(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f61(Produce, 1)
      %c61 = aie.objectfifo.acquire @f61(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f61(Consume, 1)
      aie.end
    }
    aie.core(%t8) {
      %p9 = aie.objectfifo.acquire @f9(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f9(Produce, 1)
      %c9 = aie.objectfifo.acquire @f9(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f9(Consume, 1)
      %p10 = aie.objectfifo.acquire @f10(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f10(Produce, 1)
      %c10 = aie.objectfifo.acquire @f10(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f10(Consume, 1)
      %p11 = aie.objectfifo.acquire @f11(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f11(Produce, 1)
      %c11 = aie.objectfifo.acquire @f11(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f11(Consume, 1)
      %p12 = aie.objectfifo.acquire @f12(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f12(Produce, 1)
      %c12 = aie.objectfifo.acquire @f12(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f12(Consume, 1)
      %p55 = aie.objectfifo.acquire @f55(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f55(Produce, 1)
      %c55 = aie.objectfifo.acquire @f55(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f55(Consume, 1)
      aie.end
    }
    aie.core(%t9) {
      %p56 = aie.objectfifo.acquire @f56(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f56(Produce, 1)
      %c56 = aie.objectfifo.acquire @f56(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f56(Consume, 1)
      %p57 = aie.objectfifo.acquire @f57(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f57(Produce, 1)
      %c57 = aie.objectfifo.acquire @f57(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f57(Consume, 1)
      %p58 = aie.objectfifo.acquire @f58(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f58(Produce, 1)
      %c58 = aie.objectfifo.acquire @f58(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f58(Consume, 1)
      %p59 = aie.objectfifo.acquire @f59(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f59(Produce, 1)
      %c59 = aie.objectfifo.acquire @f59(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f59(Consume, 1)
      %p62 = aie.objectfifo.acquire @f62(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f62(Produce, 1)
      %c62 = aie.objectfifo.acquire @f62(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f62(Consume, 1)
      %p64 = aie.objectfifo.acquire @f64(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f64(Produce, 1)
      %c64 = aie.objectfifo.acquire @f64(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f64(Consume, 1)
      aie.end
    }
    aie.core(%t10) {
      %p13 = aie.objectfifo.acquire @f13(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f13(Produce, 1)
      %c13 = aie.objectfifo.acquire @f13(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f13(Consume, 1)
      aie.end
    }
    aie.core(%t11) {
      %p14 = aie.objectfifo.acquire @f14(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f14(Produce, 1)
      %c14 = aie.objectfifo.acquire @f14(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f14(Consume, 1)
      %p69 = aie.objectfifo.acquire @f69(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f69(Produce, 1)
      %c69 = aie.objectfifo.acquire @f69(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f69(Consume, 1)
      aie.end
    }
    aie.core(%t12) {
      %p15 = aie.objectfifo.acquire @f15(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f15(Produce, 1)
      %c15 = aie.objectfifo.acquire @f15(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f15(Consume, 1)
      %p16 = aie.objectfifo.acquire @f16(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f16(Produce, 1)
      %c16 = aie.objectfifo.acquire @f16(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f16(Consume, 1)
      %p17 = aie.objectfifo.acquire @f17(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f17(Produce, 1)
      %c17 = aie.objectfifo.acquire @f17(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f17(Consume, 1)
      %p63 = aie.objectfifo.acquire @f63(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f63(Produce, 1)
      %c63 = aie.objectfifo.acquire @f63(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f63(Consume, 1)
      %p65 = aie.objectfifo.acquire @f65(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f65(Produce, 1)
      %c65 = aie.objectfifo.acquire @f65(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f65(Consume, 1)
      aie.end
    }
    aie.core(%t13) {
      %p66 = aie.objectfifo.acquire @f66(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f66(Produce, 1)
      %c66 = aie.objectfifo.acquire @f66(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f66(Consume, 1)
      %p67 = aie.objectfifo.acquire @f67(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f67(Produce, 1)
      %c67 = aie.objectfifo.acquire @f67(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f67(Consume, 1)
      %p68 = aie.objectfifo.acquire @f68(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f68(Produce, 1)
      %c68 = aie.objectfifo.acquire @f68(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f68(Consume, 1)
      %p73 = aie.objectfifo.acquire @f73(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f73(Produce, 1)
      %c73 = aie.objectfifo.acquire @f73(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f73(Consume, 1)
      aie.end
    }
    aie.core(%t14) {
      %p70 = aie.objectfifo.acquire @f70(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f70(Produce, 1)
      %c70 = aie.objectfifo.acquire @f70(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f70(Consume, 1)
      %p71 = aie.objectfifo.acquire @f71(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f71(Produce, 1)
      %c71 = aie.objectfifo.acquire @f71(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f71(Consume, 1)
      aie.end
    }
    aie.core(%t15) {
      %p18 = aie.objectfifo.acquire @f18(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f18(Produce, 1)
      %c18 = aie.objectfifo.acquire @f18(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f18(Consume, 1)
      aie.end
    }
    aie.core(%t16) {
      %p19 = aie.objectfifo.acquire @f19(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f19(Produce, 1)
      %c19 = aie.objectfifo.acquire @f19(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f19(Consume, 1)
      %p20 = aie.objectfifo.acquire @f20(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f20(Produce, 1)
      %c20 = aie.objectfifo.acquire @f20(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f20(Consume, 1)
      %p21 = aie.objectfifo.acquire @f21(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f21(Produce, 1)
      %c21 = aie.objectfifo.acquire @f21(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f21(Consume, 1)
      %p72 = aie.objectfifo.acquire @f72(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f72(Produce, 1)
      %c72 = aie.objectfifo.acquire @f72(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f72(Consume, 1)
      %p74 = aie.objectfifo.acquire @f74(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f74(Produce, 1)
      %c74 = aie.objectfifo.acquire @f74(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f74(Consume, 1)
      aie.end
    }
    aie.core(%t17) {
      %p22 = aie.objectfifo.acquire @f22(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f22(Produce, 1)
      %c22 = aie.objectfifo.acquire @f22(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f22(Consume, 1)
      %p75 = aie.objectfifo.acquire @f75(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f75(Produce, 1)
      %c75 = aie.objectfifo.acquire @f75(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f75(Consume, 1)
      %p76 = aie.objectfifo.acquire @f76(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f76(Produce, 1)
      %c76 = aie.objectfifo.acquire @f76(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f76(Consume, 1)
      %p77 = aie.objectfifo.acquire @f77(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f77(Produce, 1)
      %c77 = aie.objectfifo.acquire @f77(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f77(Consume, 1)
      %p83 = aie.objectfifo.acquire @f83(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f83(Produce, 1)
      %c83 = aie.objectfifo.acquire @f83(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f83(Consume, 1)
      aie.end
    }
    aie.core(%t18) {
      %p78 = aie.objectfifo.acquire @f78(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f78(Produce, 1)
      %c78 = aie.objectfifo.acquire @f78(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f78(Consume, 1)
      %p79 = aie.objectfifo.acquire @f79(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f79(Produce, 1)
      %c79 = aie.objectfifo.acquire @f79(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f79(Consume, 1)
      %p80 = aie.objectfifo.acquire @f80(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f80(Produce, 1)
      %c80 = aie.objectfifo.acquire @f80(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f80(Consume, 1)
      %p81 = aie.objectfifo.acquire @f81(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f81(Produce, 1)
      %c81 = aie.objectfifo.acquire @f81(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f81(Consume, 1)
      aie.end
    }
    aie.core(%t19) {
      %p88 = aie.objectfifo.acquire @f88(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f88(Produce, 1)
      %c88 = aie.objectfifo.acquire @f88(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f88(Consume, 1)
      %p90 = aie.objectfifo.acquire @f90(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f90(Produce, 1)
      %c90 = aie.objectfifo.acquire @f90(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f90(Consume, 1)
      aie.end
    }
    aie.core(%t20) {
      %p23 = aie.objectfifo.acquire @f23(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f23(Produce, 1)
      %c23 = aie.objectfifo.acquire @f23(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f23(Consume, 1)
      %p24 = aie.objectfifo.acquire @f24(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f24(Produce, 1)
      %c24 = aie.objectfifo.acquire @f24(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f24(Consume, 1)
      %p25 = aie.objectfifo.acquire @f25(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f25(Produce, 1)
      %c25 = aie.objectfifo.acquire @f25(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f25(Consume, 1)
      %p26 = aie.objectfifo.acquire @f26(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f26(Produce, 1)
      %c26 = aie.objectfifo.acquire @f26(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f26(Consume, 1)
      %p82 = aie.objectfifo.acquire @f82(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f82(Produce, 1)
      %c82 = aie.objectfifo.acquire @f82(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f82(Consume, 1)
      %p84 = aie.objectfifo.acquire @f84(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f84(Produce, 1)
      %c84 = aie.objectfifo.acquire @f84(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f84(Consume, 1)
      aie.end
    }
    aie.core(%t21) {
      %p85 = aie.objectfifo.acquire @f85(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f85(Produce, 1)
      %c85 = aie.objectfifo.acquire @f85(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f85(Consume, 1)
      %p86 = aie.objectfifo.acquire @f86(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f86(Produce, 1)
      %c86 = aie.objectfifo.acquire @f86(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f86(Consume, 1)
      %p87 = aie.objectfifo.acquire @f87(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f87(Produce, 1)
      %c87 = aie.objectfifo.acquire @f87(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f87(Consume, 1)
      %p91 = aie.objectfifo.acquire @f91(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f91(Produce, 1)
      %c91 = aie.objectfifo.acquire @f91(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f91(Consume, 1)
      %p92 = aie.objectfifo.acquire @f92(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f92(Produce, 1)
      %c92 = aie.objectfifo.acquire @f92(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f92(Consume, 1)
      aie.end
    }
    aie.core(%t22) {
      %p89 = aie.objectfifo.acquire @f89(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f89(Produce, 1)
      %c89 = aie.objectfifo.acquire @f89(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f89(Consume, 1)
      aie.end
    }
    aie.core(%t23) {
      %p97 = aie.objectfifo.acquire @f97(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f97(Produce, 1)
      %c97 = aie.objectfifo.acquire @f97(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f97(Consume, 1)
      %p98 = aie.objectfifo.acquire @f98(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f98(Produce, 1)
      %c98 = aie.objectfifo.acquire @f98(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f98(Consume, 1)
      %p99 = aie.objectfifo.acquire @f99(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f99(Produce, 1)
      %c99 = aie.objectfifo.acquire @f99(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f99(Consume, 1)
      aie.end
    }
    aie.core(%t24) {
      %p27 = aie.objectfifo.acquire @f27(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f27(Produce, 1)
      %c27 = aie.objectfifo.acquire @f27(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f27(Consume, 1)
      %p28 = aie.objectfifo.acquire @f28(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f28(Produce, 1)
      %c28 = aie.objectfifo.acquire @f28(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f28(Consume, 1)
      %p29 = aie.objectfifo.acquire @f29(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f29(Produce, 1)
      %c29 = aie.objectfifo.acquire @f29(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f29(Consume, 1)
      %p30 = aie.objectfifo.acquire @f30(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f30(Produce, 1)
      %c30 = aie.objectfifo.acquire @f30(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f30(Consume, 1)
      aie.end
    }
    aie.core(%t25) {
      %p31 = aie.objectfifo.acquire @f31(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f31(Produce, 1)
      %c31 = aie.objectfifo.acquire @f31(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f31(Consume, 1)
      %p93 = aie.objectfifo.acquire @f93(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f93(Produce, 1)
      %c93 = aie.objectfifo.acquire @f93(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f93(Consume, 1)
      %p94 = aie.objectfifo.acquire @f94(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f94(Produce, 1)
      %c94 = aie.objectfifo.acquire @f94(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f94(Consume, 1)
      %p95 = aie.objectfifo.acquire @f95(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f95(Produce, 1)
      %c95 = aie.objectfifo.acquire @f95(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f95(Consume, 1)
      %p96 = aie.objectfifo.acquire @f96(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f96(Produce, 1)
      %c96 = aie.objectfifo.acquire @f96(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f96(Consume, 1)
      %p100 = aie.objectfifo.acquire @f100(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f100(Produce, 1)
      %c100 = aie.objectfifo.acquire @f100(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f100(Consume, 1)
      %p101 = aie.objectfifo.acquire @f101(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f101(Produce, 1)
      %c101 = aie.objectfifo.acquire @f101(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f101(Consume, 1)
      aie.end
    }
    aie.core(%t26) {
      aie.end
    }
    aie.core(%t27) {
      %p32 = aie.objectfifo.acquire @f32(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f32(Produce, 1)
      %c32 = aie.objectfifo.acquire @f32(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f32(Consume, 1)
      %p109 = aie.objectfifo.acquire @f109(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f109(Produce, 1)
      %c109 = aie.objectfifo.acquire @f109(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f109(Consume, 1)
      aie.end
    }
    aie.core(%t28) {
      %p33 = aie.objectfifo.acquire @f33(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f33(Produce, 1)
      %c33 = aie.objectfifo.acquire @f33(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f33(Consume, 1)
      %p34 = aie.objectfifo.acquire @f34(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f34(Produce, 1)
      %c34 = aie.objectfifo.acquire @f34(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f34(Consume, 1)
      %p35 = aie.objectfifo.acquire @f35(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f35(Produce, 1)
      %c35 = aie.objectfifo.acquire @f35(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f35(Consume, 1)
      %p102 = aie.objectfifo.acquire @f102(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f102(Produce, 1)
      %c102 = aie.objectfifo.acquire @f102(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f102(Consume, 1)
      aie.end
    }
    aie.core(%t29) {
      %p103 = aie.objectfifo.acquire @f103(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f103(Produce, 1)
      %c103 = aie.objectfifo.acquire @f103(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f103(Consume, 1)
      %p104 = aie.objectfifo.acquire @f104(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f104(Produce, 1)
      %c104 = aie.objectfifo.acquire @f104(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f104(Consume, 1)
      %p105 = aie.objectfifo.acquire @f105(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f105(Produce, 1)
      %c105 = aie.objectfifo.acquire @f105(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f105(Consume, 1)
      aie.end
    }
    aie.core(%t30) {
      %p106 = aie.objectfifo.acquire @f106(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f106(Produce, 1)
      %c106 = aie.objectfifo.acquire @f106(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f106(Consume, 1)
      %p107 = aie.objectfifo.acquire @f107(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f107(Produce, 1)
      %c107 = aie.objectfifo.acquire @f107(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f107(Consume, 1)
      %p108 = aie.objectfifo.acquire @f108(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f108(Produce, 1)
      %c108 = aie.objectfifo.acquire @f108(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f108(Consume, 1)
      aie.end
    }
    aie.core(%t31) {
      aie.end
    }
  }
}
