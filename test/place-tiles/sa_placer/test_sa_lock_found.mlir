//===- test_sa_lock_found.mlir ---------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: %python %S/../../../utils/generate_self_loop_cores.py %s > %t.mlir
// RUN: %python %S/../../../utils/component_checks/sa_lock_check.py --aie-opt aie-opt --seed 1 %t.mlir | FileCheck %s

// The 5x4 grid's 66 pools of 1 to 7 locks take 317 of its 320, and only the
// exact packing fits them. The unused logical shim makes this an SA input;
// the SA records the packing it finds, so its moves do not repeat the
// search, and it agrees with allocation that the locks fit.
// CHECK: agree=1 mismatch=0 skipped=0 agreeing_lock_failures=0 agreeing_lock_stops=0

module @lock_found {
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
    aie.objectfifo @f0(%t18, {%t18}, 6 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f1(%t6, {%t6}, 4 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f2(%t6, {%t6}, 4 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f3(%t2, {%t2}, 4 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f4(%t9, {%t9}, 4 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f5(%t14, {%t14}, 3 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f6(%t3, {%t3}, 6 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f7(%t16, {%t16}, 3 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f8(%t15, {%t15}, 4 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f9(%t9, {%t9}, 6 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f10(%t2, {%t2}, 6 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f11(%t19, {%t19}, 5 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f12(%t10, {%t10}, 4 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f13(%t2, {%t2}, 5 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f14(%t11, {%t11}, 7 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f15(%t6, {%t6}, 3 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f16(%t10, {%t10}, 5 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f17(%t9, {%t9}, 3 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f18(%t7, {%t7}, 3 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f19(%t4, {%t4}, 7 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f20(%t9, {%t9}, 6 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f21(%t17, {%t17}, 5 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f22(%t6, {%t6}, 5 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f23(%t18, {%t18}, 3 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f24(%t4, {%t4}, 6 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f25(%t10, {%t10}, 4 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f26(%t6, {%t6}, 6 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f27(%t4, {%t4}, 7 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f28(%t16, {%t16}, 6 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f29(%t16, {%t16}, 5 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f30(%t17, {%t17}, 4 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f31(%t2, {%t2}, 4 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f32(%t8, {%t8}, 4 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f33(%t17, {%t17}, 5 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f34(%t12, {%t12}, 4 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f35(%t9, {%t9}, 4 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f36(%t2, {%t2}, 6 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f37(%t11, {%t11}, 6 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f38(%t8, {%t8}, 5 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f39(%t18, {%t18}, 4 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f40(%t10, {%t10}, 6 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f41(%t11, {%t11}, 4 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f42(%t1, {%t1}, 3 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f43(%t10, {%t10}, 6 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f44(%t14, {%t14}, 6 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f45(%t1, {%t1}, 3 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f46(%t10, {%t10}, 3 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f47(%t12, {%t12}, 6 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f48(%t15, {%t15}, 7 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f49(%t16, {%t16}, 7 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f50(%t12, {%t12}, 6 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f51(%t9, {%t9}, 4 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f52(%t11, {%t11}, 5 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f53(%t4, {%t4}, 5 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f54(%t2, {%t2}, 5 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f55(%t0, {%t0}, 7 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f56(%t17, {%t17}, 5 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f57(%t13, {%t13}, 3 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f58(%t12, {%t12}, 6 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f59(%t8, {%t8}, 5 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f60(%t12, {%t12}, 6 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f61(%t10, {%t10}, 5 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f62(%t12, {%t12}, 3 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f63(%t15, {%t15}, 3 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f64(%t2, {%t2}, 6 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f65(%t7, {%t7}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
  }
}
