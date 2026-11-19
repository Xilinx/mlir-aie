//===- lock_packing_aie1_stopped.mlir --------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: %python %S/../../utils/generate_self_loop_cores.py %s > %t.mlir
// RUN: not aie-opt --aie-objectFifo-stateful-transform %t.mlir 2>&1 | FileCheck %s

// The 6x4 grid's packing search runs out of steps: its pools of 3 to 7
// locks leave 8 of its 384 free. (30, 2) alone needs 9 + 8 locks of its 16,
// so the design does not fit, and the error names (30, 2) rather than the
// undecided grid.
// CHECK: error: 'aie.tile' op could not place locks within tile lock capacity
// CHECK-NEXT: %iso = aie.tile(30, 2)

module @aie1_pack_stopped {
  aie.device(xcvc1902) {
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
    %iso = aie.tile(30, 2)
    aie.objectfifo @f0(%t19, {%t19}, 6 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f1(%t8, {%t8}, 5 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f2(%t5, {%t5}, 4 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f3(%t10, {%t10}, 3 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f4(%t14, {%t14}, 7 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f5(%t2, {%t2}, 7 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f6(%t17, {%t17}, 5 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f7(%t22, {%t22}, 7 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f8(%t23, {%t23}, 3 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f9(%t5, {%t5}, 6 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f10(%t23, {%t23}, 6 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f11(%t5, {%t5}, 6 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f12(%t7, {%t7}, 4 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f13(%t3, {%t3}, 3 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f14(%t16, {%t16}, 4 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f15(%t2, {%t2}, 7 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f16(%t23, {%t23}, 6 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f17(%t9, {%t9}, 3 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f18(%t21, {%t21}, 4 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f19(%t23, {%t23}, 4 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f20(%t2, {%t2}, 6 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f21(%t6, {%t6}, 5 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f22(%t8, {%t8}, 6 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f23(%t1, {%t1}, 5 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f24(%t22, {%t22}, 4 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f25(%t13, {%t13}, 3 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f26(%t12, {%t12}, 3 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f27(%t4, {%t4}, 6 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f28(%t7, {%t7}, 3 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f29(%t23, {%t23}, 6 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f30(%t19, {%t19}, 3 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f31(%t3, {%t3}, 3 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f32(%t6, {%t6}, 7 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f33(%t10, {%t10}, 4 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f34(%t2, {%t2}, 3 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f35(%t17, {%t17}, 4 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f36(%t16, {%t16}, 3 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f37(%t18, {%t18}, 3 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f38(%t17, {%t17}, 6 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f39(%t13, {%t13}, 4 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f40(%t12, {%t12}, 3 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f41(%t20, {%t20}, 4 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f42(%t22, {%t22}, 3 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f43(%t4, {%t4}, 7 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f44(%t19, {%t19}, 4 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f45(%t1, {%t1}, 3 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f46(%t17, {%t17}, 5 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f47(%t4, {%t4}, 7 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f48(%t23, {%t23}, 5 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f49(%t1, {%t1}, 7 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f50(%t22, {%t22}, 3 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f51(%t7, {%t7}, 6 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f52(%t19, {%t19}, 4 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f53(%t1, {%t1}, 7 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f54(%t21, {%t21}, 5 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f55(%t18, {%t18}, 7 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f56(%t22, {%t22}, 7 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f57(%t11, {%t11}, 3 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f58(%t18, {%t18}, 3 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f59(%t14, {%t14}, 5 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f60(%t12, {%t12}, 4 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f61(%t18, {%t18}, 4 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f62(%t12, {%t12}, 3 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f63(%t10, {%t10}, 7 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f64(%t13, {%t13}, 3 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f65(%t6, {%t6}, 3 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f66(%t14, {%t14}, 4 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f67(%t10, {%t10}, 5 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f68(%t9, {%t9}, 3 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f69(%t3, {%t3}, 5 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f70(%t1, {%t1}, 7 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f71(%t12, {%t12}, 3 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f72(%t13, {%t13}, 7 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f73(%t11, {%t11}, 6 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f74(%t18, {%t18}, 4 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f75(%t13, {%t13}, 3 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f76(%t3, {%t3}, 4 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f77(%t16, {%t16}, 6 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f78(%t17, {%t17}, 5 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f79(%t4, {%t4}, 5 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @iso_a(%iso, {%iso}, 9 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @iso_b(%iso, {%iso}, 8 : i32) : !aie.objectfifo<memref<16xi32>>
  }
}
