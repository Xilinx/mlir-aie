//===- test_sa_lock_order.mlir ---------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: %python %S/../../../utils/component_checks/sa_lock_check.py --aie-opt aie-opt --seed 1 %s | FileCheck %s

// The SA lists the lock groups in fifo order, allocation in its structural
// order. The 6x4 grid's pools of 1 to 7 locks take 376 of its 384, and how
// many steps the exact packing takes depends on the order it searches in.
// The packing searches in an order of its own, so the SA and allocation
// reach the same verdict.
// CHECK: agree=1 mismatch=0 skipped=0 agreeing_lock_failures=0 agreeing_lock_stops=1

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
    aie.core(%t0) {
      %p58 = aie.objectfifo.acquire @f58(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f58(Produce, 1)
      %c58 = aie.objectfifo.acquire @f58(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f58(Consume, 1)
      aie.end
    }
    aie.core(%t1) {
      %p28 = aie.objectfifo.acquire @f28(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f28(Produce, 1)
      %c28 = aie.objectfifo.acquire @f28(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f28(Consume, 1)
      %p50 = aie.objectfifo.acquire @f50(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f50(Produce, 1)
      %c50 = aie.objectfifo.acquire @f50(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f50(Consume, 1)
      %p67 = aie.objectfifo.acquire @f67(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f67(Produce, 1)
      %c67 = aie.objectfifo.acquire @f67(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f67(Consume, 1)
      aie.end
    }
    aie.core(%t2) {
      %p3 = aie.objectfifo.acquire @f3(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f3(Produce, 1)
      %c3 = aie.objectfifo.acquire @f3(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f3(Consume, 1)
      %p11 = aie.objectfifo.acquire @f11(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f11(Produce, 1)
      %c11 = aie.objectfifo.acquire @f11(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f11(Consume, 1)
      %p27 = aie.objectfifo.acquire @f27(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f27(Produce, 1)
      %c27 = aie.objectfifo.acquire @f27(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f27(Consume, 1)
      aie.end
    }
    aie.core(%t3) {
      %p18 = aie.objectfifo.acquire @f18(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f18(Produce, 1)
      %c18 = aie.objectfifo.acquire @f18(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f18(Consume, 1)
      %p36 = aie.objectfifo.acquire @f36(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f36(Produce, 1)
      %c36 = aie.objectfifo.acquire @f36(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f36(Consume, 1)
      %p54 = aie.objectfifo.acquire @f54(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f54(Produce, 1)
      %c54 = aie.objectfifo.acquire @f54(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f54(Consume, 1)
      aie.end
    }
    aie.core(%t4) {
      %p1 = aie.objectfifo.acquire @f1(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f1(Produce, 1)
      %c1 = aie.objectfifo.acquire @f1(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f1(Consume, 1)
      %p26 = aie.objectfifo.acquire @f26(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f26(Produce, 1)
      %c26 = aie.objectfifo.acquire @f26(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f26(Consume, 1)
      %p30 = aie.objectfifo.acquire @f30(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f30(Produce, 1)
      %c30 = aie.objectfifo.acquire @f30(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f30(Consume, 1)
      %p42 = aie.objectfifo.acquire @f42(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f42(Produce, 1)
      %c42 = aie.objectfifo.acquire @f42(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f42(Consume, 1)
      %p75 = aie.objectfifo.acquire @f75(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f75(Produce, 1)
      %c75 = aie.objectfifo.acquire @f75(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f75(Consume, 1)
      aie.end
    }
    aie.core(%t5) {
      %p35 = aie.objectfifo.acquire @f35(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f35(Produce, 1)
      %c35 = aie.objectfifo.acquire @f35(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f35(Consume, 1)
      %p53 = aie.objectfifo.acquire @f53(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f53(Produce, 1)
      %c53 = aie.objectfifo.acquire @f53(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f53(Consume, 1)
      aie.end
    }
    aie.core(%t6) {
      %p5 = aie.objectfifo.acquire @f5(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f5(Produce, 1)
      %c5 = aie.objectfifo.acquire @f5(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f5(Consume, 1)
      %p10 = aie.objectfifo.acquire @f10(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f10(Produce, 1)
      %c10 = aie.objectfifo.acquire @f10(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f10(Consume, 1)
      %p25 = aie.objectfifo.acquire @f25(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f25(Produce, 1)
      %c25 = aie.objectfifo.acquire @f25(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f25(Consume, 1)
      %p41 = aie.objectfifo.acquire @f41(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f41(Produce, 1)
      %c41 = aie.objectfifo.acquire @f41(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f41(Consume, 1)
      %p56 = aie.objectfifo.acquire @f56(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f56(Produce, 1)
      %c56 = aie.objectfifo.acquire @f56(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f56(Consume, 1)
      %p74 = aie.objectfifo.acquire @f74(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f74(Produce, 1)
      %c74 = aie.objectfifo.acquire @f74(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f74(Consume, 1)
      aie.end
    }
    aie.core(%t7) {
      %p4 = aie.objectfifo.acquire @f4(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f4(Produce, 1)
      %c4 = aie.objectfifo.acquire @f4(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f4(Consume, 1)
      %p12 = aie.objectfifo.acquire @f12(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f12(Produce, 1)
      %c12 = aie.objectfifo.acquire @f12(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f12(Consume, 1)
      %p46 = aie.objectfifo.acquire @f46(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f46(Produce, 1)
      %c46 = aie.objectfifo.acquire @f46(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f46(Consume, 1)
      %p47 = aie.objectfifo.acquire @f47(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f47(Produce, 1)
      %c47 = aie.objectfifo.acquire @f47(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f47(Consume, 1)
      %p49 = aie.objectfifo.acquire @f49(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f49(Produce, 1)
      %c49 = aie.objectfifo.acquire @f49(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f49(Consume, 1)
      %p61 = aie.objectfifo.acquire @f61(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f61(Produce, 1)
      %c61 = aie.objectfifo.acquire @f61(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f61(Consume, 1)
      aie.end
    }
    aie.core(%t8) {
      %p51 = aie.objectfifo.acquire @f51(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f51(Produce, 1)
      %c51 = aie.objectfifo.acquire @f51(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f51(Consume, 1)
      aie.end
    }
    aie.core(%t9) {
      %p48 = aie.objectfifo.acquire @f48(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f48(Produce, 1)
      %c48 = aie.objectfifo.acquire @f48(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f48(Consume, 1)
      %p52 = aie.objectfifo.acquire @f52(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f52(Produce, 1)
      %c52 = aie.objectfifo.acquire @f52(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f52(Consume, 1)
      %p57 = aie.objectfifo.acquire @f57(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f57(Produce, 1)
      %c57 = aie.objectfifo.acquire @f57(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f57(Consume, 1)
      %p66 = aie.objectfifo.acquire @f66(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f66(Produce, 1)
      %c66 = aie.objectfifo.acquire @f66(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f66(Consume, 1)
      aie.end
    }
    aie.core(%t10) {
      %p22 = aie.objectfifo.acquire @f22(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f22(Produce, 1)
      %c22 = aie.objectfifo.acquire @f22(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f22(Consume, 1)
      %p59 = aie.objectfifo.acquire @f59(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f59(Produce, 1)
      %c59 = aie.objectfifo.acquire @f59(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f59(Consume, 1)
      aie.end
    }
    aie.core(%t11) {
      %p0 = aie.objectfifo.acquire @f0(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f0(Produce, 1)
      %c0 = aie.objectfifo.acquire @f0(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f0(Consume, 1)
      %p70 = aie.objectfifo.acquire @f70(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f70(Produce, 1)
      %c70 = aie.objectfifo.acquire @f70(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f70(Consume, 1)
      aie.end
    }
    aie.core(%t12) {
      %p19 = aie.objectfifo.acquire @f19(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f19(Produce, 1)
      %c19 = aie.objectfifo.acquire @f19(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f19(Consume, 1)
      %p20 = aie.objectfifo.acquire @f20(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f20(Produce, 1)
      %c20 = aie.objectfifo.acquire @f20(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f20(Consume, 1)
      aie.end
    }
    aie.core(%t13) {
      %p34 = aie.objectfifo.acquire @f34(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f34(Produce, 1)
      %c34 = aie.objectfifo.acquire @f34(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f34(Consume, 1)
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
    aie.core(%t14) {
      %p7 = aie.objectfifo.acquire @f7(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f7(Produce, 1)
      %c7 = aie.objectfifo.acquire @f7(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f7(Consume, 1)
      %p8 = aie.objectfifo.acquire @f8(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f8(Produce, 1)
      %c8 = aie.objectfifo.acquire @f8(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f8(Consume, 1)
      %p31 = aie.objectfifo.acquire @f31(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f31(Produce, 1)
      %c31 = aie.objectfifo.acquire @f31(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f31(Consume, 1)
      %p32 = aie.objectfifo.acquire @f32(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f32(Produce, 1)
      %c32 = aie.objectfifo.acquire @f32(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f32(Consume, 1)
      %p63 = aie.objectfifo.acquire @f63(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f63(Produce, 1)
      %c63 = aie.objectfifo.acquire @f63(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f63(Consume, 1)
      aie.end
    }
    aie.core(%t15) {
      %p9 = aie.objectfifo.acquire @f9(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f9(Produce, 1)
      %c9 = aie.objectfifo.acquire @f9(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f9(Consume, 1)
      %p15 = aie.objectfifo.acquire @f15(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f15(Produce, 1)
      %c15 = aie.objectfifo.acquire @f15(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f15(Consume, 1)
      %p24 = aie.objectfifo.acquire @f24(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f24(Produce, 1)
      %c24 = aie.objectfifo.acquire @f24(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f24(Consume, 1)
      aie.end
    }
    aie.core(%t16) {
      %p14 = aie.objectfifo.acquire @f14(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f14(Produce, 1)
      %c14 = aie.objectfifo.acquire @f14(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f14(Consume, 1)
      %p40 = aie.objectfifo.acquire @f40(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f40(Produce, 1)
      %c40 = aie.objectfifo.acquire @f40(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f40(Consume, 1)
      %p43 = aie.objectfifo.acquire @f43(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f43(Produce, 1)
      %c43 = aie.objectfifo.acquire @f43(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f43(Consume, 1)
      %p44 = aie.objectfifo.acquire @f44(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f44(Produce, 1)
      %c44 = aie.objectfifo.acquire @f44(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f44(Consume, 1)
      %p45 = aie.objectfifo.acquire @f45(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f45(Produce, 1)
      %c45 = aie.objectfifo.acquire @f45(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f45(Consume, 1)
      %p69 = aie.objectfifo.acquire @f69(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f69(Produce, 1)
      %c69 = aie.objectfifo.acquire @f69(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f69(Consume, 1)
      %p71 = aie.objectfifo.acquire @f71(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f71(Produce, 1)
      %c71 = aie.objectfifo.acquire @f71(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f71(Consume, 1)
      aie.end
    }
    aie.core(%t17) {
      %p33 = aie.objectfifo.acquire @f33(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f33(Produce, 1)
      %c33 = aie.objectfifo.acquire @f33(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f33(Consume, 1)
      aie.end
    }
    aie.core(%t18) {
      %p60 = aie.objectfifo.acquire @f60(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f60(Produce, 1)
      %c60 = aie.objectfifo.acquire @f60(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f60(Consume, 1)
      %p65 = aie.objectfifo.acquire @f65(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f65(Produce, 1)
      %c65 = aie.objectfifo.acquire @f65(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f65(Consume, 1)
      aie.end
    }
    aie.core(%t19) {
      %p73 = aie.objectfifo.acquire @f73(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f73(Produce, 1)
      %c73 = aie.objectfifo.acquire @f73(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f73(Consume, 1)
      aie.end
    }
    aie.core(%t20) {
      %p6 = aie.objectfifo.acquire @f6(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f6(Produce, 1)
      %c6 = aie.objectfifo.acquire @f6(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f6(Consume, 1)
      %p39 = aie.objectfifo.acquire @f39(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f39(Produce, 1)
      %c39 = aie.objectfifo.acquire @f39(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f39(Consume, 1)
      aie.end
    }
    aie.core(%t21) {
      %p17 = aie.objectfifo.acquire @f17(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f17(Produce, 1)
      %c17 = aie.objectfifo.acquire @f17(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f17(Consume, 1)
      %p21 = aie.objectfifo.acquire @f21(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f21(Produce, 1)
      %c21 = aie.objectfifo.acquire @f21(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f21(Consume, 1)
      %p55 = aie.objectfifo.acquire @f55(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f55(Produce, 1)
      %c55 = aie.objectfifo.acquire @f55(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f55(Consume, 1)
      %p68 = aie.objectfifo.acquire @f68(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f68(Produce, 1)
      %c68 = aie.objectfifo.acquire @f68(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f68(Consume, 1)
      aie.end
    }
    aie.core(%t22) {
      %p2 = aie.objectfifo.acquire @f2(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f2(Produce, 1)
      %c2 = aie.objectfifo.acquire @f2(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f2(Consume, 1)
      %p13 = aie.objectfifo.acquire @f13(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f13(Produce, 1)
      %c13 = aie.objectfifo.acquire @f13(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f13(Consume, 1)
      %p23 = aie.objectfifo.acquire @f23(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f23(Produce, 1)
      %c23 = aie.objectfifo.acquire @f23(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f23(Consume, 1)
      %p29 = aie.objectfifo.acquire @f29(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f29(Produce, 1)
      %c29 = aie.objectfifo.acquire @f29(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f29(Consume, 1)
      %p37 = aie.objectfifo.acquire @f37(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f37(Produce, 1)
      %c37 = aie.objectfifo.acquire @f37(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f37(Consume, 1)
      %p38 = aie.objectfifo.acquire @f38(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f38(Produce, 1)
      %c38 = aie.objectfifo.acquire @f38(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f38(Consume, 1)
      %p72 = aie.objectfifo.acquire @f72(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f72(Produce, 1)
      %c72 = aie.objectfifo.acquire @f72(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f72(Consume, 1)
      aie.end
    }
    aie.core(%t23) {
      %p16 = aie.objectfifo.acquire @f16(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f16(Produce, 1)
      %c16 = aie.objectfifo.acquire @f16(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f16(Consume, 1)
      aie.end
    }
  }
}
