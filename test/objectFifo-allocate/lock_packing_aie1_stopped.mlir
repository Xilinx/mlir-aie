//===- lock_packing_aie1_stopped.mlir --------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: not aie-opt --aie-objectFifo-stateful-transform %s 2>&1 | FileCheck %s

// The 4x4 grid's packing search runs out of steps: its pools of 3 to 7
// locks leave 8 of its 256 free. (30, 2) alone needs 9 + 8 locks of its 16,
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
    %iso = aie.tile(30, 2)
    aie.objectfifo @f0(%t8, {%t8}, 6 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f1(%t11, {%t11}, 7 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f2(%t12, {%t12}, 4 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f3(%t11, {%t11}, 3 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f4(%t8, {%t8}, 6 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f5(%t7, {%t7}, 6 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f6(%t0, {%t0}, 7 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f7(%t4, {%t4}, 7 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f8(%t11, {%t11}, 6 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f9(%t10, {%t10}, 4 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f10(%t1, {%t1}, 4 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f11(%t6, {%t6}, 7 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f12(%t10, {%t10}, 3 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f13(%t2, {%t2}, 6 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f14(%t1, {%t1}, 3 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f15(%t7, {%t7}, 7 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f16(%t13, {%t13}, 3 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f17(%t3, {%t3}, 6 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f18(%t4, {%t4}, 6 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f19(%t10, {%t10}, 7 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f20(%t5, {%t5}, 7 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f21(%t5, {%t5}, 3 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f22(%t2, {%t2}, 7 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f23(%t13, {%t13}, 6 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f24(%t15, {%t15}, 7 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f25(%t12, {%t12}, 6 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f26(%t0, {%t0}, 7 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f27(%t6, {%t6}, 3 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f28(%t11, {%t11}, 5 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f29(%t12, {%t12}, 5 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f30(%t3, {%t3}, 5 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f31(%t7, {%t7}, 5 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f32(%t11, {%t11}, 5 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f33(%t5, {%t5}, 5 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f34(%t12, {%t12}, 3 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f35(%t1, {%t1}, 6 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f36(%t0, {%t0}, 7 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f37(%t13, {%t13}, 4 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f38(%t12, {%t12}, 3 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f39(%t3, {%t3}, 4 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f40(%t7, {%t7}, 7 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f41(%t2, {%t2}, 4 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f42(%t1, {%t1}, 5 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f43(%t8, {%t8}, 6 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f44(%t11, {%t11}, 6 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f45(%t1, {%t1}, 7 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f46(%t14, {%t14}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @iso_a(%iso, {%iso}, 9 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @iso_b(%iso, {%iso}, 8 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.core(%t0) {
      %p6 = aie.objectfifo.acquire @f6(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f6(Produce, 1)
      %c6 = aie.objectfifo.acquire @f6(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f6(Consume, 1)
      %p26 = aie.objectfifo.acquire @f26(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f26(Produce, 1)
      %c26 = aie.objectfifo.acquire @f26(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f26(Consume, 1)
      %p36 = aie.objectfifo.acquire @f36(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f36(Produce, 1)
      %c36 = aie.objectfifo.acquire @f36(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f36(Consume, 1)
      aie.end
    }
    aie.core(%t1) {
      %p10 = aie.objectfifo.acquire @f10(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f10(Produce, 1)
      %c10 = aie.objectfifo.acquire @f10(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f10(Consume, 1)
      %p14 = aie.objectfifo.acquire @f14(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f14(Produce, 1)
      %c14 = aie.objectfifo.acquire @f14(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f14(Consume, 1)
      %p35 = aie.objectfifo.acquire @f35(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f35(Produce, 1)
      %c35 = aie.objectfifo.acquire @f35(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f35(Consume, 1)
      %p42 = aie.objectfifo.acquire @f42(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f42(Produce, 1)
      %c42 = aie.objectfifo.acquire @f42(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f42(Consume, 1)
      %p45 = aie.objectfifo.acquire @f45(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f45(Produce, 1)
      %c45 = aie.objectfifo.acquire @f45(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f45(Consume, 1)
      aie.end
    }
    aie.core(%t2) {
      %p13 = aie.objectfifo.acquire @f13(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f13(Produce, 1)
      %c13 = aie.objectfifo.acquire @f13(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f13(Consume, 1)
      %p22 = aie.objectfifo.acquire @f22(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f22(Produce, 1)
      %c22 = aie.objectfifo.acquire @f22(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f22(Consume, 1)
      %p41 = aie.objectfifo.acquire @f41(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f41(Produce, 1)
      %c41 = aie.objectfifo.acquire @f41(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f41(Consume, 1)
      aie.end
    }
    aie.core(%t3) {
      %p17 = aie.objectfifo.acquire @f17(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f17(Produce, 1)
      %c17 = aie.objectfifo.acquire @f17(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f17(Consume, 1)
      %p30 = aie.objectfifo.acquire @f30(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f30(Produce, 1)
      %c30 = aie.objectfifo.acquire @f30(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f30(Consume, 1)
      %p39 = aie.objectfifo.acquire @f39(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f39(Produce, 1)
      %c39 = aie.objectfifo.acquire @f39(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f39(Consume, 1)
      aie.end
    }
    aie.core(%t4) {
      %p7 = aie.objectfifo.acquire @f7(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f7(Produce, 1)
      %c7 = aie.objectfifo.acquire @f7(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f7(Consume, 1)
      %p18 = aie.objectfifo.acquire @f18(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f18(Produce, 1)
      %c18 = aie.objectfifo.acquire @f18(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f18(Consume, 1)
      aie.end
    }
    aie.core(%t5) {
      %p20 = aie.objectfifo.acquire @f20(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f20(Produce, 1)
      %c20 = aie.objectfifo.acquire @f20(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f20(Consume, 1)
      %p21 = aie.objectfifo.acquire @f21(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f21(Produce, 1)
      %c21 = aie.objectfifo.acquire @f21(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f21(Consume, 1)
      %p33 = aie.objectfifo.acquire @f33(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f33(Produce, 1)
      %c33 = aie.objectfifo.acquire @f33(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f33(Consume, 1)
      aie.end
    }
    aie.core(%t6) {
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
    aie.core(%t7) {
      %p5 = aie.objectfifo.acquire @f5(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f5(Produce, 1)
      %c5 = aie.objectfifo.acquire @f5(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f5(Consume, 1)
      %p15 = aie.objectfifo.acquire @f15(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f15(Produce, 1)
      %c15 = aie.objectfifo.acquire @f15(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f15(Consume, 1)
      %p31 = aie.objectfifo.acquire @f31(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f31(Produce, 1)
      %c31 = aie.objectfifo.acquire @f31(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f31(Consume, 1)
      %p40 = aie.objectfifo.acquire @f40(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f40(Produce, 1)
      %c40 = aie.objectfifo.acquire @f40(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f40(Consume, 1)
      aie.end
    }
    aie.core(%t8) {
      %p0 = aie.objectfifo.acquire @f0(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f0(Produce, 1)
      %c0 = aie.objectfifo.acquire @f0(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f0(Consume, 1)
      %p4 = aie.objectfifo.acquire @f4(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f4(Produce, 1)
      %c4 = aie.objectfifo.acquire @f4(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f4(Consume, 1)
      %p43 = aie.objectfifo.acquire @f43(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f43(Produce, 1)
      %c43 = aie.objectfifo.acquire @f43(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f43(Consume, 1)
      aie.end
    }
    aie.core(%t9) {
      aie.end
    }
    aie.core(%t10) {
      %p9 = aie.objectfifo.acquire @f9(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f9(Produce, 1)
      %c9 = aie.objectfifo.acquire @f9(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f9(Consume, 1)
      %p12 = aie.objectfifo.acquire @f12(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f12(Produce, 1)
      %c12 = aie.objectfifo.acquire @f12(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f12(Consume, 1)
      %p19 = aie.objectfifo.acquire @f19(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f19(Produce, 1)
      %c19 = aie.objectfifo.acquire @f19(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f19(Consume, 1)
      aie.end
    }
    aie.core(%t11) {
      %p1 = aie.objectfifo.acquire @f1(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f1(Produce, 1)
      %c1 = aie.objectfifo.acquire @f1(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f1(Consume, 1)
      %p3 = aie.objectfifo.acquire @f3(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f3(Produce, 1)
      %c3 = aie.objectfifo.acquire @f3(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f3(Consume, 1)
      %p8 = aie.objectfifo.acquire @f8(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f8(Produce, 1)
      %c8 = aie.objectfifo.acquire @f8(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f8(Consume, 1)
      %p28 = aie.objectfifo.acquire @f28(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f28(Produce, 1)
      %c28 = aie.objectfifo.acquire @f28(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f28(Consume, 1)
      %p32 = aie.objectfifo.acquire @f32(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f32(Produce, 1)
      %c32 = aie.objectfifo.acquire @f32(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f32(Consume, 1)
      %p44 = aie.objectfifo.acquire @f44(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f44(Produce, 1)
      %c44 = aie.objectfifo.acquire @f44(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f44(Consume, 1)
      aie.end
    }
    aie.core(%t12) {
      %p2 = aie.objectfifo.acquire @f2(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f2(Produce, 1)
      %c2 = aie.objectfifo.acquire @f2(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f2(Consume, 1)
      %p25 = aie.objectfifo.acquire @f25(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f25(Produce, 1)
      %c25 = aie.objectfifo.acquire @f25(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f25(Consume, 1)
      %p29 = aie.objectfifo.acquire @f29(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f29(Produce, 1)
      %c29 = aie.objectfifo.acquire @f29(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f29(Consume, 1)
      %p34 = aie.objectfifo.acquire @f34(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f34(Produce, 1)
      %c34 = aie.objectfifo.acquire @f34(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f34(Consume, 1)
      %p38 = aie.objectfifo.acquire @f38(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f38(Produce, 1)
      %c38 = aie.objectfifo.acquire @f38(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f38(Consume, 1)
      aie.end
    }
    aie.core(%t13) {
      %p16 = aie.objectfifo.acquire @f16(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f16(Produce, 1)
      %c16 = aie.objectfifo.acquire @f16(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f16(Consume, 1)
      %p23 = aie.objectfifo.acquire @f23(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f23(Produce, 1)
      %c23 = aie.objectfifo.acquire @f23(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f23(Consume, 1)
      %p37 = aie.objectfifo.acquire @f37(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f37(Produce, 1)
      %c37 = aie.objectfifo.acquire @f37(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f37(Consume, 1)
      aie.end
    }
    aie.core(%t14) {
      %p46 = aie.objectfifo.acquire @f46(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f46(Produce, 1)
      %c46 = aie.objectfifo.acquire @f46(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f46(Consume, 1)
      aie.end
    }
    aie.core(%t15) {
      %p24 = aie.objectfifo.acquire @f24(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f24(Produce, 1)
      %c24 = aie.objectfifo.acquire @f24(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f24(Consume, 1)
      aie.end
    }
    aie.core(%iso) {
      %iso_ap = aie.objectfifo.acquire @iso_a(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @iso_a(Produce, 1)
      %iso_ac = aie.objectfifo.acquire @iso_a(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @iso_a(Consume, 1)
      %iso_bp = aie.objectfifo.acquire @iso_b(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @iso_b(Produce, 1)
      %iso_bc = aie.objectfifo.acquire @iso_b(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @iso_b(Consume, 1)
      aie.end
    }
  }
}
