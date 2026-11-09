//===- lock_packing_aie1_stopped.mlir --------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: not aie-opt --aie-objectFifo-stateful-transform %s 2>&1 | FileCheck %s

// The 3x3 grid's packing search runs out of steps: each tile has 15 locks
// free and every pool there takes an even count but one of 3. (30, 2) alone
// needs 9 + 8 locks of its 16, so the design does not fit, and the error
// names (30, 2) rather than the undecided grid.
// CHECK: error: 'aie.tile' op could not place locks within tile lock capacity
// CHECK-NEXT: %iso = aie.tile(30, 2)

module @aie1_pack_stopped {
  aie.device(xcvc1902) {
    %t0 = aie.tile(5, 1)
    %t1 = aie.tile(5, 2)
    %t2 = aie.tile(5, 3)
    %t3 = aie.tile(6, 1)
    %t4 = aie.tile(6, 2)
    %t5 = aie.tile(6, 3)
    %t6 = aie.tile(7, 1)
    %t7 = aie.tile(7, 2)
    %t8 = aie.tile(7, 3)
    %iso = aie.tile(30, 2)
    %l0 = aie.lock(%t0, 15)
    %l1 = aie.lock(%t1, 15)
    %l2 = aie.lock(%t2, 15)
    %l3 = aie.lock(%t3, 15)
    %l4 = aie.lock(%t4, 15)
    %l5 = aie.lock(%t5, 15)
    %l6 = aie.lock(%t6, 15)
    %l7 = aie.lock(%t7, 15)
    %l8 = aie.lock(%t8, 15)
    aie.objectfifo @f0(%t8, {%t8}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f1(%t4, {%t4}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f2(%t2, {%t2}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f3(%t1, {%t1}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f4(%t2, {%t2}, 4 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f5(%t0, {%t0}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f6(%t4, {%t4}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f7(%t1, {%t1}, 4 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f8(%t0, {%t0}, 4 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f9(%t7, {%t7}, 4 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f10(%t6, {%t6}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f11(%t0, {%t0}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f12(%t7, {%t7}, 4 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f13(%t1, {%t1}, 4 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f14(%t5, {%t5}, 4 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f15(%t8, {%t8}, 4 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f16(%t8, {%t8}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f17(%t5, {%t5}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f18(%t0, {%t0}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f19(%t3, {%t3}, 4 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f20(%t3, {%t3}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f21(%t6, {%t6}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f22(%t8, {%t8}, 3 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f23(%t5, {%t5}, 4 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f24(%t4, {%t4}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f25(%t4, {%t4}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f26(%t3, {%t3}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f27(%t2, {%t2}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f28(%t1, {%t1}, 4 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f29(%t4, {%t4}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f30(%t2, {%t2}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f31(%t7, {%t7}, 4 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f32(%t3, {%t3}, 4 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f33(%t6, {%t6}, 4 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f34(%t0, {%t0}, 4 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f35(%t5, {%t5}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f36(%t8, {%t8}, 4 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f37(%t6, {%t6}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f38(%t7, {%t7}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f39(%t3, {%t3}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f40(%t4, {%t4}, 4 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f41(%t2, {%t2}, 4 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f42(%t8, {%t8}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f43(%t6, {%t6}, 4 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f44(%t5, {%t5}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @iso_a(%iso, {%iso}, 9 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @iso_b(%iso, {%iso}, 8 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.core(%t0) {
      %p5 = aie.objectfifo.acquire @f5(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f5(Produce, 1)
      %c5 = aie.objectfifo.acquire @f5(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f5(Consume, 1)
      %p8 = aie.objectfifo.acquire @f8(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f8(Produce, 1)
      %c8 = aie.objectfifo.acquire @f8(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f8(Consume, 1)
      %p11 = aie.objectfifo.acquire @f11(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f11(Produce, 1)
      %c11 = aie.objectfifo.acquire @f11(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f11(Consume, 1)
      %p18 = aie.objectfifo.acquire @f18(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f18(Produce, 1)
      %c18 = aie.objectfifo.acquire @f18(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f18(Consume, 1)
      %p34 = aie.objectfifo.acquire @f34(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f34(Produce, 1)
      %c34 = aie.objectfifo.acquire @f34(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f34(Consume, 1)
      aie.end
    }
    aie.core(%t1) {
      %p3 = aie.objectfifo.acquire @f3(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f3(Produce, 1)
      %c3 = aie.objectfifo.acquire @f3(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f3(Consume, 1)
      %p7 = aie.objectfifo.acquire @f7(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f7(Produce, 1)
      %c7 = aie.objectfifo.acquire @f7(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f7(Consume, 1)
      %p13 = aie.objectfifo.acquire @f13(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f13(Produce, 1)
      %c13 = aie.objectfifo.acquire @f13(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f13(Consume, 1)
      %p28 = aie.objectfifo.acquire @f28(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f28(Produce, 1)
      %c28 = aie.objectfifo.acquire @f28(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f28(Consume, 1)
      aie.end
    }
    aie.core(%t2) {
      %p2 = aie.objectfifo.acquire @f2(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f2(Produce, 1)
      %c2 = aie.objectfifo.acquire @f2(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f2(Consume, 1)
      %p4 = aie.objectfifo.acquire @f4(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f4(Produce, 1)
      %c4 = aie.objectfifo.acquire @f4(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f4(Consume, 1)
      %p27 = aie.objectfifo.acquire @f27(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f27(Produce, 1)
      %c27 = aie.objectfifo.acquire @f27(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f27(Consume, 1)
      %p30 = aie.objectfifo.acquire @f30(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f30(Produce, 1)
      %c30 = aie.objectfifo.acquire @f30(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f30(Consume, 1)
      %p41 = aie.objectfifo.acquire @f41(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f41(Produce, 1)
      %c41 = aie.objectfifo.acquire @f41(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f41(Consume, 1)
      aie.end
    }
    aie.core(%t3) {
      %p19 = aie.objectfifo.acquire @f19(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f19(Produce, 1)
      %c19 = aie.objectfifo.acquire @f19(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f19(Consume, 1)
      %p20 = aie.objectfifo.acquire @f20(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f20(Produce, 1)
      %c20 = aie.objectfifo.acquire @f20(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f20(Consume, 1)
      %p26 = aie.objectfifo.acquire @f26(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f26(Produce, 1)
      %c26 = aie.objectfifo.acquire @f26(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f26(Consume, 1)
      %p32 = aie.objectfifo.acquire @f32(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f32(Produce, 1)
      %c32 = aie.objectfifo.acquire @f32(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f32(Consume, 1)
      %p39 = aie.objectfifo.acquire @f39(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f39(Produce, 1)
      %c39 = aie.objectfifo.acquire @f39(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f39(Consume, 1)
      aie.end
    }
    aie.core(%t4) {
      %p1 = aie.objectfifo.acquire @f1(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f1(Produce, 1)
      %c1 = aie.objectfifo.acquire @f1(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f1(Consume, 1)
      %p6 = aie.objectfifo.acquire @f6(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f6(Produce, 1)
      %c6 = aie.objectfifo.acquire @f6(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f6(Consume, 1)
      %p24 = aie.objectfifo.acquire @f24(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f24(Produce, 1)
      %c24 = aie.objectfifo.acquire @f24(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f24(Consume, 1)
      %p25 = aie.objectfifo.acquire @f25(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f25(Produce, 1)
      %c25 = aie.objectfifo.acquire @f25(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f25(Consume, 1)
      %p29 = aie.objectfifo.acquire @f29(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f29(Produce, 1)
      %c29 = aie.objectfifo.acquire @f29(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f29(Consume, 1)
      %p40 = aie.objectfifo.acquire @f40(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f40(Produce, 1)
      %c40 = aie.objectfifo.acquire @f40(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f40(Consume, 1)
      aie.end
    }
    aie.core(%t5) {
      %p14 = aie.objectfifo.acquire @f14(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f14(Produce, 1)
      %c14 = aie.objectfifo.acquire @f14(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f14(Consume, 1)
      %p17 = aie.objectfifo.acquire @f17(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f17(Produce, 1)
      %c17 = aie.objectfifo.acquire @f17(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f17(Consume, 1)
      %p23 = aie.objectfifo.acquire @f23(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f23(Produce, 1)
      %c23 = aie.objectfifo.acquire @f23(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f23(Consume, 1)
      %p35 = aie.objectfifo.acquire @f35(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f35(Produce, 1)
      %c35 = aie.objectfifo.acquire @f35(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f35(Consume, 1)
      %p44 = aie.objectfifo.acquire @f44(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f44(Produce, 1)
      %c44 = aie.objectfifo.acquire @f44(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f44(Consume, 1)
      aie.end
    }
    aie.core(%t6) {
      %p10 = aie.objectfifo.acquire @f10(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f10(Produce, 1)
      %c10 = aie.objectfifo.acquire @f10(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f10(Consume, 1)
      %p21 = aie.objectfifo.acquire @f21(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f21(Produce, 1)
      %c21 = aie.objectfifo.acquire @f21(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f21(Consume, 1)
      %p33 = aie.objectfifo.acquire @f33(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f33(Produce, 1)
      %c33 = aie.objectfifo.acquire @f33(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f33(Consume, 1)
      %p37 = aie.objectfifo.acquire @f37(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f37(Produce, 1)
      %c37 = aie.objectfifo.acquire @f37(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f37(Consume, 1)
      %p43 = aie.objectfifo.acquire @f43(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f43(Produce, 1)
      %c43 = aie.objectfifo.acquire @f43(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f43(Consume, 1)
      aie.end
    }
    aie.core(%t7) {
      %p9 = aie.objectfifo.acquire @f9(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f9(Produce, 1)
      %c9 = aie.objectfifo.acquire @f9(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f9(Consume, 1)
      %p12 = aie.objectfifo.acquire @f12(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f12(Produce, 1)
      %c12 = aie.objectfifo.acquire @f12(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f12(Consume, 1)
      %p31 = aie.objectfifo.acquire @f31(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f31(Produce, 1)
      %c31 = aie.objectfifo.acquire @f31(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f31(Consume, 1)
      %p38 = aie.objectfifo.acquire @f38(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f38(Produce, 1)
      %c38 = aie.objectfifo.acquire @f38(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f38(Consume, 1)
      aie.end
    }
    aie.core(%t8) {
      %p0 = aie.objectfifo.acquire @f0(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f0(Produce, 1)
      %c0 = aie.objectfifo.acquire @f0(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f0(Consume, 1)
      %p15 = aie.objectfifo.acquire @f15(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f15(Produce, 1)
      %c15 = aie.objectfifo.acquire @f15(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f15(Consume, 1)
      %p16 = aie.objectfifo.acquire @f16(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f16(Produce, 1)
      %c16 = aie.objectfifo.acquire @f16(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f16(Consume, 1)
      %p22 = aie.objectfifo.acquire @f22(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f22(Produce, 1)
      %c22 = aie.objectfifo.acquire @f22(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f22(Consume, 1)
      %p36 = aie.objectfifo.acquire @f36(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f36(Produce, 1)
      %c36 = aie.objectfifo.acquire @f36(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f36(Consume, 1)
      %p42 = aie.objectfifo.acquire @f42(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f42(Produce, 1)
      %c42 = aie.objectfifo.acquire @f42(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f42(Consume, 1)
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
