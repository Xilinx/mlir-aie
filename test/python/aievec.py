# Copyright (C) 2023 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %python %s | FileCheck %s

import inspect

from aie.extras import types as T
from aie.extras.dialects import arith
from aie.extras.runtime.passes import Pipeline as p, run_pipeline

from aie.helpers.dialects.func import func
from aie.iron.controlflow import range_

from aie.dialects import affine, aievec, tosa, vector

# noinspection PyUnresolvedReferences
from aie.ir import AffineMap, AffineDimExpr
from util import construct_and_print_module


# CHECK-LABEL: TEST: test_emit
# CHECK: module {
# CHECK:   func.func @demo_fun1() -> i32 {
# CHECK:     %c1_i32 = arith.constant 1 : i32
# CHECK:     return %c1_i32 : i32
# CHECK:   }
# CHECK: }
@construct_and_print_module
def test_emit():
    @func
    def demo_fun1():
        one = arith.constant(1)
        return one

    assert hasattr(demo_fun1, "emit")
    assert inspect.ismethod(demo_fun1.emit)
    demo_fun1.emit()


@construct_and_print_module
def test_aievec():
    @func
    def mul_elem(
        A: T.memref(2048, T.i16()),
        B: T.memref(2048, T.i16()),
        C: T.memref(2048, T.i16()),
    ):
        for i in range_(0, 2048, 32):
            v0 = aievec.upd(T.vector(32, T.i16()), A, [i])
            v1 = aievec.upd(T.vector(32, T.i16()), B, [i])
            v2 = aievec.mul_elem(
                T.vector(32, T.i32()),
                v0,
                v1,
            )
            v3 = aievec.srs(T.vector(32, T.i16()), v2, arith.constant(0))
            vector.transfer_write(
                None,
                v3,
                C,
                [i],
                AffineMap.get_identity(1),
                in_bounds=[True],
            )

    # CHECK-LABEL:   func.func @mul_elem(
    # CHECK-SAME:                       %[[VAL_0:.*]]: memref<2048xi16>, %[[VAL_1:.*]]: memref<2048xi16>, %[[VAL_2:.*]]: memref<2048xi16>) {
    # CHECK:           %[[VAL_3:.*]] = arith.constant 0 : index
    # CHECK:           %[[VAL_4:.*]] = arith.constant 2048 : index
    # CHECK:           %[[VAL_5:.*]] = arith.constant 32 : index
    # CHECK:           scf.for %[[VAL_6:.*]] = %[[VAL_3]] to %[[VAL_4]] step %[[VAL_5]] {
    # CHECK:             %[[VAL_7:.*]] = aievec.upd %[[VAL_0]]{{\[}}%[[VAL_6]]] {index = 0 : i8, offset = 0 : i32} : memref<2048xi16>, vector<32xi16>
    # CHECK:             %[[VAL_8:.*]] = aievec.upd %[[VAL_1]]{{\[}}%[[VAL_6]]] {index = 0 : i8, offset = 0 : i32} : memref<2048xi16>, vector<32xi16>
    # CHECK:             %[[VAL_9:.*]] = aievec.mul_elem %[[VAL_7]], %[[VAL_8]] : vector<32xi16>, vector<32xi16>, vector<32xi32>
    # CHECK:             %[[VAL_10:.*]] = arith.constant 0 : i32
    # CHECK:             %[[VAL_11:.*]] = aievec.srs %[[VAL_9]], %[[VAL_10]] : vector<32xi32>, i32, vector<32xi16>
    # CHECK:             vector.transfer_write %[[VAL_11]], %[[VAL_2]]{{\[}}%[[VAL_6]]] {in_bounds = [true]} : vector<32xi16>, memref<2048xi16>
    # CHECK:           }
    # CHECK:           return
    # CHECK:         }
    mul_elem.emit()


# CHECK-LABEL: TEST: test_tiled_nonsquare_tile_matrix_mult_vectorized
@construct_and_print_module
def test_tiled_nonsquare_tile_matrix_mult_vectorized(module):

    vec16int32 = T.vector(16, T.i32())
    vec16int64 = T.vector(16, T.i64())
    perm_map = AffineMap.get(2, 0, [AffineDimExpr.get(1)])

    @func(emit=True)
    def matmul_i32_i32(
        A: T.memref(16, 32, T.i32()),
        B: T.memref(32, 16, T.i32()),
        C: T.memref(16, 16, T.i32()),
    ):
        c0 = arith.constant(0, index=True)
        for j in range_(0, 16):
            c_vec = aievec.upd(vec16int32, C, [j, c0])
            accum = aievec.ups(vec16int64, c_vec)
            for k in range_(0, 32, 8):
                a_vec = aievec.upd(vec16int32, A, [j, k])
                for i in range(0, 8):
                    broad_a = aievec.broadcast(vec16int32, a_vec, idx=i)
                    b_vec = aievec.upd(vec16int32, B, [k + i, c0])
                    accum = aievec.mac_elem(vec16int64, broad_a, b_vec, accum)

                shift_round_sat = aievec.srs(vec16int32, accum, arith.constant(0))
                vector.transfer_write(
                    None,
                    shift_round_sat,
                    C,
                    [j, c0],
                    permutation_map=perm_map,
                    in_bounds=[True],
                )

    # CHECK-LABEL:   func.func @matmul_i32_i32(
    # CHECK-SAME:                              %[[VAL_0:.*]]: memref<16x32xi32>, %[[VAL_1:.*]]: memref<32x16xi32>, %[[VAL_2:.*]]: memref<16x16xi32>) {
    # CHECK:           %[[VAL_3:.*]] = arith.constant 0 : index
    # CHECK:           %[[VAL_4:.*]] = arith.constant 0 : index
    # CHECK:           %[[VAL_5:.*]] = arith.constant 16 : index
    # CHECK:           %[[VAL_6:.*]] = arith.constant 1 : index
    # CHECK:           scf.for %[[VAL_7:.*]] = %[[VAL_4]] to %[[VAL_5]] step %[[VAL_6]] {
    # CHECK:             %[[VAL_8:.*]] = aievec.upd %[[VAL_2]]{{\[}}%[[VAL_7]], %[[VAL_3]]] {index = 0 : i8, offset = 0 : i32} : memref<16x16xi32>, vector<16xi32>
    # CHECK:             %[[VAL_9:.*]] = aievec.ups %[[VAL_8]] {shift = 0 : i8} : vector<16xi32>, vector<16xi64>
    # CHECK:             %[[VAL_10:.*]] = arith.constant 0 : index
    # CHECK:             %[[VAL_11:.*]] = arith.constant 32 : index
    # CHECK:             %[[VAL_12:.*]] = arith.constant 8 : index
    # CHECK:             scf.for %[[VAL_13:.*]] = %[[VAL_10]] to %[[VAL_11]] step %[[VAL_12]] {
    # CHECK:               %[[VAL_14:.*]] = aievec.upd %[[VAL_0]]{{\[}}%[[VAL_7]], %[[VAL_13]]] {index = 0 : i8, offset = 0 : i32} : memref<16x32xi32>, vector<16xi32>
    # CHECK:               %[[VAL_15:.*]] = aievec.broadcast %[[VAL_14]] {idx = 0 : i8} : vector<16xi32>, vector<16xi32>
    # CHECK:               %[[VAL_16:.*]] = arith.constant 0 : index
    # CHECK:               %[[VAL_17:.*]] = arith.addi %[[VAL_13]], %[[VAL_16]] : index
    # CHECK:               %[[VAL_18:.*]] = aievec.upd %[[VAL_1]]{{\[}}%[[VAL_17]], %[[VAL_3]]] {index = 0 : i8, offset = 0 : i32} : memref<32x16xi32>, vector<16xi32>
    # CHECK:               %[[VAL_19:.*]] = aievec.mac_elem %[[VAL_15]], %[[VAL_18]], %[[VAL_9]] : vector<16xi32>, vector<16xi32>, vector<16xi64>
    # CHECK:               %[[VAL_20:.*]] = aievec.broadcast %[[VAL_14]] {idx = 1 : i8} : vector<16xi32>, vector<16xi32>
    # CHECK:               %[[VAL_21:.*]] = arith.constant 1 : index
    # CHECK:               %[[VAL_22:.*]] = arith.addi %[[VAL_13]], %[[VAL_21]] : index
    # CHECK:               %[[VAL_23:.*]] = aievec.upd %[[VAL_1]]{{\[}}%[[VAL_22]], %[[VAL_3]]] {index = 0 : i8, offset = 0 : i32} : memref<32x16xi32>, vector<16xi32>
    # CHECK:               %[[VAL_24:.*]] = aievec.mac_elem %[[VAL_20]], %[[VAL_23]], %[[VAL_19]] : vector<16xi32>, vector<16xi32>, vector<16xi64>
    # CHECK:               %[[VAL_25:.*]] = aievec.broadcast %[[VAL_14]] {idx = 2 : i8} : vector<16xi32>, vector<16xi32>
    # CHECK:               %[[VAL_26:.*]] = arith.constant 2 : index
    # CHECK:               %[[VAL_27:.*]] = arith.addi %[[VAL_13]], %[[VAL_26]] : index
    # CHECK:               %[[VAL_28:.*]] = aievec.upd %[[VAL_1]]{{\[}}%[[VAL_27]], %[[VAL_3]]] {index = 0 : i8, offset = 0 : i32} : memref<32x16xi32>, vector<16xi32>
    # CHECK:               %[[VAL_29:.*]] = aievec.mac_elem %[[VAL_25]], %[[VAL_28]], %[[VAL_24]] : vector<16xi32>, vector<16xi32>, vector<16xi64>
    # CHECK:               %[[VAL_30:.*]] = aievec.broadcast %[[VAL_14]] {idx = 3 : i8} : vector<16xi32>, vector<16xi32>
    # CHECK:               %[[VAL_31:.*]] = arith.constant 3 : index
    # CHECK:               %[[VAL_32:.*]] = arith.addi %[[VAL_13]], %[[VAL_31]] : index
    # CHECK:               %[[VAL_33:.*]] = aievec.upd %[[VAL_1]]{{\[}}%[[VAL_32]], %[[VAL_3]]] {index = 0 : i8, offset = 0 : i32} : memref<32x16xi32>, vector<16xi32>
    # CHECK:               %[[VAL_34:.*]] = aievec.mac_elem %[[VAL_30]], %[[VAL_33]], %[[VAL_29]] : vector<16xi32>, vector<16xi32>, vector<16xi64>
    # CHECK:               %[[VAL_35:.*]] = aievec.broadcast %[[VAL_14]] {idx = 4 : i8} : vector<16xi32>, vector<16xi32>
    # CHECK:               %[[VAL_36:.*]] = arith.constant 4 : index
    # CHECK:               %[[VAL_37:.*]] = arith.addi %[[VAL_13]], %[[VAL_36]] : index
    # CHECK:               %[[VAL_38:.*]] = aievec.upd %[[VAL_1]]{{\[}}%[[VAL_37]], %[[VAL_3]]] {index = 0 : i8, offset = 0 : i32} : memref<32x16xi32>, vector<16xi32>
    # CHECK:               %[[VAL_39:.*]] = aievec.mac_elem %[[VAL_35]], %[[VAL_38]], %[[VAL_34]] : vector<16xi32>, vector<16xi32>, vector<16xi64>
    # CHECK:               %[[VAL_40:.*]] = aievec.broadcast %[[VAL_14]] {idx = 5 : i8} : vector<16xi32>, vector<16xi32>
    # CHECK:               %[[VAL_41:.*]] = arith.constant 5 : index
    # CHECK:               %[[VAL_42:.*]] = arith.addi %[[VAL_13]], %[[VAL_41]] : index
    # CHECK:               %[[VAL_43:.*]] = aievec.upd %[[VAL_1]]{{\[}}%[[VAL_42]], %[[VAL_3]]] {index = 0 : i8, offset = 0 : i32} : memref<32x16xi32>, vector<16xi32>
    # CHECK:               %[[VAL_44:.*]] = aievec.mac_elem %[[VAL_40]], %[[VAL_43]], %[[VAL_39]] : vector<16xi32>, vector<16xi32>, vector<16xi64>
    # CHECK:               %[[VAL_45:.*]] = aievec.broadcast %[[VAL_14]] {idx = 6 : i8} : vector<16xi32>, vector<16xi32>
    # CHECK:               %[[VAL_46:.*]] = arith.constant 6 : index
    # CHECK:               %[[VAL_47:.*]] = arith.addi %[[VAL_13]], %[[VAL_46]] : index
    # CHECK:               %[[VAL_48:.*]] = aievec.upd %[[VAL_1]]{{\[}}%[[VAL_47]], %[[VAL_3]]] {index = 0 : i8, offset = 0 : i32} : memref<32x16xi32>, vector<16xi32>
    # CHECK:               %[[VAL_49:.*]] = aievec.mac_elem %[[VAL_45]], %[[VAL_48]], %[[VAL_44]] : vector<16xi32>, vector<16xi32>, vector<16xi64>
    # CHECK:               %[[VAL_50:.*]] = aievec.broadcast %[[VAL_14]] {idx = 7 : i8} : vector<16xi32>, vector<16xi32>
    # CHECK:               %[[VAL_51:.*]] = arith.constant 7 : index
    # CHECK:               %[[VAL_52:.*]] = arith.addi %[[VAL_13]], %[[VAL_51]] : index
    # CHECK:               %[[VAL_53:.*]] = aievec.upd %[[VAL_1]]{{\[}}%[[VAL_52]], %[[VAL_3]]] {index = 0 : i8, offset = 0 : i32} : memref<32x16xi32>, vector<16xi32>
    # CHECK:               %[[VAL_54:.*]] = aievec.mac_elem %[[VAL_50]], %[[VAL_53]], %[[VAL_49]] : vector<16xi32>, vector<16xi32>, vector<16xi64>
    # CHECK:               %[[VAL_55:.*]] = arith.constant 0 : i32
    # CHECK:               %[[VAL_56:.*]] = aievec.srs %[[VAL_54]], %[[VAL_55]] : vector<16xi64>, i32, vector<16xi32>
    # CHECK:               vector.transfer_write %[[VAL_56]], %[[VAL_2]]{{\[}}%[[VAL_7]], %[[VAL_3]]] {in_bounds = [true]} : vector<16xi32>, memref<16x16xi32>
    # CHECK:             }
    # CHECK:           }
    # CHECK:           return
    # CHECK:         }
    print(module)
