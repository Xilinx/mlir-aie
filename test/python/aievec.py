# Copyright (C) 2023, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %python %s | FileCheck %s

import inspect

from aie.extras.dialects.ext import arith
from aie.extras.dialects.ext.func import func
from aie.extras.runtime.passes import Pipeline as p, run_pipeline

from aie.dialects import affine, aievec, tosa, vector

# noinspection PyUnresolvedReferences
import aie.dialects.aie
from aie.dialects.aie import translate_aie_vec_to_cpp

# noinspection PyUnresolvedReferences
from aie.extras import types as T
from aie.ir import AffineMap, AffineDimExpr
from util import construct_and_print_module
from aie.extras.dialects.ext.scf import _for as range_


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


@construct_and_print_module
def test_gemm64_int16(module):
    range_ = affine.for_

    id_map = affine.AffineMap.get_identity(2)

    @func
    def matmul(
        A: T.memref(64, 64, T.i16()),
        B: T.memref(64, 64, T.i16()),
        C: T.memref(64, 64, T.i16()),
    ):
        for i in range_(0, 64):
            for j in range_(0, 64):
                for k in range_(0, 64):
                    a = affine.load(T.i16(), A, [i, k], id_map)
                    b = affine.load(T.i16(), B, [k, j], id_map)
                    c = affine.load(T.i16(), C, [i, j], id_map)
                    d = arith.muli(a, b)
                    e = arith.addi(c, d)
                    affine.store(e, C, [i, j], id_map)

                    affine.yield_([])
                affine.yield_([])
            affine.yield_([])

    matmul.emit()

    # CHECK:  func.func @matmul(%arg0: memref<64x64xi16>, %arg1: memref<64x64xi16>, %arg2: memref<64x64xi16>) {
    # CHECK:    affine.for %arg3 = 0 to 64 {
    # CHECK:      affine.for %arg4 = 0 to 64 {
    # CHECK:        affine.for %arg5 = 0 to 64 {
    # CHECK:          %0 = affine.load %arg0[%arg3, %arg5] : memref<64x64xi16>
    # CHECK:          %1 = affine.load %arg1[%arg5, %arg4] : memref<64x64xi16>
    # CHECK:          %2 = affine.load %arg2[%arg3, %arg4] : memref<64x64xi16>
    # CHECK:          %3 = arith.muli %0, %1 : i16
    # CHECK:          %4 = arith.addi %2, %3 : i16
    # CHECK:          affine.store %4, %arg2[%arg3, %arg4] : memref<64x64xi16>
    # CHECK:        }
    # CHECK:      }
    # CHECK:    }
    # CHECK:    return
    # CHECK:  }
    print(module)

    mod = run_pipeline(
        module,
        p().Func(
            p()
            .affine_loop_unroll(unroll_factor=16)
            .affine_scalrep()
            .canonicalize()
            .affine_super_vectorize(virtual_vector_size=16)
            .add_pass("convert-vector-to-aievec")
            .lower_affine()
            .canonicalize()
        ),
    )
    print(mod)

    cpp = translate_aie_vec_to_cpp(mod.operation)
    # CHECK: void matmul(int16_t * restrict v1, int16_t * restrict v2, int16_t * restrict v3) {
    # CHECK:   size_t v4 = 15;
    # CHECK:   size_t v5 = 14;
    # CHECK:   size_t v6 = 13;
    # CHECK:   size_t v7 = 12;
    # CHECK:   size_t v8 = 11;
    # CHECK:   size_t v9 = 10;
    # CHECK:   size_t v10 = 9;
    # CHECK:   size_t v11 = 8;
    # CHECK:   size_t v12 = 7;
    # CHECK:   size_t v13 = 6;
    # CHECK:   size_t v14 = 5;
    # CHECK:   size_t v15 = 4;
    # CHECK:   size_t v16 = 3;
    # CHECK:   size_t v17 = 2;
    # CHECK:   size_t v18 = 16;
    # CHECK:   int32_t v19 = 0;
    # CHECK:   size_t v20 = 0;
    # CHECK:   size_t v21 = 64;
    # CHECK:   size_t v22 = 1;
    # CHECK:   for (size_t v23 = v20; v23 < v21; v23 += v22)
    # CHECK:   chess_prepare_for_pipelining
    # CHECK:   chess_loop_range(64, 64)
    # CHECK:   {
    # CHECK:     for (size_t v24 = v20; v24 < v21; v24 += v18)
    # CHECK:     chess_prepare_for_pipelining
    # CHECK:     chess_loop_range(4, 4)
    # CHECK:     {
    # CHECK:       v16int16 v25 = *(v16int16 *)(v3 + 64*v23+v24);
    # CHECK:       v16acc48 v26 = ups(v25, 0);
    # CHECK:       for (size_t v27 = v20; v27 < v21; v27 += v18)
    # CHECK:       chess_prepare_for_pipelining
    # CHECK:       chess_loop_range(4, 4)
    # CHECK:       {
    # CHECK:         v16int16 v28 = *(v16int16 *)(v1 + 64*v23+v27);
    # CHECK:         v16int16 v29 = *(v16int16 *)(v2 + 64*v27+v24);
    # CHECK:         size_t v30 = v27 + v22;
    # CHECK:         v16int16 v31 = *(v16int16 *)(v2 + 64*v30+v24);
    # CHECK:         v32int16 v32 = concat(v29, v31);
    # CHECK:         v26 = mac16(v26, v32, 0, 0x73727170, 0x77767574, 0x3120, v28, 0, 0, 0, 1);
    # CHECK:         size_t v33 = v27 + v17;
    # CHECK:         v16int16 v34 = *(v16int16 *)(v2 + 64*v33+v24);
    # CHECK:         size_t v35 = v27 + v16;
    # CHECK:         v16int16 v36 = *(v16int16 *)(v2 + 64*v35+v24);
    # CHECK:         v32int16 v37 = concat(v34, v36);
    # CHECK:         v26 = mac16(v26, v37, 0, 0x73727170, 0x77767574, 0x3120, v28, 2, 0, 0, 1);
    # CHECK:         size_t v38 = v27 + v15;
    # CHECK:         v16int16 v39 = *(v16int16 *)(v2 + 64*v38+v24);
    # CHECK:         size_t v40 = v27 + v14;
    # CHECK:         v16int16 v41 = *(v16int16 *)(v2 + 64*v40+v24);
    # CHECK:         v32int16 v42 = concat(v39, v41);
    # CHECK:         v26 = mac16(v26, v42, 0, 0x73727170, 0x77767574, 0x3120, v28, 4, 0, 0, 1);
    # CHECK:         size_t v43 = v27 + v13;
    # CHECK:         v16int16 v44 = *(v16int16 *)(v2 + 64*v43+v24);
    # CHECK:         size_t v45 = v27 + v12;
    # CHECK:         v16int16 v46 = *(v16int16 *)(v2 + 64*v45+v24);
    # CHECK:         v32int16 v47 = concat(v44, v46);
    # CHECK:         v26 = mac16(v26, v47, 0, 0x73727170, 0x77767574, 0x3120, v28, 6, 0, 0, 1);
    # CHECK:         size_t v48 = v27 + v11;
    # CHECK:         v16int16 v49 = *(v16int16 *)(v2 + 64*v48+v24);
    # CHECK:         size_t v50 = v27 + v10;
    # CHECK:         v16int16 v51 = *(v16int16 *)(v2 + 64*v50+v24);
    # CHECK:         v32int16 v52 = concat(v49, v51);
    # CHECK:         v26 = mac16(v26, v52, 0, 0x73727170, 0x77767574, 0x3120, v28, 8, 0, 0, 1);
    # CHECK:         size_t v53 = v27 + v9;
    # CHECK:         v16int16 v54 = *(v16int16 *)(v2 + 64*v53+v24);
    # CHECK:         size_t v55 = v27 + v8;
    # CHECK:         v16int16 v56 = *(v16int16 *)(v2 + 64*v55+v24);
    # CHECK:         v32int16 v57 = concat(v54, v56);
    # CHECK:         v26 = mac16(v26, v57, 0, 0x73727170, 0x77767574, 0x3120, v28, 10, 0, 0, 1);
    # CHECK:         size_t v58 = v27 + v7;
    # CHECK:         v16int16 v59 = *(v16int16 *)(v2 + 64*v58+v24);
    # CHECK:         size_t v60 = v27 + v6;
    # CHECK:         v16int16 v61 = *(v16int16 *)(v2 + 64*v60+v24);
    # CHECK:         v32int16 v62 = concat(v59, v61);
    # CHECK:         v26 = mac16(v26, v62, 0, 0x73727170, 0x77767574, 0x3120, v28, 12, 0, 0, 1);
    # CHECK:         size_t v63 = v27 + v5;
    # CHECK:         v16int16 v64 = *(v16int16 *)(v2 + 64*v63+v24);
    # CHECK:         size_t v65 = v27 + v4;
    # CHECK:         v16int16 v66 = *(v16int16 *)(v2 + 64*v65+v24);
    # CHECK:         v32int16 v67 = concat(v64, v66);
    # CHECK:         v26 = mac16(v26, v67, 0, 0x73727170, 0x77767574, 0x3120, v28, 14, 0, 0, 1);
    # CHECK:         v16int16 v68 = srs(v26, v19);
    # CHECK:         *(v16int16 *)(v3 + 64*v23+v24) = v68;
    # CHECK:       }
    # CHECK:     }
    # CHECK:   }
    # CHECK:   return;
    # CHECK: }
    print(cpp)


@construct_and_print_module
def test_i8xi8_add_elem(module):
    @func
    def dut(A: T.tensor(1024, T.i8()), B: T.tensor(1024, T.i8())):
        v1 = tosa.add(T.tensor(1024, T.i8()), A, B)
        return v1

    dut.emit()

    pipe = (
        p()
        .Func(
            p()
            .add_pass("tosa-to-linalg-named")
            .add_pass("tosa-to-linalg")
            .tosa_to_tensor()
            .add_pass("dynamic-size-no-implicit-broadcast")
            .linalg_fuse_elementwise_ops()
            .linalg_fold_unit_extent_dims()
            .eliminate_empty_tensors()
            .empty_tensor_to_alloc_tensor()
        )
        .one_shot_bufferize(
            # allow_return_allocs=True,
            allow_unknown_ops=True,
            bufferize_function_boundaries=True,
            function_boundary_type_conversion="identity-layout-map",
            unknown_type_conversion="identity-layout-map",
        )
        .drop_equivalent_buffer_results()
        .buffer_results_to_out_params()
        .Func(p().buffer_deallocation())
        .canonicalize()
        .cse()
        .convert_linalg_to_affine_loops()
        .Func(p().affine_super_vectorize(virtual_vector_size=64))
        .add_pass("convert-vector-to-aievec", aie_target="aie2")
        .lower_affine()
    )

    mod = run_pipeline(module, pipe)
    # CHECK: func.func @dut(%arg0: memref<1024xi8>, %arg1: memref<1024xi8>, %arg2: memref<1024xi8>) {
    # CHECK:   %c0 = arith.constant 0 : index
    # CHECK:   %c1024 = arith.constant 1024 : index
    # CHECK:   %c64 = arith.constant 64 : index
    # CHECK:   scf.for %arg3 = %c0 to %c1024 step %c64 {
    # CHECK:     %0 = aievec.upd %arg0[%arg3] {index = 0 : i8, offset = 0 : i32} : memref<1024xi8>, vector<64xi8>
    # CHECK:     %1 = aievec.upd %arg1[%arg3] {index = 0 : i8, offset = 0 : i32} : memref<1024xi8>, vector<64xi8>
    # CHECK:     %2 = aievec.add_elem %0, %1 : vector<64xi8>
    # CHECK:     vector.transfer_write %2, %arg2[%arg3] {in_bounds = [true]} : vector<64xi8>, memref<1024xi8>
    # CHECK:   }
    # CHECK:   return
    # CHECK: }
    print(mod)

    cpp = translate_aie_vec_to_cpp(mod.operation, aie2=True)
    # CHECK: void dut(int8_t * restrict v1, int8_t * restrict v2, int8_t * restrict v3) {
    # CHECK:   size_t v4 = 0;
    # CHECK:   size_t v5 = 1024;
    # CHECK:   size_t v6 = 64;
    # CHECK:   for (size_t v7 = v4; v7 < v5; v7 += v6)
    # CHECK:   chess_prepare_for_pipelining
    # CHECK:   chess_loop_range(16, 16)
    # CHECK:   {
    # CHECK:     v64int8 v8 = *(v64int8 *)(v1 + v7);
    # CHECK:     v64int8 v9 = *(v64int8 *)(v2 + v7);
    # CHECK:     v64int8 v10 = add(v8, v9);
    # CHECK:     *(v64int8 *)(v3 + v7) = v10;
    # CHECK:   }
    # CHECK:   return;
    # CHECK: }
    print(cpp)


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
