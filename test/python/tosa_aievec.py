# Copyright (C) 2023, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import inspect
from pathlib import Path

# noinspection PyUnresolvedReferences
import aie.dialects.aie

# noinspection PyUnresolvedReferences
import aie.dialects.aievec
from aie.dialects import affine
from aie.dialects import vector, aievec, scf
from aie.dialects.aie import translate_aie_vec_to_cpp
from aie.extras import types as T
from aie.extras.dialects import arith
from aie.extras.dialects.func import func
from aie.extras.passes import Pipeline as p, run_pipeline
from aie.extras.util import mlir_mod_ctx
from aie.ir import ShapedType, AffineMap

from util import construct_and_print_module

# RUN: %python %s | FileCheck %s

S = ShapedType.get_dynamic_size()

THIS_DIR = Path(__file__).parent.absolute()


def get_asm(operation):
    return operation.get_asm(enable_debug_info=True, pretty_debug_info=True).replace(
        str(THIS_DIR), "THIS_DIR"
    )


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
    def mul_mul(
        A: T.memref(2048, T.f32()),
        B: T.memref(2048, T.f32()),
        C: T.memref(2048, T.f32()),
        d: T.f32(),
    ):
        v0 = vector.broadcast(T.vector(8, T.f32()), d)
        v1 = aievec.concat([v0, v0])
        for i in scf.for_(0, 2048, 8):
            v2 = aievec.upd(T.vector(8, T.f32()), A, [i])
            v3 = aievec.upd(T.vector(8, T.f32()), B, [i])
            v4 = aievec.mul(
                T.vector(8, T.f32()),
                v1,
                v2,
                xoffsets="0x76543210",
                xstart="0",
                zoffsets="0x76543210",
                zstart="0",
            )
            v5 = aievec.concat([v4, v4])
            v6 = aievec.mul(
                T.vector(8, T.f32()),
                v5,
                v3,
                xoffsets="0x76543210",
                xstart="0",
                zoffsets="0x76543210",
                zstart="0",
            )
            vector.transfer_write(
                None,
                v6,
                C,
                [i],
                AffineMap.get_identity(1),
                in_bounds=[True],
            )

            scf.yield_([])

    # CHECK-LABEL:   func.func @mul_mul(
    # CHECK-SAME:                       %[[VAL_0:.*]]: memref<2048xf32>, %[[VAL_1:.*]]: memref<2048xf32>, %[[VAL_2:.*]]: memref<2048xf32>, %[[VAL_3:.*]]: f32) {
    # CHECK:           %[[VAL_4:.*]] = vector.broadcast %[[VAL_3]] : f32 to vector<8xf32>
    # CHECK:           %[[VAL_5:.*]] = aievec.concat %[[VAL_4]], %[[VAL_4]] : vector<8xf32>, vector<16xf32>
    # CHECK:           %[[VAL_6:.*]] = arith.constant 0 : index
    # CHECK:           %[[VAL_7:.*]] = arith.constant 2048 : index
    # CHECK:           %[[VAL_8:.*]] = arith.constant 8 : index
    # CHECK:           scf.for %[[VAL_9:.*]] = %[[VAL_6]] to %[[VAL_7]] step %[[VAL_8]] {
    # CHECK:             %[[VAL_10:.*]] = aievec.upd %[[VAL_0]]{{\[}}%[[VAL_9]]] {index = 0 : i8, offset = 0 : i32} : memref<2048xf32>, vector<8xf32>
    # CHECK:             %[[VAL_11:.*]] = aievec.upd %[[VAL_1]]{{\[}}%[[VAL_9]]] {index = 0 : i8, offset = 0 : i32} : memref<2048xf32>, vector<8xf32>
    # CHECK:             %[[VAL_12:.*]] = aievec.mul %[[VAL_5]], %[[VAL_10]] {xoffsets = "0x76543210", xstart = "0", zoffsets = "0x76543210", zstart = "0"} : vector<16xf32>, vector<8xf32>, vector<8xf32>
    # CHECK:             %[[VAL_13:.*]] = aievec.concat %[[VAL_12]], %[[VAL_12]] : vector<8xf32>, vector<16xf32>
    # CHECK:             %[[VAL_14:.*]] = aievec.mul %[[VAL_13]], %[[VAL_11]] {xoffsets = "0x76543210", xstart = "0", zoffsets = "0x76543210", zstart = "0"} : vector<16xf32>, vector<8xf32>, vector<8xf32>
    # CHECK:             vector.transfer_write %[[VAL_14]], %[[VAL_2]]{{\[}}%[[VAL_9]]] {in_bounds = [true]} : vector<8xf32>, memref<2048xf32>
    # CHECK:           }
    # CHECK:           return
    # CHECK:         }
    mul_mul.emit()


def test_gemm64_int16():
    range_ = affine.for_

    with mlir_mod_ctx() as ctx:
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
        print(ctx.module)

        mod = run_pipeline(
            ctx.module,
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


test_gemm64_int16()
