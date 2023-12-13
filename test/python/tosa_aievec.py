# Copyright (C) 2023, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import inspect
from pathlib import Path

# noinspection PyUnresolvedReferences
import aie.dialects.aievec
from aie.extras.dialects import arith
from aie.extras.dialects.func import func
from aie.extras import types as T
from aie.ir import ShapedType, AffineMap
from aie.dialects import vector, aievec, scf
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
