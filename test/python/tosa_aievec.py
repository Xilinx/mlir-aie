# Copyright (C) 2023, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import inspect
from pathlib import Path
from textwrap import dedent

from aie.extras.dialects import arith
from aie.extras.dialects.func import func
from aie.extras.util import mlir_mod_ctx
from aie.ir import ShapedType
from util import construct_and_print_module
from inspect import currentframe, getframeinfo

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


def test_location_tracking():
    with mlir_mod_ctx() as ctx:

        frameinfo = getframeinfo(currentframe())

        @func
        def demo_fun1():
            one = arith.constant(1)
            return one

        demo_fun1.emit()

    asm = get_asm(ctx.module.operation)
    correct = dedent(
        f"""\
    module {{
      func.func @demo_fun1() -> i32 {{
        %c1_i32 = arith.constant 1 : i32 THIS_DIR/tosa_aievec.py:{frameinfo.lineno + 4}:18
        return %c1_i32 : i32 [unknown]
      }} THIS_DIR/tosa_aievec.py:{frameinfo.lineno + 2}:9
    }} [unknown]
    #loc = [unknown]
    #loc1 = THIS_DIR/tosa_aievec.py:{frameinfo.lineno + 2}:9
    #loc2 = THIS_DIR/tosa_aievec.py:{frameinfo.lineno + 4}:18
    """
    )
    assert asm == correct


test_location_tracking()
