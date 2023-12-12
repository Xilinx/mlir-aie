# Copyright (C) 2022, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %PYTHON %s | FileCheck %s
# REQUIRES: python_passes

from pathlib import Path
from textwrap import dedent

# noinspection PyUnresolvedReferences
import aie.dialects.aie
from aie._mlir_libs._aie_python_passes import (
    create_python_router_pass,
    pass_manager_add_owned_pass,
)

from aie.ir import Context, Location, Module
from aie.passmanager import PassManager
from aie.extras.util import Router

TIMEOUT = 10


def run(f):
    with Context(), Location.unknown():
        print("\nTEST:", f.__name__)
        f()


THIS_FILE = __file__

# CHECK-LABEL: TEST: test_broadcast
@run
def test_broadcast():
    with open(Path(THIS_FILE).parent.parent / "create-flows" / "broadcast.mlir") as f:
        mlir_module = Module.parse(f.read())
    r = Router(timeout=TIMEOUT)
    pass_ = create_python_router_pass(r)
    pm = PassManager()
    pass_manager_add_owned_pass(pm, pass_)
    pm.add("aie-find-flows")

    device = mlir_module.body.operations[0]
    pm.run(device.operation)

    # CHECK: %[[T03:.*]] = AIE.tile(0, 3)
    # CHECK: %[[T02:.*]] = AIE.tile(0, 2)
    # CHECK: %[[T00:.*]] = AIE.tile(0, 0)
    # CHECK: %[[T13:.*]] = AIE.tile(1, 3)
    # CHECK: %[[T11:.*]] = AIE.tile(1, 1)
    # CHECK: %[[T10:.*]] = AIE.tile(1, 0)
    # CHECK: %[[T20:.*]] = AIE.tile(2, 0)
    # CHECK: %[[T30:.*]] = AIE.tile(3, 0)
    # CHECK: %[[T22:.*]] = AIE.tile(2, 2)
    # CHECK: %[[T31:.*]] = AIE.tile(3, 1)
    # CHECK: %[[T60:.*]] = AIE.tile(6, 0)
    # CHECK: %[[T70:.*]] = AIE.tile(7, 0)
    # CHECK: %[[T71:.*]] = AIE.tile(7, 1)
    # CHECK: %[[T72:.*]] = AIE.tile(7, 2)
    # CHECK: %[[T73:.*]] = AIE.tile(7, 3)
    # CHECK: %[[T80:.*]] = AIE.tile(8, 0)
    # CHECK: %[[T82:.*]] = AIE.tile(8, 2)
    # CHECK: %[[T83:.*]] = AIE.tile(8, 3)
    #
    # CHECK: AIE.flow(%[[T20]], DMA : 0, %[[T71]], DMA : 0)
    # CHECK: AIE.flow(%[[T20]], DMA : 0, %[[T31]], DMA : 0)
    # CHECK: AIE.flow(%[[T20]], DMA : 0, %[[T82]], DMA : 0)
    # CHECK: AIE.flow(%[[T20]], DMA : 0, %[[T13]], DMA : 0)
    # CHECK: AIE.flow(%[[T60]], DMA : 0, %[[T83]], DMA : 1)
    # CHECK: AIE.flow(%[[T60]], DMA : 0, %[[T22]], DMA : 1)
    # CHECK: AIE.flow(%[[T60]], DMA : 0, %[[T02]], DMA : 1)
    # CHECK: AIE.flow(%[[T60]], DMA : 0, %[[T31]], DMA : 1)
    print(mlir_module)


# CHECK-LABEL: TEST: test_flow_test_1
@run
def test_flow_test_1():
    with open(Path(THIS_FILE).parent.parent / "create-flows" / "flow_test_1.mlir") as f:
        mlir_module = Module.parse(f.read())
    r = Router(timeout=TIMEOUT)
    pass_ = create_python_router_pass(r)
    pm = PassManager()
    pass_manager_add_owned_pass(pm, pass_)
    pm.add("aie-find-flows")

    device = mlir_module.body.operations[0]
    pm.run(device.operation)

    # CHECK: %[[t20:.*]] = AIE.tile(2, 0)
    # CHECK: %[[t30:.*]] = AIE.tile(3, 0)
    # CHECK: %[[t34:.*]] = AIE.tile(3, 4)
    # CHECK: %[[t43:.*]] = AIE.tile(4, 3)
    # CHECK: %[[t44:.*]] = AIE.tile(4, 4)
    # CHECK: %[[t54:.*]] = AIE.tile(5, 4)
    # CHECK: %[[t60:.*]] = AIE.tile(6, 0)
    # CHECK: %[[t63:.*]] = AIE.tile(6, 3)
    # CHECK: %[[t70:.*]] = AIE.tile(7, 0)
    # CHECK: %[[t72:.*]] = AIE.tile(7, 2)
    # CHECK: %[[t83:.*]] = AIE.tile(8, 3)
    # CHECK: %[[t84:.*]] = AIE.tile(8, 4)
    #
    # CHECK: AIE.flow(%[[t20]], DMA : 0, %[[t63]], DMA : 0)
    # CHECK: AIE.flow(%[[t20]], DMA : 1, %[[t83]], DMA : 0)
    # CHECK: AIE.flow(%[[t30]], DMA : 0, %[[t72]], DMA : 0)
    # CHECK: AIE.flow(%[[t30]], DMA : 1, %[[t54]], DMA : 0)
    #
    # CHECK: AIE.flow(%[[t34]], Core : 0, %[[t63]], Core : 1)
    # CHECK: AIE.flow(%[[t34]], DMA : 1, %[[t70]], DMA : 0)
    # CHECK: AIE.flow(%[[t43]], Core : 0, %[[t84]], Core : 1)
    # CHECK: AIE.flow(%[[t43]], DMA : 1, %[[t60]], DMA : 1)
    #
    # CHECK: AIE.flow(%[[t44]], Core : 0, %[[t54]], Core : 1)
    # CHECK: AIE.flow(%[[t44]], DMA : 1, %[[t60]], DMA : 0)
    # CHECK: AIE.flow(%[[t54]], Core : 0, %[[t43]], Core : 1)
    # CHECK: AIE.flow(%[[t54]], DMA : 1, %[[t30]], DMA : 1)
    #
    # CHECK: AIE.flow(%[[t60]], DMA : 0, %[[t44]], DMA : 0)
    # CHECK: AIE.flow(%[[t60]], DMA : 1, %[[t43]], DMA : 0)
    # CHECK: AIE.flow(%[[t63]], Core : 0, %[[t34]], Core : 1)
    # CHECK: AIE.flow(%[[t63]], DMA : 1, %[[t20]], DMA : 1)
    #
    # CHECK: AIE.flow(%[[t70]], DMA : 0, %[[t34]], DMA : 0)
    # CHECK: AIE.flow(%[[t70]], DMA : 1, %[[t84]], DMA : 0)
    # CHECK: AIE.flow(%[[t72]], Core : 0, %[[t83]], Core : 1)
    # CHECK: AIE.flow(%[[t72]], DMA : 1, %[[t30]], DMA : 0)
    #
    # CHECK: AIE.flow(%[[t83]], Core : 0, %[[t44]], Core : 1)
    # CHECK: AIE.flow(%[[t83]], DMA : 1, %[[t20]], DMA : 0)
    # CHECK: AIE.flow(%[[t84]], Core : 0, %[[t72]], Core : 1)
    # CHECK: AIE.flow(%[[t84]], DMA : 1, %[[t70]], DMA : 1)
    print(mlir_module)


# CHECK-LABEL: TEST: test_flow_test_2
@run
def test_flow_test_2():
    with open(Path(THIS_FILE).parent.parent / "create-flows" / "flow_test_2.mlir") as f:
        mlir_module = Module.parse(f.read())
    r = Router(timeout=TIMEOUT)
    pass_ = create_python_router_pass(r)
    pm = PassManager()
    pass_manager_add_owned_pass(pm, pass_)
    pm.add("aie-find-flows")
    device = mlir_module.body.operations[0]
    pm.run(device.operation)

    # CHECK: %[[t01:.*]] = AIE.tile(0, 1)
    # CHECK: %[[t02:.*]] = AIE.tile(0, 2)
    # CHECK: %[[t03:.*]] = AIE.tile(0, 3)
    # CHECK: %[[t04:.*]] = AIE.tile(0, 4)
    # CHECK: %[[t11:.*]] = AIE.tile(1, 1)
    # CHECK: %[[t12:.*]] = AIE.tile(1, 2)
    # CHECK: %[[t13:.*]] = AIE.tile(1, 3)
    # CHECK: %[[t14:.*]] = AIE.tile(1, 4)
    # CHECK: %[[t20:.*]] = AIE.tile(2, 0)
    # CHECK: %[[t21:.*]] = AIE.tile(2, 1)
    # CHECK: %[[t22:.*]] = AIE.tile(2, 2)
    # CHECK: %[[t23:.*]] = AIE.tile(2, 3)
    # CHECK: %[[t24:.*]] = AIE.tile(2, 4)
    # CHECK: %[[t30:.*]] = AIE.tile(3, 0)
    # CHECK: %[[t31:.*]] = AIE.tile(3, 1)
    # CHECK: %[[t32:.*]] = AIE.tile(3, 2)
    # CHECK: %[[t33:.*]] = AIE.tile(3, 3)
    # CHECK: %[[t34:.*]] = AIE.tile(3, 4)

    # CHECK: AIE.flow(%[[t01]], Core : 0, %[[t12]], Core : 0)
    # CHECK: AIE.flow(%[[t02]], DMA : 0, %[[t20]], DMA : 0)
    # CHECK: AIE.flow(%[[t04]], Core : 0, %[[t13]], Core : 0)
    # CHECK: AIE.flow(%[[t11]], Core : 0, %[[t01]], Core : 0)
    # CHECK: AIE.flow(%[[t12]], Core : 0, %[[t02]], Core : 0)
    # CHECK: AIE.flow(%[[t13]], DMA : 0, %[[t20]], DMA : 1)
    # CHECK: AIE.flow(%[[t14]], Core : 0, %[[t04]], Core : 0)
    # CHECK: AIE.flow(%[[t20]], DMA : 0, %[[t11]], DMA : 0)
    # CHECK: AIE.flow(%[[t20]], DMA : 1, %[[t14]], DMA : 0)
    # CHECK: AIE.flow(%[[t21]], Core : 0, %[[t33]], Core : 0)
    # CHECK: AIE.flow(%[[t22]], Core : 0, %[[t34]], Core : 0)
    # CHECK: AIE.flow(%[[t23]], Core : 1, %[[t34]], Core : 1)
    # CHECK: AIE.flow(%[[t23]], DMA : 0, %[[t30]], DMA : 0)
    # CHECK: AIE.flow(%[[t24]], Core : 0, %[[t23]], Core : 0)
    # CHECK: AIE.flow(%[[t24]], Core : 1, %[[t33]], Core : 1)
    # CHECK: AIE.flow(%[[t30]], DMA : 0, %[[t21]], DMA : 0)
    # CHECK: AIE.flow(%[[t30]], DMA : 1, %[[t31]], DMA : 1)
    # CHECK: AIE.flow(%[[t31]], Core : 1, %[[t23]], Core : 1)
    # CHECK: AIE.flow(%[[t32]], DMA : 1, %[[t30]], DMA : 1)
    # CHECK: AIE.flow(%[[t33]], Core : 0, %[[t22]], Core : 0)
    # CHECK: AIE.flow(%[[t33]], Core : 1, %[[t32]], Core : 1)
    # CHECK: AIE.flow(%[[t34]], Core : 0, %[[t24]], Core : 0)
    # CHECK: AIE.flow(%[[t34]], Core : 1, %[[t24]], Core : 1)
    print(mlir_module)


# CHECK-LABEL: TEST: test_flow_test_3
@run
def test_flow_test_3():
    with open(Path(THIS_FILE).parent.parent / "create-flows" / "flow_test_3.mlir") as f:
        mlir_module = Module.parse(f.read())
    r = Router(timeout=TIMEOUT)
    pass_ = create_python_router_pass(r)
    pm = PassManager()
    pass_manager_add_owned_pass(pm, pass_)
    pm.add("aie-find-flows")

    device = mlir_module.body.operations[0]
    pm.run(device.operation)

    # CHECK: %[[t01:.*]] = AIE.tile(0, 1)
    # CHECK: %[[t02:.*]] = AIE.tile(0, 2)
    # CHECK: %[[t03:.*]] = AIE.tile(0, 3)
    # CHECK: %[[t04:.*]] = AIE.tile(0, 4)
    # CHECK: %[[t11:.*]] = AIE.tile(1, 1)
    # CHECK: %[[t12:.*]] = AIE.tile(1, 2)
    # CHECK: %[[t13:.*]] = AIE.tile(1, 3)
    # CHECK: %[[t14:.*]] = AIE.tile(1, 4)
    # CHECK: %[[t20:.*]] = AIE.tile(2, 0)
    # CHECK: %[[t21:.*]] = AIE.tile(2, 1)
    # CHECK: %[[t22:.*]] = AIE.tile(2, 2)
    # CHECK: %[[t23:.*]] = AIE.tile(2, 3)
    # CHECK: %[[t24:.*]] = AIE.tile(2, 4)
    # CHECK: %[[t30:.*]] = AIE.tile(3, 0)
    # CHECK: %[[t71:.*]] = AIE.tile(7, 1)
    # CHECK: %[[t72:.*]] = AIE.tile(7, 2)
    # CHECK: %[[t73:.*]] = AIE.tile(7, 3)
    # CHECK: %[[t74:.*]] = AIE.tile(7, 4)
    # CHECK: %[[t81:.*]] = AIE.tile(8, 1)
    # CHECK: %[[t82:.*]] = AIE.tile(8, 2)
    # CHECK: %[[t83:.*]] = AIE.tile(8, 3)
    # CHECK: %[[t84:.*]] = AIE.tile(8, 4)
    #
    # CHECK: AIE.flow(%[[t01]], Core : 0, %[[t83]], Core : 0)
    # CHECK: AIE.flow(%[[t01]], Core : 1, %[[t72]], Core : 1)
    # CHECK: AIE.flow(%[[t02]], Core : 1, %[[t24]], Core : 1)
    # CHECK: AIE.flow(%[[t03]], Core : 0, %[[t71]], Core : 0)
    # CHECK: AIE.flow(%[[t11]], Core : 0, %[[t24]], Core : 0)
    # CHECK: AIE.flow(%[[t14]], Core : 0, %[[t01]], Core : 0)
    # CHECK: AIE.flow(%[[t20]], DMA : 0, %[[t03]], DMA : 0)
    # CHECK: AIE.flow(%[[t20]], DMA : 1, %[[t83]], DMA : 1)
    # CHECK: AIE.flow(%[[t21]], Core : 0, %[[t73]], Core : 0)
    # CHECK: AIE.flow(%[[t24]], Core : 1, %[[t71]], Core : 1)
    # CHECK: AIE.flow(%[[t24]], DMA : 0, %[[t20]], DMA : 0)
    # CHECK: AIE.flow(%[[t30]], DMA : 0, %[[t14]], DMA : 0)
    # CHECK: AIE.flow(%[[t71]], Core : 0, %[[t84]], Core : 0)
    # CHECK: AIE.flow(%[[t71]], Core : 1, %[[t84]], Core : 1)
    # CHECK: AIE.flow(%[[t72]], Core : 1, %[[t02]], Core : 1)
    # CHECK: AIE.flow(%[[t73]], Core : 0, %[[t82]], Core : 0)
    # CHECK: AIE.flow(%[[t82]], DMA : 0, %[[t30]], DMA : 0)
    # CHECK: AIE.flow(%[[t83]], Core : 0, %[[t21]], Core : 0)
    # CHECK: AIE.flow(%[[t83]], Core : 1, %[[t01]], Core : 1)
    # CHECK: AIE.flow(%[[t84]], Core : 0, %[[t11]], Core : 0)
    # CHECK: AIE.flow(%[[t84]], DMA : 1, %[[t20]], DMA : 1)
    print(mlir_module)


# CHECK-LABEL: TEST: test_many_flows
@run
def test_many_flows():
    with open(Path(THIS_FILE).parent.parent / "create-flows" / "many_flows.mlir") as f:
        mlir_module = Module.parse(f.read())
    r = Router(timeout=TIMEOUT)
    pass_ = create_python_router_pass(r)
    pm = PassManager()
    pass_manager_add_owned_pass(pm, pass_)
    pm.add("aie-find-flows")

    device = mlir_module.body.operations[0]
    pm.run(device.operation)

    # CHECK: %[[T02:.*]] = AIE.tile(0, 2)
    # CHECK: %[[T03:.*]] = AIE.tile(0, 3)
    # CHECK: %[[T11:.*]] = AIE.tile(1, 1)
    # CHECK: %[[T13:.*]] = AIE.tile(1, 3)
    # CHECK: %[[T20:.*]] = AIE.tile(2, 0)
    # CHECK: %[[T22:.*]] = AIE.tile(2, 2)
    # CHECK: %[[T30:.*]] = AIE.tile(3, 0)
    # CHECK: %[[T31:.*]] = AIE.tile(3, 1)
    # CHECK: %[[T60:.*]] = AIE.tile(6, 0)
    # CHECK: %[[T70:.*]] = AIE.tile(7, 0)
    # CHECK: %[[T73:.*]] = AIE.tile(7, 3)
    # CHECK: AIE.flow(%[[T02]], Core : 1, %[[T22]], Core : 1)
    # CHECK: AIE.flow(%[[T02]], DMA : 0, %[[T60]], DMA : 0)
    # CHECK: AIE.flow(%[[T03]], Core : 0, %[[T13]], Core : 0)
    # CHECK: AIE.flow(%[[T03]], Core : 1, %[[T02]], Core : 0)
    # CHECK: AIE.flow(%[[T03]], DMA : 0, %[[T70]], DMA : 0)
    # CHECK: AIE.flow(%[[T13]], Core : 1, %[[T22]], Core : 0)
    # CHECK: AIE.flow(%[[T13]], DMA : 0, %[[T70]], DMA : 1)
    # CHECK: AIE.flow(%[[T22]], DMA : 0, %[[T60]], DMA : 1)
    # CHECK: AIE.flow(%[[T31]], DMA : 0, %[[T20]], DMA : 1)
    # CHECK: AIE.flow(%[[T31]], DMA : 1, %[[T30]], DMA : 1)
    # CHECK: AIE.flow(%[[T73]], Core : 0, %[[T31]], Core : 0)
    # CHECK: AIE.flow(%[[T73]], Core : 1, %[[T31]], Core : 1)
    # CHECK: AIE.flow(%[[T73]], DMA : 0, %[[T20]], DMA : 0)
    # CHECK: AIE.flow(%[[T73]], DMA : 1, %[[T30]], DMA : 0)
    print(mlir_module)


# CHECK-LABEL: TEST: test_many_flows2
@run
def test_many_flows2():
    with open(Path(THIS_FILE).parent.parent / "create-flows" / "many_flows2.mlir") as f:
        mlir_module = Module.parse(f.read())
    r = Router(timeout=TIMEOUT)
    pass_ = create_python_router_pass(r)
    pm = PassManager()
    pass_manager_add_owned_pass(pm, pass_)
    pm.add("aie-find-flows")

    device = mlir_module.body.operations[0]
    pm.run(device.operation)

    # CHECK: %[[T02:.*]] = AIE.tile(0, 2)
    # CHECK: %[[T03:.*]] = AIE.tile(0, 3)
    # CHECK: %[[T11:.*]] = AIE.tile(1, 1)
    # CHECK: %[[T13:.*]] = AIE.tile(1, 3)
    # CHECK: %[[T20:.*]] = AIE.tile(2, 0)
    # CHECK: %[[T22:.*]] = AIE.tile(2, 2)
    # CHECK: %[[T30:.*]] = AIE.tile(3, 0)
    # CHECK: %[[T31:.*]] = AIE.tile(3, 1)
    # CHECK: %[[T60:.*]] = AIE.tile(6, 0)
    # CHECK: %[[T70:.*]] = AIE.tile(7, 0)
    # CHECK: %[[T73:.*]] = AIE.tile(7, 3)
    #
    # CHECK: AIE.flow(%[[T02]], DMA : 0, %[[T60]], DMA : 0)
    # CHECK: AIE.flow(%[[T03]], Core : 0, %[[T02]], Core : 1)
    # CHECK: AIE.flow(%[[T03]], Core : 1, %[[T02]], Core : 0)
    # CHECK: AIE.flow(%[[T03]], DMA : 0, %[[T30]], DMA : 0)
    # CHECK: AIE.flow(%[[T03]], DMA : 1, %[[T70]], DMA : 1)
    # CHECK: AIE.flow(%[[T13]], Core : 1, %[[T31]], Core : 1)
    # CHECK: AIE.flow(%[[T22]], Core : 0, %[[T13]], Core : 0)
    # CHECK: AIE.flow(%[[T22]], DMA : 0, %[[T20]], DMA : 0)
    # CHECK: AIE.flow(%[[T31]], DMA : 0, %[[T20]], DMA : 1)
    # CHECK: AIE.flow(%[[T31]], DMA : 1, %[[T30]], DMA : 1)
    # CHECK: AIE.flow(%[[T73]], Core : 0, %[[T31]], Core : 0)
    # CHECK: AIE.flow(%[[T73]], Core : 1, %[[T22]], Core : 1)
    # CHECK: AIE.flow(%[[T73]], DMA : 0, %[[T60]], DMA : 1)
    # CHECK: AIE.flow(%[[T73]], DMA : 1, %[[T70]], DMA : 0)
    print(mlir_module)


# CHECK-LABEL: TEST: test_memtile
@run
def test_memtile():
    with open(Path(THIS_FILE).parent.parent / "create-flows" / "memtile.mlir") as f:
        mlir_module = Module.parse(f.read())
    r = Router(timeout=TIMEOUT)
    pass_ = create_python_router_pass(r)
    pm = PassManager()
    pass_manager_add_owned_pass(pm, pass_)
    pm.add("aie-find-flows")

    device = mlir_module.body.operations[0]
    pm.run(device.operation)

    # CHECK: %[[T04:.*]] = AIE.tile(0, 4)
    # CHECK: %[[T03:.*]] = AIE.tile(0, 3)
    # CHECK: %[[T02:.*]] = AIE.tile(0, 2)
    # CHECK: %[[T01:.*]] = AIE.tile(0, 1)
    # CHECK: AIE.flow(%[[T04]], DMA : 0, %[[T02]], DMA : 4)
    # CHECK: AIE.flow(%[[T04]], DMA : 1, %[[T02]], DMA : 5)
    # CHECK: AIE.flow(%[[T03]], DMA : 0, %[[T02]], DMA : 2)
    # CHECK: AIE.flow(%[[T03]], DMA : 1, %[[T02]], DMA : 3)
    # CHECK: AIE.flow(%[[T02]], DMA : 0, %[[T01]], DMA : 0)
    # CHECK: AIE.flow(%[[T02]], DMA : 1, %[[T01]], DMA : 1)
    # CHECK: AIE.flow(%[[T02]], DMA : 2, %[[T03]], DMA : 0)
    # CHECK: AIE.flow(%[[T02]], DMA : 3, %[[T03]], DMA : 1)
    # CHECK: AIE.flow(%[[T02]], DMA : 4, %[[T04]], DMA : 0)
    # CHECK: AIE.flow(%[[T02]], DMA : 5, %[[T04]], DMA : 1)
    # CHECK: AIE.flow(%[[T01]], DMA : 0, %[[T02]], DMA : 0)
    # CHECK: AIE.flow(%[[T01]], DMA : 1, %[[T02]], DMA : 1)
    print(mlir_module)


# CHECK-LABEL: TEST: test_memtile_routing_constraints
@run
def test_memtile_routing_constraints():
    with open(
        Path(THIS_FILE).parent.parent
        / "create-flows"
        / "memtile_routing_constraints.mlir"
    ) as f:
        mlir_module = Module.parse(f.read())
    r = Router(timeout=TIMEOUT)
    pass_ = create_python_router_pass(r)
    pm = PassManager()
    pass_manager_add_owned_pass(pm, pass_)
    pm.add("aie-find-flows")

    device = mlir_module.body.operations[0]
    pm.run(device.operation)

    # %[[T24:.*]] = AIE.tile(2, 4)
    # %[[T23:.*]] = AIE.tile(2, 3)
    # %[[T22:.*]] = AIE.tile(2, 2)
    # %[[T21:.*]] = AIE.tile(2, 1)
    # %[[T20:.*]] = AIE.tile(2, 0)
    # AIE.flow(%[[T22]], DMA : 0, %[[T21]], DMA : 0)
    # AIE.flow(%[[T23]], DMA : 0, %[[T20]], DMA : 0)
    print(mlir_module)


# CHECK-LABEL: TEST: test_mmult
@run
def test_mmult():
    with open(Path(THIS_FILE).parent.parent / "create-flows" / "mmult.mlir") as f:
        mlir_module = Module.parse(f.read())
    r = Router(timeout=TIMEOUT)
    pass_ = create_python_router_pass(r)
    pm = PassManager()
    pass_manager_add_owned_pass(pm, pass_)
    pm.add("aie-find-flows")

    device = mlir_module.body.operations[0]
    pm.run(device.operation)

    # CHEC: %[[T1:.*]] = AIE.tile(7, 0)
    # CHEC: %[[T3:.*]] = AIE.tile(8, 3)
    # CHEC: %[[T15:.*]] = AIE.tile(6, 0)
    # CHEC: %[[T17:.*]] = AIE.tile(7, 3)
    # CHEC: %[[T29:.*]] = AIE.tile(3, 0)
    # CHEC: %[[T31:.*]] = AIE.tile(8, 2)
    # CHEC: %[[T43:.*]] = AIE.tile(2, 0)
    # CHEC: %[[T45:.*]] = AIE.tile(7, 2)

    # CHEC: AIE.flow(%[[T1]], DMA : 0, %[[T3]], DMA : 0)
    # CHEC: AIE.flow(%[[T1]], DMA : 1, %[[T3]], DMA : 1)
    # CHEC: AIE.flow(%[[T3]], DMA : 0, %[[T29]], DMA : 1)
    # CHEC: AIE.flow(%[[T15]], DMA : 0, %[[T17]], DMA : 0)
    # CHEC: AIE.flow(%[[T15]], DMA : 1, %[[T17]], DMA : 1)
    # CHEC: AIE.flow(%[[T17]], DMA : 0, %[[T29]], DMA : 0)
    # CHEC: AIE.flow(%[[T29]], DMA : 0, %[[T31]], DMA : 0)
    # CHEC: AIE.flow(%[[T29]], DMA : 1, %[[T31]], DMA : 1)
    # CHEC: AIE.flow(%[[T31]], DMA : 0, %[[T43]], DMA : 1)
    # CHEC: AIE.flow(%[[T43]], DMA : 0, %[[T45]], DMA : 0)
    # CHEC: AIE.flow(%[[T43]], DMA : 1, %[[T45]], DMA : 1)
    # CHEC: AIE.flow(%[[T45]], DMA : 0, %[[T43]], DMA : 0)
    print(mlir_module)


# CHECK-LABEL: TEST: test_more_flows_shim
@run
def test_more_flows_shim():
    with open(
        Path(THIS_FILE).parent.parent / "create-flows" / "more_flows_shim.mlir"
    ) as f:
        for mlir_module in f.read().split("// -----"):
            mlir_module = Module.parse(mlir_module)
            r = Router(timeout=TIMEOUT)
            pass_ = create_python_router_pass(r)
            pm = PassManager()
            pass_manager_add_owned_pass(pm, pass_)
            pm.add("aie-find-flows")

            device = mlir_module.body.operations[0]
            pm.run(device.operation)

            print(mlir_module)

    # CHECK-LABEL: test70
    # CHECK: %[[T70:.*]] = AIE.tile(7, 0)
    # CHECK: %[[T71:.*]] = AIE.tile(7, 1)
    # CHECK:  %[[SB70:.*]] = AIE.switchbox(%[[T70]])  {
    # CHECK:    AIE.connect<North : 0, South : 2>
    # CHECK:  }
    # CHECK:  %[[SH70:.*]] = AIE.shimmux(%[[T70]])  {
    # CHECK:    AIE.connect<North : 2, PLIO : 2>
    # CHECK:  }
    # CHECK:  %[[SB71:.*]] = AIE.switchbox(%[[T71]])  {
    # CHECK:    AIE.connect<North : 0, South : 0>
    # CHECK:  }

    # CHECK-LABEL: test60
    # CHECK: %[[T60:.*]] = AIE.tile(6, 0)
    # CHECK: %[[T61:.*]] = AIE.tile(6, 1)
    # CHECK:  %[[SB60:.*]] = AIE.switchbox(%[[T60]])  {
    # CHECK:    AIE.connect<South : 6, North : 0>
    # CHECK:  }
    # CHECK:  %[[SH60:.*]] = AIE.shimmux(%[[T60]])  {
    # CHECK:    AIE.connect<PLIO : 6, North : 6>
    # CHECK:  }
    # CHECK:  %[[SB61:.*]] = AIE.switchbox(%[[T61]])  {
    # CHECK:    AIE.connect<South : 0, DMA : 1>
    # CHECK:  }

    # CHECK-LABEL: test40
    # CHECK: %[[T40:.*]] = AIE.tile(4, 0)
    # CHECK: %[[T41:.*]] = AIE.tile(4, 1)
    # CHECK:  %[[SB40:.*]] = AIE.switchbox(%[[T40]])  {
    # CHECK:    AIE.connect<North : 0, South : 3>
    # CHECK:    AIE.connect<South : 4, North : 0>
    # CHECK:  }
    # CHECK:  %[[SB41:.*]] = AIE.switchbox(%[[T41]])  {
    # CHECK:    AIE.connect<North : 0, South : 0>
    # CHECK:    AIE.connect<South : 0, North : 0>
    # CHECK:  }

    # CHECK-LABEL: test100
    # CHECK: %[[T100:.*]] = AIE.tile(10, 0)
    # CHECK: %[[T101:.*]] = AIE.tile(10, 1)
    # CHECK:  %[[SB100:.*]] = AIE.switchbox(%[[T100]])  {
    # CHECK:    AIE.connect<North : 0, South : 4>
    # CHECK:  }
    # CHECK:  %[[SH100:.*]] = AIE.shimmux(%[[T100]])  {
    # CHECK:    AIE.connect<North : 4, NOC : 2>
    # CHECK:  }
    # CHECK:  %[[SB101:.*]] = AIE.switchbox(%[[T101]])  {
    # CHECK:    AIE.connect<North : 0, South : 0>
    # CHECK:  }


# CHECK-LABEL: TEST: test_over_flows
@run
def test_over_flows():
    with open(Path(THIS_FILE).parent.parent / "create-flows" / "over_flows.mlir") as f:
        mlir_module = Module.parse(f.read())
    r = Router(timeout=TIMEOUT)
    pass_ = create_python_router_pass(r)
    pm = PassManager()
    pass_manager_add_owned_pass(pm, pass_)
    pm.add("aie-find-flows")

    device = mlir_module.body.operations[0]
    pm.run(device.operation)

    # CHECK: %[[T03:.*]] = AIE.tile(0, 3)
    # CHECK: %[[T02:.*]] = AIE.tile(0, 2)
    # CHECK: %[[T00:.*]] = AIE.tile(0, 0)
    # CHECK: %[[T13:.*]] = AIE.tile(1, 3)
    # CHECK: %[[T11:.*]] = AIE.tile(1, 1)
    # CHECK: %[[T10:.*]] = AIE.tile(1, 0)
    # CHECK: %[[T20:.*]] = AIE.tile(2, 0)
    # CHECK: %[[T30:.*]] = AIE.tile(3, 0)
    # CHECK: %[[T22:.*]] = AIE.tile(2, 2)
    # CHECK: %[[T31:.*]] = AIE.tile(3, 1)
    # CHECK: %[[T60:.*]] = AIE.tile(6, 0)
    # CHECK: %[[T70:.*]] = AIE.tile(7, 0)
    # CHECK: %[[T71:.*]] = AIE.tile(7, 1)
    # CHECK: %[[T72:.*]] = AIE.tile(7, 2)
    # CHECK: %[[T73:.*]] = AIE.tile(7, 3)
    # CHECK: %[[T80:.*]] = AIE.tile(8, 0)
    # CHECK: %[[T82:.*]] = AIE.tile(8, 2)
    # CHECK: %[[T83:.*]] = AIE.tile(8, 3)
    # CHECK: AIE.flow(%[[T71]], DMA : 0, %[[T20]], DMA : 0)
    # CHECK: AIE.flow(%[[T71]], DMA : 1, %[[T20]], DMA : 1)
    # CHECK: AIE.flow(%[[T72]], DMA : 0, %[[T60]], DMA : 0)
    # CHECK: AIE.flow(%[[T72]], DMA : 1, %[[T60]], DMA : 1)
    # CHECK: AIE.flow(%[[T73]], DMA : 0, %[[T70]], DMA : 0)
    # CHECK: AIE.flow(%[[T73]], DMA : 1, %[[T70]], DMA : 1)
    # CHECK: AIE.flow(%[[T83]], DMA : 0, %[[T30]], DMA : 0)
    # CHECK: AIE.flow(%[[T83]], DMA : 1, %[[T30]], DMA : 1)
    print(mlir_module)


# this test just tests that there's no error for multiple connections to a single target bundle/channel
# CHECK-LABEL: TEST: test_overlap
@run
def test_overlap():
    src = dedent(
        """\
        module @aie.herd_0 {
          AIE.device(xcvc1902) {
            %tile_3_0 = AIE.tile(3, 0)
            %tile_3_1 = AIE.tile(3, 1)
            %tile_6_1 = AIE.tile(6, 1)
            %tile_7_3 = AIE.tile(7, 3)
            %tile_8_2 = AIE.tile(8, 2)
            %tile_8_3 = AIE.tile(8, 3)
            %switchbox_3_0 = AIE.switchbox(%tile_3_0) {
              AIE.connect<South : 3, North : 0>
              AIE.connect<South : 7, North : 1>
              AIE.connect<North : 0, South : 2>
              AIE.connect<North : 1, South : 3>
            }
            AIE.flow(%tile_3_1, South : 0, %tile_8_2, DMA : 0)
            AIE.flow(%tile_3_1, South : 1, %tile_8_2, DMA : 1) 
            AIE.flow(%tile_6_1, South : 0, %tile_7_3, DMA : 0)
            AIE.flow(%tile_6_1, South : 1, %tile_7_3, DMA : 1)
          }
        }
    """
    )
    mlir_module = Module.parse(src)
    r = Router(timeout=TIMEOUT)
    pass_ = create_python_router_pass(r)
    pm = PassManager()
    pass_manager_add_owned_pass(pm, pass_)
    pm.add("aie-find-flows")

    device = mlir_module.body.operations[0]
    pm.run(device.operation)


# CHECK-LABEL: TEST: test_routed_herd_3x1_mine_1
@run
def test_routed_herd_3x1_mine_1():
    src = dedent(
        """\
        module {
          AIE.device(xcvc1902) {
            %tile_0_0 = AIE.tile(0, 0)
            %tile_0_1 = AIE.tile(0, 1)
            %tile_0_2 = AIE.tile(0, 2)
            %tile_0_3 = AIE.tile(0, 3)
            %tile_0_4 = AIE.tile(0, 4)
            %tile_1_0 = AIE.tile(1, 0)
            %tile_1_1 = AIE.tile(1, 1)
            %tile_1_2 = AIE.tile(1, 2)
            %tile_1_3 = AIE.tile(1, 3)
            %tile_1_4 = AIE.tile(1, 4)
            %tile_2_0 = AIE.tile(2, 0)
            %tile_2_1 = AIE.tile(2, 1)
            %tile_2_2 = AIE.tile(2, 2)
            %tile_2_3 = AIE.tile(2, 3)
            %tile_2_4 = AIE.tile(2, 4)
            %tile_3_0 = AIE.tile(3, 0)
            %tile_3_1 = AIE.tile(3, 1)
            %tile_3_2 = AIE.tile(3, 2)
            %tile_3_3 = AIE.tile(3, 3)
            %tile_3_4 = AIE.tile(3, 4)
            %tile_4_0 = AIE.tile(4, 0)
            %tile_4_1 = AIE.tile(4, 1)
            %tile_4_2 = AIE.tile(4, 2)
            %tile_4_3 = AIE.tile(4, 3)
            %tile_4_4 = AIE.tile(4, 4)
            %tile_5_0 = AIE.tile(5, 0)
            %tile_5_1 = AIE.tile(5, 1)
            %tile_5_2 = AIE.tile(5, 2)
            %tile_5_3 = AIE.tile(5, 3)
            %tile_5_4 = AIE.tile(5, 4)
            %tile_6_0 = AIE.tile(6, 0)
            %tile_6_1 = AIE.tile(6, 1)
            %tile_6_2 = AIE.tile(6, 2)
            %tile_6_3 = AIE.tile(6, 3)
            %tile_6_4 = AIE.tile(6, 4)
            %tile_7_0 = AIE.tile(7, 0)
            %tile_7_1 = AIE.tile(7, 1)
            %tile_7_2 = AIE.tile(7, 2)
            %tile_7_3 = AIE.tile(7, 3)
            %tile_7_4 = AIE.tile(7, 4)
            %tile_8_0 = AIE.tile(8, 0)
            %tile_8_1 = AIE.tile(8, 1)
            %tile_8_2 = AIE.tile(8, 2)
            %tile_8_3 = AIE.tile(8, 3)
            %tile_8_4 = AIE.tile(8, 4)
            %tile_9_0 = AIE.tile(9, 0)
            %tile_9_1 = AIE.tile(9, 1)
            %tile_9_2 = AIE.tile(9, 2)
            %tile_9_3 = AIE.tile(9, 3)
            %tile_9_4 = AIE.tile(9, 4)
            %tile_10_0 = AIE.tile(10, 0)
            %tile_10_1 = AIE.tile(10, 1)
            %tile_10_2 = AIE.tile(10, 2)
            %tile_10_3 = AIE.tile(10, 3)
            %tile_10_4 = AIE.tile(10, 4)
            %tile_11_0 = AIE.tile(11, 0)
            %tile_11_1 = AIE.tile(11, 1)
            %tile_11_2 = AIE.tile(11, 2)
            %tile_11_3 = AIE.tile(11, 3)
            %tile_11_4 = AIE.tile(11, 4)
            %tile_12_1 = AIE.tile(12, 1)
            %tile_12_2 = AIE.tile(12, 2)
            %tile_12_3 = AIE.tile(12, 3)
            %tile_12_4 = AIE.tile(12, 4)
            %tile_18_0 = AIE.tile(18, 0)
            %tile_19_0 = AIE.tile(19, 0)
            
            %switchbox_0_1 = AIE.switchbox(%tile_0_1) {
              AIE.connect<South : 0, North : 0>
            }
            %switchbox_0_2 = AIE.switchbox(%tile_0_2) {
              AIE.connect<South : 0, North : 0>
            }
            %switchbox_0_3 = AIE.switchbox(%tile_0_3) {
              AIE.connect<South : 0, DMA : 0>
              AIE.connect<East : 0, DMA : 1>
            }
            %switchbox_1_1 = AIE.switchbox(%tile_1_1) {
              AIE.connect<South : 0, North : 0>
            }
            %switchbox_1_2 = AIE.switchbox(%tile_1_2) {
              AIE.connect<South : 0, North : 0>
            }
            %switchbox_1_3 = AIE.switchbox(%tile_1_3) {
              AIE.connect<South : 0, West : 0>
            }
            %switchbox_1_4 = AIE.switchbox(%tile_1_4) {
              AIE.connect<East : 0, DMA : 0>
            }
            %switchbox_2_1 = AIE.switchbox(%tile_2_1) {
              AIE.connect<South : 0, North : 0>
            }
            %switchbox_2_2 = AIE.switchbox(%tile_2_2) {
              AIE.connect<South : 0, North : 0>
            }
            %switchbox_2_3 = AIE.switchbox(%tile_2_3) {
              AIE.connect<South : 0, North : 0>
            }
            %switchbox_2_4 = AIE.switchbox(%tile_2_4) {
              AIE.connect<South : 0, West : 0>
            }
            %switchbox_3_1 = AIE.switchbox(%tile_3_1) {
              AIE.connect<South : 0, North : 0>
            }
            %switchbox_3_2 = AIE.switchbox(%tile_3_2) {
              AIE.connect<South : 0, North : 0>
            }
            %switchbox_3_3 = AIE.switchbox(%tile_3_3) {
              AIE.connect<South : 0, DMA : 0>
            }
            %switchbox_3_4 = AIE.switchbox(%tile_3_4) {
            }
            %switchbox_4_1 = AIE.switchbox(%tile_4_1) {
              AIE.connect<South : 0, North : 0>
            }
            %switchbox_4_2 = AIE.switchbox(%tile_4_2) {
              AIE.connect<South : 0, DMA : 0>
            }
            %switchbox_5_1 = AIE.switchbox(%tile_5_1) {
              AIE.connect<South : 0, North : 0>
            }
            %switchbox_5_2 = AIE.switchbox(%tile_5_2) {
              AIE.connect<South : 0, North : 0>
            }
            %switchbox_5_3 = AIE.switchbox(%tile_5_3) {
              AIE.connect<South : 0, DMA : 0>
            }
            %switchbox_6_1 = AIE.switchbox(%tile_6_1) {
              AIE.connect<South : 0, North : 0>
              AIE.connect<South : 1, North : 1>
            }
            %switchbox_6_2 = AIE.switchbox(%tile_6_2) {
              AIE.connect<South : 0, North : 0>
              AIE.connect<South : 1, North : 1>
            }
            %switchbox_6_3 = AIE.switchbox(%tile_6_3) {
              AIE.connect<South : 0, DMA : 0>
              AIE.connect<South : 1, DMA : 1>
            }
            %switchbox_7_1 = AIE.switchbox(%tile_7_1) {
              AIE.connect<South : 0, North : 0>
              AIE.connect<South : 1, North : 1>
            }
            %switchbox_7_2 = AIE.switchbox(%tile_7_2) {
              AIE.connect<South : 0, North : 0>
              AIE.connect<South : 1, North : 1>
            }
            %switchbox_7_3 = AIE.switchbox(%tile_7_3) {
              AIE.connect<South : 0, North : 0>
              AIE.connect<South : 1, North : 1>
            }
            %switchbox_7_4 = AIE.switchbox(%tile_7_4) {
              AIE.connect<South : 0, DMA : 0>
              AIE.connect<South : 1, DMA : 1>
            }
            %switchbox_9_1 = AIE.switchbox(%tile_9_1) {
              AIE.connect<South : 0, North : 0>
            }
            %switchbox_9_2 = AIE.switchbox(%tile_9_2) {
              AIE.connect<South : 0, DMA : 0>
            }
            %switchbox_10_1 = AIE.switchbox(%tile_10_1) {
              AIE.connect<South : 0, North : 0>
            }
            %switchbox_10_2 = AIE.switchbox(%tile_10_2) {
              AIE.connect<South : 0, DMA : 0>
            }
            %switchbox_11_1 = AIE.switchbox(%tile_11_1) {
              AIE.connect<South : 0, North : 0>
              AIE.connect<South : 1, North : 1>
            }
            %switchbox_11_2 = AIE.switchbox(%tile_11_2) {
              AIE.connect<South : 0, North : 0>
              AIE.connect<South : 1, North : 1>
            }
            %switchbox_11_3 = AIE.switchbox(%tile_11_3) {
              AIE.connect<South : 0, DMA : 0>
              AIE.connect<South : 1, DMA : 1>
            }
            AIE.flow(%tile_6_0, DMA : 1, %tile_4_0, North : 0)
            AIE.flow(%tile_7_0, DMA : 0, %tile_1_0, North : 0)
            AIE.flow(%tile_7_0, DMA : 1, %tile_5_0, North : 0)
            AIE.flow(%tile_19_0, DMA : 0, %tile_7_0, North : 0)
          }
        }
        """
    )

    mlir_module = Module.parse(src)
    r = Router(timeout=TIMEOUT)
    pass_ = create_python_router_pass(r)
    pm = PassManager()
    pass_manager_add_owned_pass(pm, pass_)
    pm.add("aie-find-flows")

    device = mlir_module.body.operations[0]
    pm.run(device.operation)

    # CHECK: AIE.flow(%tile_6_0, DMA : 1, %tile_4_2, DMA : 0)
    # CHECK: AIE.flow(%tile_7_0, DMA : 0, %tile_0_3, DMA : 1)
    # AIE.flow(%tile_7_0, DMA : 0, %tile_7_4, DMA : 1)
    # CHECK: AIE.flow(%tile_7_0, DMA : 1, %tile_5_3, DMA : 0)
    # CHECK: AIE.flow(%tile_19_0, DMA : 0, %tile_7_4, DMA : 0)
    print(mlir_module)


# CHECK-LABEL: TEST: test_routed_herd_3x1_mine_2
@run
def test_routed_herd_3x1_mine_2():
    src = dedent(
        """\
        module {
          AIE.device(xcvc1902) {
            %tile_0_0 = AIE.tile(0, 0)
            %tile_1_0 = AIE.tile(1, 0)
            %tile_2_0 = AIE.tile(2, 0)
            %tile_3_0 = AIE.tile(3, 0)
            %tile_4_0 = AIE.tile(4, 0)
            %tile_5_0 = AIE.tile(5, 0)
            %tile_6_0 = AIE.tile(6, 0)
            %tile_7_0 = AIE.tile(7, 0)
            %tile_8_0 = AIE.tile(8, 0)
            %tile_9_0 = AIE.tile(9, 0)
            %tile_10_0 = AIE.tile(10, 0)
            %tile_11_0 = AIE.tile(11, 0)
            %tile_18_0 = AIE.tile(18, 0)
            %tile_19_0 = AIE.tile(19, 0)
            %tile_0_1 = AIE.tile(0, 1)
            %tile_0_2 = AIE.tile(0, 2)
            %tile_0_3 = AIE.tile(0, 3)
            %tile_0_4 = AIE.tile(0, 4)
            %tile_1_1 = AIE.tile(1, 1)
            %tile_1_2 = AIE.tile(1, 2)
            %tile_1_3 = AIE.tile(1, 3)
            %tile_1_4 = AIE.tile(1, 4)
            %tile_2_1 = AIE.tile(2, 1)
            %tile_2_2 = AIE.tile(2, 2)
            %tile_2_3 = AIE.tile(2, 3)
            %tile_2_4 = AIE.tile(2, 4)
            %tile_3_1 = AIE.tile(3, 1)
            %tile_3_2 = AIE.tile(3, 2)
            %tile_3_3 = AIE.tile(3, 3)
            %tile_3_4 = AIE.tile(3, 4)
            %tile_4_1 = AIE.tile(4, 1)
            %tile_4_2 = AIE.tile(4, 2)
            %tile_4_3 = AIE.tile(4, 3)
            %tile_4_4 = AIE.tile(4, 4)
            %tile_5_1 = AIE.tile(5, 1)
            %tile_5_2 = AIE.tile(5, 2)
            %tile_5_3 = AIE.tile(5, 3)
            %tile_5_4 = AIE.tile(5, 4)
            %tile_6_1 = AIE.tile(6, 1)
            %tile_6_2 = AIE.tile(6, 2)
            %tile_6_3 = AIE.tile(6, 3)
            %tile_6_4 = AIE.tile(6, 4)
            %tile_7_1 = AIE.tile(7, 1)
            %tile_7_2 = AIE.tile(7, 2)
            %tile_7_3 = AIE.tile(7, 3)
            %tile_7_4 = AIE.tile(7, 4)
            %tile_8_1 = AIE.tile(8, 1)
            %tile_8_2 = AIE.tile(8, 2)
            %tile_8_3 = AIE.tile(8, 3)
            %tile_8_4 = AIE.tile(8, 4)
            %tile_9_1 = AIE.tile(9, 1)
            %tile_9_2 = AIE.tile(9, 2)
            %tile_9_3 = AIE.tile(9, 3)
            %tile_9_4 = AIE.tile(9, 4)
            %tile_10_1 = AIE.tile(10, 1)
            %tile_10_2 = AIE.tile(10, 2)
            %tile_10_3 = AIE.tile(10, 3)
            %tile_10_4 = AIE.tile(10, 4)
            %tile_11_1 = AIE.tile(11, 1)
            %tile_11_2 = AIE.tile(11, 2)
            %tile_11_3 = AIE.tile(11, 3)
            %tile_11_4 = AIE.tile(11, 4)
            %tile_12_1 = AIE.tile(12, 1)
            %tile_12_2 = AIE.tile(12, 2)
            %tile_12_3 = AIE.tile(12, 3)
            %tile_12_4 = AIE.tile(12, 4)
            %switchbox_0_1 = AIE.switchbox(%tile_0_1) {
              AIE.connect<South : 0, North : 0>
            }
            %switchbox_0_2 = AIE.switchbox(%tile_0_2) {
              AIE.connect<South : 0, North : 0>
            }
            %switchbox_0_3 = AIE.switchbox(%tile_0_3) {
              AIE.connect<South : 0, DMA : 0>
              AIE.connect<East : 0, DMA : 1>
            }
            %switchbox_0_4 = AIE.switchbox(%tile_0_4) {
            }
            %switchbox_1_1 = AIE.switchbox(%tile_1_1) {
              AIE.connect<South : 0, North : 0>
            }
            %switchbox_1_2 = AIE.switchbox(%tile_1_2) {
              AIE.connect<South : 0, North : 0>
            }
            %switchbox_1_3 = AIE.switchbox(%tile_1_3) {
              AIE.connect<South : 0, West : 0>
            }
            %switchbox_1_4 = AIE.switchbox(%tile_1_4) {
              AIE.connect<East : 0, DMA : 0>
            }
            %switchbox_2_1 = AIE.switchbox(%tile_2_1) {
              AIE.connect<South : 0, North : 0>
            }
            %switchbox_2_2 = AIE.switchbox(%tile_2_2) {
              AIE.connect<South : 0, North : 0>
            }
            %switchbox_2_3 = AIE.switchbox(%tile_2_3) {
              AIE.connect<South : 0, North : 0>
            }
            %switchbox_2_4 = AIE.switchbox(%tile_2_4) {
              AIE.connect<South : 0, West : 0>
            }
            %switchbox_3_1 = AIE.switchbox(%tile_3_1) {
              AIE.connect<South : 0, North : 0>
            }
            %switchbox_3_2 = AIE.switchbox(%tile_3_2) {
              AIE.connect<South : 0, North : 0>
            }
            %switchbox_3_3 = AIE.switchbox(%tile_3_3) {
              AIE.connect<South : 0, DMA : 0>
            }
            %switchbox_3_4 = AIE.switchbox(%tile_3_4) {
            }
            %switchbox_4_1 = AIE.switchbox(%tile_4_1) {
              AIE.connect<South : 0, North : 0>
            }
            %switchbox_4_2 = AIE.switchbox(%tile_4_2) {
              AIE.connect<South : 0, DMA : 0>
            }
            %switchbox_4_3 = AIE.switchbox(%tile_4_3) {
            }
            %switchbox_4_4 = AIE.switchbox(%tile_4_4) {
            }
            %switchbox_5_1 = AIE.switchbox(%tile_5_1) {
              AIE.connect<South : 0, North : 0>
            }
            %switchbox_5_2 = AIE.switchbox(%tile_5_2) {
              AIE.connect<South : 0, North : 0>
            }
            %switchbox_5_3 = AIE.switchbox(%tile_5_3) {
              AIE.connect<South : 0, DMA : 0>
            }
            %switchbox_5_4 = AIE.switchbox(%tile_5_4) {
            }
            %switchbox_6_1 = AIE.switchbox(%tile_6_1) {
              AIE.connect<South : 0, North : 0>
              AIE.connect<South : 1, North : 1>
            }
            %switchbox_6_2 = AIE.switchbox(%tile_6_2) {
              AIE.connect<South : 0, North : 0>
              AIE.connect<South : 1, North : 1>
            }
            %switchbox_6_3 = AIE.switchbox(%tile_6_3) {
              AIE.connect<South : 0, DMA : 0>
              AIE.connect<South : 1, DMA : 1>
            }
            %switchbox_6_4 = AIE.switchbox(%tile_6_4) {
            }
            %switchbox_7_1 = AIE.switchbox(%tile_7_1) {
              AIE.connect<South : 0, North : 0>
              AIE.connect<South : 1, North : 1>
            }
            %switchbox_7_2 = AIE.switchbox(%tile_7_2) {
              AIE.connect<South : 0, North : 0>
              AIE.connect<South : 1, North : 1>
            }
            %switchbox_7_3 = AIE.switchbox(%tile_7_3) {
              AIE.connect<South : 0, North : 0>
              AIE.connect<South : 1, North : 1>
            }
            %switchbox_7_4 = AIE.switchbox(%tile_7_4) {
              AIE.connect<South : 0, DMA : 0>
              AIE.connect<South : 1, DMA : 1>
            }
            %switchbox_8_1 = AIE.switchbox(%tile_8_1) {
            }
            %switchbox_8_2 = AIE.switchbox(%tile_8_2) {
            }
            %switchbox_8_3 = AIE.switchbox(%tile_8_3) {
            }
            %switchbox_8_4 = AIE.switchbox(%tile_8_4) {
            }
            %switchbox_9_1 = AIE.switchbox(%tile_9_1) {
              AIE.connect<South : 0, North : 0>
            }
            %switchbox_9_2 = AIE.switchbox(%tile_9_2) {
              AIE.connect<South : 0, DMA : 0>
            }
            %switchbox_9_3 = AIE.switchbox(%tile_9_3) {
            }
            %switchbox_9_4 = AIE.switchbox(%tile_9_4) {
            }
            %switchbox_10_1 = AIE.switchbox(%tile_10_1) {
              AIE.connect<South : 0, North : 0>
            }
            %switchbox_10_2 = AIE.switchbox(%tile_10_2) {
              AIE.connect<South : 0, DMA : 0>
            }
            %switchbox_10_3 = AIE.switchbox(%tile_10_3) {
            }
            %switchbox_10_4 = AIE.switchbox(%tile_10_4) {
            }
            %switchbox_11_1 = AIE.switchbox(%tile_11_1) {
              AIE.connect<South : 0, North : 0>
              AIE.connect<South : 1, North : 1>
            }
            %switchbox_11_2 = AIE.switchbox(%tile_11_2) {
              AIE.connect<South : 0, North : 0>
              AIE.connect<South : 1, North : 1>
            }
            %switchbox_11_3 = AIE.switchbox(%tile_11_3) {
              AIE.connect<South : 0, DMA : 0>
              AIE.connect<South : 1, DMA : 1>
            }
            %switchbox_11_4 = AIE.switchbox(%tile_11_4) {
            }
            // AIE.flow(%tile_2_0, DMA : 0, %tile_2_0, North : 0)
            AIE.flow(%tile_2_0, DMA : 1, %tile_6_0, North : 1)
            // AIE.flow(%tile_3_0, DMA : 0, %tile_3_0, North : 0)
            AIE.flow(%tile_3_0, DMA : 1, %tile_7_0, North : 1)
            AIE.flow(%tile_6_0, DMA : 0, %tile_0_0, North : 0)
            AIE.flow(%tile_6_0, DMA : 1, %tile_4_0, North : 0)
            AIE.flow(%tile_7_0, DMA : 0, %tile_1_0, North : 0)
            AIE.flow(%tile_7_0, DMA : 1, %tile_5_0, North : 0)
            // AIE.flow(%tile_10_0, DMA : 0, %tile_10_0, North : 0)
            // AIE.flow(%tile_11_0, DMA : 0, %tile_11_0, North : 0)
            AIE.flow(%tile_18_0, DMA : 0, %tile_6_0, North : 0)
            AIE.flow(%tile_18_0, DMA : 1, %tile_9_0, North : 0)
            AIE.flow(%tile_19_0, DMA : 0, %tile_7_0, North : 0)
            AIE.flow(%tile_19_0, DMA : 1, %tile_11_0, North : 1)
          }
        }
        """
    )

    mlir_module = Module.parse(src)
    r = Router(timeout=TIMEOUT)
    pass_ = create_python_router_pass(r)
    pm = PassManager()
    pass_manager_add_owned_pass(pm, pass_)
    pm.add("aie-find-flows")

    device = mlir_module.body.operations[0]
    pm.run(device.operation)
    # CHECK: AIE.flow(%tile_2_0, DMA : 1, %tile_6_3, DMA : 1)
    # CHECK: AIE.flow(%tile_3_0, DMA : 1, %tile_7_4, DMA : 1)
    # AIE.flow(%tile_3_0, DMA : 1, %tile_3_3, DMA : 0)
    # CHECK: AIE.flow(%tile_6_0, DMA : 0, %tile_0_3, DMA : 0)
    # CHECK: AIE.flow(%tile_6_0, DMA : 1, %tile_4_2, DMA : 0)
    # CHECK: AIE.flow(%tile_7_0, DMA : 0, %tile_0_3, DMA : 1)
    # CHECK: AIE.flow(%tile_7_0, DMA : 1, %tile_5_3, DMA : 0)
    # CHECK: AIE.flow(%tile_18_0, DMA : 0, %tile_6_3, DMA : 0)
    # CHECK: AIE.flow(%tile_18_0, DMA : 1, %tile_9_2, DMA : 0)
    # CHECK: AIE.flow(%tile_19_0, DMA : 0, %tile_7_4, DMA : 0)
    # CHECK: AIE.flow(%tile_19_0, DMA : 1, %tile_11_3, DMA : 1)
    print(mlir_module)


# CHECK-LABEL: TEST: test_routed_herd_3x2_mine_1
@run
def test_routed_herd_3x2_mine_1():
    src = dedent(
        """\
        module {
          AIE.device(xcvc1902) {
            %tile_0_0 = AIE.tile(0, 0)
            %tile_1_0 = AIE.tile(1, 0)
            %tile_2_0 = AIE.tile(2, 0)
            %tile_3_0 = AIE.tile(3, 0)
            %tile_4_0 = AIE.tile(4, 0)
            %tile_5_0 = AIE.tile(5, 0)
            %tile_6_0 = AIE.tile(6, 0)
            %tile_7_0 = AIE.tile(7, 0)
            %tile_8_0 = AIE.tile(8, 0)
            %tile_9_0 = AIE.tile(9, 0)
            %tile_10_0 = AIE.tile(10, 0)
            %tile_11_0 = AIE.tile(11, 0)
            %tile_18_0 = AIE.tile(18, 0)
            %tile_19_0 = AIE.tile(19, 0)
            %tile_0_1 = AIE.tile(0, 1)
            %tile_0_2 = AIE.tile(0, 2)
            %tile_0_3 = AIE.tile(0, 3)
            %tile_0_4 = AIE.tile(0, 4)
            %tile_0_5 = AIE.tile(0, 5)
            %tile_0_6 = AIE.tile(0, 6)
            %tile_0_7 = AIE.tile(0, 7)
            %tile_0_8 = AIE.tile(0, 8)
            %tile_1_1 = AIE.tile(1, 1)
            %tile_1_2 = AIE.tile(1, 2)
            %tile_1_3 = AIE.tile(1, 3)
            %tile_1_4 = AIE.tile(1, 4)
            %tile_1_5 = AIE.tile(1, 5)
            %tile_1_6 = AIE.tile(1, 6)
            %tile_1_7 = AIE.tile(1, 7)
            %tile_1_8 = AIE.tile(1, 8)
            %tile_2_1 = AIE.tile(2, 1)
            %tile_2_2 = AIE.tile(2, 2)
            %tile_2_3 = AIE.tile(2, 3)
            %tile_2_4 = AIE.tile(2, 4)
            %tile_2_5 = AIE.tile(2, 5)
            %tile_2_6 = AIE.tile(2, 6)
            %tile_2_7 = AIE.tile(2, 7)
            %tile_2_8 = AIE.tile(2, 8)
            %tile_3_1 = AIE.tile(3, 1)
            %tile_3_2 = AIE.tile(3, 2)
            %tile_3_3 = AIE.tile(3, 3)
            %tile_3_4 = AIE.tile(3, 4)
            %tile_3_5 = AIE.tile(3, 5)
            %tile_3_6 = AIE.tile(3, 6)
            %tile_3_7 = AIE.tile(3, 7)
            %tile_3_8 = AIE.tile(3, 8)
            %tile_4_1 = AIE.tile(4, 1)
            %tile_4_2 = AIE.tile(4, 2)
            %tile_4_3 = AIE.tile(4, 3)
            %tile_4_4 = AIE.tile(4, 4)
            %tile_4_5 = AIE.tile(4, 5)
            %tile_4_6 = AIE.tile(4, 6)
            %tile_4_7 = AIE.tile(4, 7)
            %tile_4_8 = AIE.tile(4, 8)
            %tile_5_1 = AIE.tile(5, 1)
            %tile_5_2 = AIE.tile(5, 2)
            %tile_5_3 = AIE.tile(5, 3)
            %tile_5_4 = AIE.tile(5, 4)
            %tile_5_5 = AIE.tile(5, 5)
            %tile_5_6 = AIE.tile(5, 6)
            %tile_5_7 = AIE.tile(5, 7)
            %tile_5_8 = AIE.tile(5, 8)
            %tile_6_1 = AIE.tile(6, 1)
            %tile_6_2 = AIE.tile(6, 2)
            %tile_6_3 = AIE.tile(6, 3)
            %tile_6_4 = AIE.tile(6, 4)
            %tile_6_5 = AIE.tile(6, 5)
            %tile_6_6 = AIE.tile(6, 6)
            %tile_6_7 = AIE.tile(6, 7)
            %tile_6_8 = AIE.tile(6, 8)
            %tile_7_1 = AIE.tile(7, 1)
            %tile_7_2 = AIE.tile(7, 2)
            %tile_7_3 = AIE.tile(7, 3)
            %tile_7_4 = AIE.tile(7, 4)
            %tile_7_5 = AIE.tile(7, 5)
            %tile_7_6 = AIE.tile(7, 6)
            %tile_7_7 = AIE.tile(7, 7)
            %tile_7_8 = AIE.tile(7, 8)
            %tile_8_1 = AIE.tile(8, 1)
            %tile_8_2 = AIE.tile(8, 2)
            %tile_8_3 = AIE.tile(8, 3)
            %tile_8_4 = AIE.tile(8, 4)
            %tile_8_5 = AIE.tile(8, 5)
            %tile_8_6 = AIE.tile(8, 6)
            %tile_8_7 = AIE.tile(8, 7)
            %tile_8_8 = AIE.tile(8, 8)
            %tile_9_1 = AIE.tile(9, 1)
            %tile_9_2 = AIE.tile(9, 2)
            %tile_9_3 = AIE.tile(9, 3)
            %tile_9_4 = AIE.tile(9, 4)
            %tile_9_5 = AIE.tile(9, 5)
            %tile_9_6 = AIE.tile(9, 6)
            %tile_9_7 = AIE.tile(9, 7)
            %tile_9_8 = AIE.tile(9, 8)
            %tile_10_1 = AIE.tile(10, 1)
            %tile_10_2 = AIE.tile(10, 2)
            %tile_10_3 = AIE.tile(10, 3)
            %tile_10_4 = AIE.tile(10, 4)
            %tile_10_5 = AIE.tile(10, 5)
            %tile_10_6 = AIE.tile(10, 6)
            %tile_10_7 = AIE.tile(10, 7)
            %tile_10_8 = AIE.tile(10, 8)
            %tile_11_1 = AIE.tile(11, 1)
            %tile_11_2 = AIE.tile(11, 2)
            %tile_11_3 = AIE.tile(11, 3)
            %tile_11_4 = AIE.tile(11, 4)
            %tile_11_5 = AIE.tile(11, 5)
            %tile_11_6 = AIE.tile(11, 6)
            %tile_11_7 = AIE.tile(11, 7)
            %tile_11_8 = AIE.tile(11, 8)
            %tile_12_1 = AIE.tile(12, 1)
            %tile_12_2 = AIE.tile(12, 2)
            %tile_12_3 = AIE.tile(12, 3)
            %tile_12_4 = AIE.tile(12, 4)
            %tile_12_5 = AIE.tile(12, 5)
            %tile_12_6 = AIE.tile(12, 6)
            %tile_12_7 = AIE.tile(12, 7)
            %tile_12_8 = AIE.tile(12, 8)
            %tile_13_0 = AIE.tile(13, 0)
            %tile_13_1 = AIE.tile(13, 1)
            %tile_13_2 = AIE.tile(13, 2)
            %tile_13_3 = AIE.tile(13, 3)
            %tile_13_4 = AIE.tile(13, 4)
            %tile_13_5 = AIE.tile(13, 5)
            %tile_13_6 = AIE.tile(13, 6)
            %tile_13_7 = AIE.tile(13, 7)
            %tile_13_8 = AIE.tile(13, 8)
            %tile_14_1 = AIE.tile(14, 1)
            %tile_14_2 = AIE.tile(14, 2)
            %tile_14_3 = AIE.tile(14, 3)
            %tile_14_4 = AIE.tile(14, 4)
            %tile_14_5 = AIE.tile(14, 5)
            %tile_14_6 = AIE.tile(14, 6)
            %tile_14_7 = AIE.tile(14, 7)
            %tile_14_8 = AIE.tile(14, 8)
            %switchbox_0_1 = AIE.switchbox(%tile_0_1) {
            }
            %switchbox_0_2 = AIE.switchbox(%tile_0_2) {
            }
            %switchbox_0_3 = AIE.switchbox(%tile_0_3) {
            }
            %switchbox_0_4 = AIE.switchbox(%tile_0_4) {
            }
            %switchbox_1_1 = AIE.switchbox(%tile_1_1) {
            }
            %switchbox_1_2 = AIE.switchbox(%tile_1_2) {
            }
            %switchbox_1_3 = AIE.switchbox(%tile_1_3) {
            }
            %switchbox_1_4 = AIE.switchbox(%tile_1_4) {
            }
            %switchbox_2_1 = AIE.switchbox(%tile_2_1) {
            }
            %switchbox_2_2 = AIE.switchbox(%tile_2_2) {
            }
            %switchbox_2_3 = AIE.switchbox(%tile_2_3) {
            }
            %switchbox_2_4 = AIE.switchbox(%tile_2_4) {
              AIE.connect<East : 0, North : 0>
            }
            %switchbox_2_5 = AIE.switchbox(%tile_2_5) {
              AIE.connect<South : 0, Core : 0>
              AIE.connect<DMA : 0, East : 0>
            }
            %switchbox_3_1 = AIE.switchbox(%tile_3_1) {
              AIE.connect<South : 0, DMA : 0>
              AIE.connect<Core : 0, North : 0>
            }
            %switchbox_3_2 = AIE.switchbox(%tile_3_2) {
              AIE.connect<South : 0, North : 0>
            }
            %switchbox_3_3 = AIE.switchbox(%tile_3_3) {
              AIE.connect<South : 0, North : 0>
            }
            %switchbox_3_4 = AIE.switchbox(%tile_3_4) {
              AIE.connect<South : 0, West : 0>
            }
            %switchbox_3_5 = AIE.switchbox(%tile_3_5) {
              AIE.connect<West : 0, East : 0>
            }
            %switchbox_4_1 = AIE.switchbox(%tile_4_1) {
            }
            %switchbox_4_2 = AIE.switchbox(%tile_4_2) {
            }
            %switchbox_4_3 = AIE.switchbox(%tile_4_3) {
            }
            %switchbox_4_4 = AIE.switchbox(%tile_4_4) {
            }
            %switchbox_5_1 = AIE.switchbox(%tile_5_1) {
            }
            %switchbox_5_2 = AIE.switchbox(%tile_5_2) {
            }
            %switchbox_5_3 = AIE.switchbox(%tile_5_3) {
            }
            %switchbox_5_4 = AIE.switchbox(%tile_5_4) {
            }
            %switchbox_5_5 = AIE.switchbox(%tile_5_5) {
            }
            %switchbox_5_6 = AIE.switchbox(%tile_5_6) {
              AIE.connect<East : 0, West : 0>
            }
            %switchbox_6_1 = AIE.switchbox(%tile_6_1) {
            }
            %switchbox_6_2 = AIE.switchbox(%tile_6_2) {
            }
            %switchbox_6_3 = AIE.switchbox(%tile_6_3) {
            }
            %switchbox_6_4 = AIE.switchbox(%tile_6_4) {
            }
            %switchbox_6_5 = AIE.switchbox(%tile_6_5) {
            }
            %switchbox_6_6 = AIE.switchbox(%tile_6_6) {
              AIE.connect<East : 0, Core : 0>
              AIE.connect<DMA : 0, West : 0>
            }
            %switchbox_7_1 = AIE.switchbox(%tile_7_1) {
            }
            %switchbox_7_2 = AIE.switchbox(%tile_7_2) {
            }
            %switchbox_7_3 = AIE.switchbox(%tile_7_3) {
              AIE.connect<East : 0, DMA : 0>
              AIE.connect<Core : 0, North : 0>
            }
            %switchbox_7_4 = AIE.switchbox(%tile_7_4) {
              AIE.connect<South : 0, North : 0>
            }
            %switchbox_7_5 = AIE.switchbox(%tile_7_5) {
              AIE.connect<South : 0, North : 0>
            }
            %switchbox_7_6 = AIE.switchbox(%tile_7_6) {
              AIE.connect<South : 0, West : 0>
            }
            %switchbox_8_1 = AIE.switchbox(%tile_8_1) {
            }
            %switchbox_8_2 = AIE.switchbox(%tile_8_2) {
            }
            %switchbox_8_3 = AIE.switchbox(%tile_8_3) {
              AIE.connect<East : 0, West : 0>
            }
            %switchbox_8_4 = AIE.switchbox(%tile_8_4) {
            }
            %switchbox_9_1 = AIE.switchbox(%tile_9_1) {
            }
            %switchbox_9_2 = AIE.switchbox(%tile_9_2) {
            }
            %switchbox_9_3 = AIE.switchbox(%tile_9_3) {
            }
            %switchbox_9_4 = AIE.switchbox(%tile_9_4) {
            }
            %switchbox_10_1 = AIE.switchbox(%tile_10_1) {
            }
            %switchbox_10_2 = AIE.switchbox(%tile_10_2) {
            }
            %switchbox_10_3 = AIE.switchbox(%tile_10_3) {
            }
            %switchbox_10_4 = AIE.switchbox(%tile_10_4) {
            }
            %switchbox_11_1 = AIE.switchbox(%tile_11_1) {
            }
            %switchbox_11_2 = AIE.switchbox(%tile_11_2) {
            }
            %switchbox_11_3 = AIE.switchbox(%tile_11_3) {
            }
            %switchbox_11_4 = AIE.switchbox(%tile_11_4) {
            }
            %switchbox_12_1 = AIE.switchbox(%tile_12_1) {
            }
            %switchbox_12_2 = AIE.switchbox(%tile_12_2) {
            }
            %switchbox_12_3 = AIE.switchbox(%tile_12_3) {
            }
            %switchbox_12_4 = AIE.switchbox(%tile_12_4) {
            }
            %switchbox_12_5 = AIE.switchbox(%tile_12_5) {
              AIE.connect<East : 0, Core : 0>
              AIE.connect<DMA : 0, East : 0>
            }
            %switchbox_13_1 = AIE.switchbox(%tile_13_1) {
              AIE.connect<South : 0, North : 0>
            }
            %switchbox_13_2 = AIE.switchbox(%tile_13_2) {
              AIE.connect<South : 0, North : 0>
            }
            %switchbox_13_3 = AIE.switchbox(%tile_13_3) {
              AIE.connect<South : 0, DMA : 0>
              AIE.connect<Core : 0, North : 0>
            }
            %switchbox_13_4 = AIE.switchbox(%tile_13_4) {
              AIE.connect<South : 0, North : 0>
            }
            %switchbox_13_5 = AIE.switchbox(%tile_13_5) {
              AIE.connect<South : 0, West : 0>
              AIE.connect<West : 0, East : 0>
            }
            // AIE.flow(%tile_3_0, DMA : 0, %tile_3_0, North : 0)
            AIE.flow(%tile_4_5, West : 0, %tile_6_0, DMA : 0)
            AIE.flow(%tile_10_0, DMA : 0, %tile_9_3, West : 0)
            AIE.flow(%tile_4_6, East : 0, %tile_2_0, DMA : 0)
            AIE.flow(%tile_11_0, DMA : 0, %tile_13_0, North : 0)
            AIE.flow(%tile_14_5, West : 0, %tile_18_0, DMA : 0)
          }
        }

        """
    )

    mlir_module = Module.parse(src)
    r = Router(timeout=TIMEOUT)
    pass_ = create_python_router_pass(r)
    pm = PassManager()
    pass_manager_add_owned_pass(pm, pass_)
    pm.add("aie-find-flows")

    device = mlir_module.body.operations[0]
    pm.run(device.operation)
    # CHECK: AIE.flow(%tile_10_0, DMA : 0, %tile_7_3, DMA : 0)
    # CHECK: AIE.flow(%tile_11_0, DMA : 0, %tile_13_3, DMA : 0)
    # CHECK: AIE.flow(%tile_2_5, DMA : 0, %tile_6_0, DMA : 0)
    # CHECK: AIE.flow(%tile_3_1, Core : 0, %tile_2_5, Core : 0)
    # CHECK: AIE.flow(%tile_6_6, DMA : 0, %tile_2_0, DMA : 0)
    # CHECK: AIE.flow(%tile_7_3, Core : 0, %tile_6_6, Core : 0)
    # CHECK: AIE.flow(%tile_12_5, DMA : 0, %tile_18_0, DMA : 0)
    # CHECK: AIE.flow(%tile_13_3, Core : 0, %tile_12_5, Core : 0)
    print(mlir_module)


# CHECK-LABEL: TEST: test_simple
@run
def test_simple():
    with open(Path(THIS_FILE).parent.parent / "create-flows" / "simple.mlir") as f:
        mlir_module = Module.parse(f.read())
    r = Router(timeout=TIMEOUT)
    pass_ = create_python_router_pass(r)
    pm = PassManager()
    pass_manager_add_owned_pass(pm, pass_)
    pm.add("aie-find-flows")

    device = mlir_module.body.operations[0]
    pm.run(device.operation)

    # CHECK: %[[T01:.*]] = AIE.tile(0, 1)
    # CHECK: %[[T12:.*]] = AIE.tile(1, 2)
    # CHECK: AIE.flow(%[[T01]], DMA : 0, %[[T12]], Core : 1)
    print(mlir_module)


# CHECK-LABEL: TEST: test_simple2
@run
def test_simple2():
    with open(Path(THIS_FILE).parent.parent / "create-flows" / "simple2.mlir") as f:
        mlir_module = Module.parse(f.read())
    r = Router(timeout=TIMEOUT)
    pass_ = create_python_router_pass(r)
    pm = PassManager()
    pass_manager_add_owned_pass(pm, pass_)
    pm.add("aie-find-flows")

    device = mlir_module.body.operations[0]
    pm.run(device.operation)

    # CHECK: %[[T23:.*]] = AIE.tile(2, 3)
    # CHECK: %[[T32:.*]] = AIE.tile(3, 2)
    # CHECK: AIE.flow(%[[T23]], Core : 1, %[[T32]], DMA : 0)
    print(mlir_module)


# CHECK-LABEL: TEST: test_simple_flows2
@run
def test_simple_flows2():
    with open(
        Path(THIS_FILE).parent.parent / "create-flows" / "simple_flows2.mlir"
    ) as f:
        mlir_module = Module.parse(f.read())
    r = Router(timeout=TIMEOUT)
    pass_ = create_python_router_pass(r)
    pm = PassManager()
    pass_manager_add_owned_pass(pm, pass_)
    pm.add("aie-find-flows")

    device = mlir_module.body.operations[0]
    pm.run(device.operation)

    # CHECK: %[[T23:.*]] = AIE.tile(2, 3)
    # CHECK: %[[T22:.*]] = AIE.tile(2, 2)
    # CHECK: %[[T11:.*]] = AIE.tile(1, 1)
    # CHECK: AIE.flow(%[[T23]], Core : 0, %[[T22]], Core : 1)
    # CHECK: AIE.flow(%[[T22]], Core : 0, %[[T11]], Core : 0)
    print(mlir_module)


# CHECK-LABEL: TEST: test_simple_flows_shim
@run
def test_simple_flows_shim():
    with open(
        Path(THIS_FILE).parent.parent / "create-flows" / "simple_flows_shim.mlir"
    ) as f:
        for mlir_module in f.read().split("// -----"):
            mlir_module = Module.parse(mlir_module)
            r = Router(timeout=TIMEOUT)
            pass_ = create_python_router_pass(r)
            pm = PassManager()
            pass_manager_add_owned_pass(pm, pass_)
            pm.add("aie-find-flows")

            device = mlir_module.body.operations[0]
            pm.run(device.operation)

            print(mlir_module)

    # CHECK: %[[T21:.*]] = AIE.tile(2, 1)
    # CHECK: %[[T20:.*]] = AIE.tile(2, 0)
    # CHECK:  %{{.*}} = AIE.switchbox(%[[T20]])  {
    # CHECK:    AIE.connect<North : 0, South : 0>
    # CHECK:  }
    # CHECK:  %{{.*}} = AIE.switchbox(%[[T21]])  {
    # CHECK:    AIE.connect<North : 0, South : 0>
    # CHECK:  }

    # CHECK: %[[T20:.*]] = AIE.tile(2, 0)
    # CHECK: %[[T21:.*]] = AIE.tile(2, 1)
    # CHECK:  %{{.*}} = AIE.switchbox(%[[T20]])  {
    # CHECK:    AIE.connect<North : 0, South : 3>
    # CHECK:  }
    # CHECK:  %{{.*}} = AIE.shimmux(%[[T20]])  {
    # CHECK:    AIE.connect<North : 3, DMA : 1>
    # CHECK:  }
    # CHECK:  %{{.*}} = AIE.switchbox(%[[T21]])  {
    # CHECK:    AIE.connect<Core : 0, South : 0>
    # CHECK:  }

    # CHECK: %[[T20:.*]] = AIE.tile(2, 0)
    # CHECK: %[[T30:.*]] = AIE.tile(3, 0)
    # CHECK:  %{{.*}} = AIE.switchbox(%[[T20]])  {
    # CHECK:    AIE.connect<South : 3, East : 0>
    # CHECK:  }
    # CHECK:  %{{.*}} = AIE.shimmux(%[[T20]])  {
    # CHECK:    AIE.connect<DMA : 0, North : 3>
    # CHECK:  }
    # CHECK:  %{{.*}} = AIE.switchbox(%[[T30]])  {
    # CHECK:    AIE.connect<West : 0, South : 3>
    # CHECK:  }
    # CHECK:  %{{.*}} = AIE.shimmux(%[[T30]])  {
    # CHECK:    AIE.connect<North : 3, DMA : 1>
    # CHECK:  }
