# Copyright (C) 2022, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import os
from pathlib import Path

# RUN: %PYTHON %s | FileCheck %s
# REQUIRES: python_passes


from pprint import pprint

from aie._mlir_libs._aie_python_passes import (
    create_pathfinder_flows_with_python_pass,
    pass_manager_add_owned_pass,
)
from aie.dialects.aie import register_dialect
from aie.ir import Context, Location, Module
from aie.passmanager import PassManager
from aie.util import Router


def run(f):
    with Context() as ctx, Location.unknown():
        register_dialect(ctx)
        print("\nTEST:", f.__name__)
        try:
            f(ctx)
        except Exception as e:
            print(e)


THIS_FILE = __file__

# CHECK-LABEL: TEST: test_broadcast
@run
def test_broadcast(ctx):
    mlir_module = Module.parse(
        open(Path(THIS_FILE).parent.parent / "create-flows" / "broadcast.mlir").read()
    )
    r = Router()
    pass_ = create_pathfinder_flows_with_python_pass(r)
    pm = PassManager()
    pass_manager_add_owned_pass(pm, pass_)
    device = mlir_module.body.operations[0]
    pm.run(device.operation)

    # CHECK: PathEndPoint(Switchbox(2, 0): (DMA: 0))
    # OrderedDict([(Switchbox(2, 0), (DMA: 0) -> {(W: 0), (N: 0), (N: 1), (E: 0)}),
    #              (Switchbox(1, 0), (E: 0) -> {(N: 0)}),
    #              (Switchbox(1, 1), (S: 0) -> {(N: 0)}),
    #              (Switchbox(1, 2), (S: 0) -> {(N: 0)}),
    #              (Switchbox(1, 3), (S: 0) -> {(DMA: 0)}),
    #              (Switchbox(2, 1), (S: 0) -> {(N: 0), (E: 0)}),
    #              (Switchbox(3, 1), (W: 0) -> {(DMA: 0)}),
    #              (Switchbox(3, 0), (W: 0) -> {(E: 0)}),
    #              (Switchbox(4, 0), (W: 0) -> {(N: 0)}),
    #              (Switchbox(4, 1), (S: 0) -> {(E: 0)}),
    #              (Switchbox(5, 1), (W: 0) -> {(E: 0)}),
    #              (Switchbox(6, 1), (W: 0) -> {(E: 0)}),
    #              (Switchbox(7, 1), (W: 0) -> {(DMA: 0)}),
    #              (Switchbox(2, 2), (S: 0) -> {(E: 0)}),
    #              (Switchbox(3, 2), (W: 0) -> {(E: 0)}),
    #              (Switchbox(4, 2), (W: 0) -> {(E: 0)}),
    #              (Switchbox(5, 2), (W: 0) -> {(E: 0)}),
    #              (Switchbox(6, 2), (W: 0) -> {(E: 0)}),
    #              (Switchbox(7, 2), (W: 0) -> {(E: 0)}),
    #              (Switchbox(8, 2), (W: 0) -> {(DMA: 0)})])
    # PathEndPoint(Switchbox(6, 0): (DMA: 0))
    # OrderedDict([(Switchbox(6, 0), (DMA: 0) -> {(W: 0), (W: 1), (N: 0), (E: 0)}),
    #              (Switchbox(5, 0), (E: 0) -> {(W: 0), (N: 0)}),
    #              (Switchbox(4, 0), (E: 0) -> {(W: 0)}),
    #              (Switchbox(3, 0), (E: 0) -> {(N: 0)}),
    #              (Switchbox(3, 1), (S: 0) -> {(DMA: 1), (W: 0)}),
    #              (Switchbox(2, 1), (E: 0) -> {(W: 0)}),
    #              (Switchbox(1, 1), (E: 0) -> {(W: 0)}),
    #              (Switchbox(0, 1), (E: 0) -> {(N: 0)}),
    #              (Switchbox(0, 2), (S: 0) -> {(DMA: 1)}),
    #              (Switchbox(7, 0), (W: 0) -> {(N: 0)}),
    #              (Switchbox(7, 1), (S: 0) -> {(N: 0)}),
    #              (Switchbox(7, 2), (S: 0) -> {(N: 0)}),
    #              (Switchbox(7, 3), (S: 0) -> {(E: 0)}),
    #              (Switchbox(8, 3), (W: 0) -> {(DMA: 1)}),
    #              (Switchbox(5, 1), (S: 0) -> {(W: 0), (N: 0)}),
    #              (Switchbox(5, 2), (S: 0) -> {(W: 0)}),
    #              (Switchbox(4, 2), (E: 0) -> {(W: 0)}),
    #              (Switchbox(3, 2), (E: 0) -> {(W: 0)}),
    #              (Switchbox(2, 2), (E: 0) -> {(DMA: 1)}),
    #              (Switchbox(6, 1), (S: 0) -> {(W: 0)}),
    #              (Switchbox(4, 1), (E: 0) -> {(W: 0)})])
    for src, paths in r.routing_solution.items():
        print(src)
        pprint(paths)

    # CHECK: %switchbox_1_0 = AIE.switchbox(%tile_1_0) {
    # CHECK:   AIE.connect<East : 0, North : 0>
    # CHECK: }
    # CHECK: %switchbox_1_1 = AIE.switchbox(%tile_1_1) {
    # CHECK:   AIE.connect<South : 0, North : 0>
    # CHECK:   AIE.connect<East : 0, West : 0>
    # CHECK: }
    # CHECK: %tile_1_2 = AIE.tile(1, 2)
    # CHECK: %switchbox_1_2 = AIE.switchbox(%tile_1_2) {
    # CHECK:   AIE.connect<South : 0, North : 0>
    # CHECK: }
    # CHECK: %switchbox_1_3 = AIE.switchbox(%tile_1_3) {
    # CHECK:   AIE.connect<South : 0, DMA : 0>
    # CHECK: }
    # CHECK: %switchbox_2_0 = AIE.switchbox(%tile_2_0) {
    # CHECK:   AIE.connect<South : 3, West : 0>
    # CHECK:   AIE.connect<South : 3, North : 0>
    # CHECK:   AIE.connect<South : 3, North : 1>
    # CHECK:   AIE.connect<South : 3, East : 0>
    # CHECK: }
    # CHECK: %shimmux_2_0 = AIE.shimmux(%tile_2_0) {
    # CHECK:   AIE.connect<DMA : 0, North : 3>
    # CHECK: }
    # CHECK: %tile_2_1 = AIE.tile(2, 1)
    # CHECK: %switchbox_2_1 = AIE.switchbox(%tile_2_1) {
    # CHECK:   AIE.connect<South : 0, North : 0>
    # CHECK:   AIE.connect<South : 0, East : 0>
    # CHECK:   AIE.connect<East : 0, West : 0>
    # CHECK: }
    # CHECK: %switchbox_2_2 = AIE.switchbox(%tile_2_2) {
    # CHECK:   AIE.connect<South : 0, East : 0>
    # CHECK:   AIE.connect<East : 0, DMA : 1>
    # CHECK: }
    # CHECK: %switchbox_3_0 = AIE.switchbox(%tile_3_0) {
    # CHECK:   AIE.connect<West : 0, East : 0>
    # CHECK:   AIE.connect<East : 0, North : 0>
    # CHECK: }
    # CHECK: %switchbox_3_1 = AIE.switchbox(%tile_3_1) {
    # CHECK:   AIE.connect<West : 0, DMA : 0>
    # CHECK:   AIE.connect<South : 0, DMA : 1>
    # CHECK:   AIE.connect<South : 0, West : 0>
    # CHECK: }
    # CHECK: %tile_3_2 = AIE.tile(3, 2)
    # CHECK: %switchbox_3_2 = AIE.switchbox(%tile_3_2) {
    # CHECK:   AIE.connect<West : 0, East : 0>
    # CHECK:   AIE.connect<East : 0, West : 0>
    # CHECK: }
    # CHECK: %tile_4_0 = AIE.tile(4, 0)
    # CHECK: %switchbox_4_0 = AIE.switchbox(%tile_4_0) {
    # CHECK:   AIE.connect<West : 0, North : 0>
    # CHECK:   AIE.connect<East : 0, West : 0>
    # CHECK: }
    # CHECK: %tile_4_1 = AIE.tile(4, 1)
    # CHECK: %switchbox_4_1 = AIE.switchbox(%tile_4_1) {
    # CHECK:   AIE.connect<South : 0, East : 0>
    # CHECK:   AIE.connect<East : 0, West : 0>
    # CHECK: }
    # CHECK: %tile_4_2 = AIE.tile(4, 2)
    # CHECK: %switchbox_4_2 = AIE.switchbox(%tile_4_2) {
    # CHECK:   AIE.connect<West : 0, East : 0>
    # CHECK:   AIE.connect<East : 0, West : 0>
    # CHECK: }
    # CHECK: %tile_5_1 = AIE.tile(5, 1)
    # CHECK: %switchbox_5_1 = AIE.switchbox(%tile_5_1) {
    # CHECK:   AIE.connect<West : 0, East : 0>
    # CHECK:   AIE.connect<South : 0, West : 0>
    # CHECK:   AIE.connect<South : 0, North : 0>
    # CHECK: }
    # CHECK: %tile_5_2 = AIE.tile(5, 2)
    # CHECK: %switchbox_5_2 = AIE.switchbox(%tile_5_2) {
    # CHECK:   AIE.connect<West : 0, East : 0>
    # CHECK:   AIE.connect<South : 0, West : 0>
    # CHECK: }
    # CHECK: %tile_6_1 = AIE.tile(6, 1)
    # CHECK: %switchbox_6_1 = AIE.switchbox(%tile_6_1) {
    # CHECK:   AIE.connect<West : 0, East : 0>
    # CHECK:   AIE.connect<South : 0, West : 0>
    # CHECK: }
    # CHECK: %tile_6_2 = AIE.tile(6, 2)
    # CHECK: %switchbox_6_2 = AIE.switchbox(%tile_6_2) {
    # CHECK:   AIE.connect<West : 0, East : 0>
    # CHECK: }
    # CHECK: %switchbox_7_1 = AIE.switchbox(%tile_7_1) {
    # CHECK:   AIE.connect<West : 0, DMA : 0>
    # CHECK:   AIE.connect<South : 0, North : 0>
    # CHECK: }
    # CHECK: %switchbox_7_2 = AIE.switchbox(%tile_7_2) {
    # CHECK:   AIE.connect<West : 0, East : 0>
    # CHECK:   AIE.connect<South : 0, North : 0>
    # CHECK: }
    # CHECK: %switchbox_8_2 = AIE.switchbox(%tile_8_2) {
    # CHECK:   AIE.connect<West : 0, DMA : 0>
    # CHECK: }
    # CHECK: %tile_0_1 = AIE.tile(0, 1)
    # CHECK: %switchbox_0_1 = AIE.switchbox(%tile_0_1) {
    # CHECK:   AIE.connect<East : 0, North : 0>
    # CHECK: }
    # CHECK: %switchbox_0_2 = AIE.switchbox(%tile_0_2) {
    # CHECK:   AIE.connect<South : 0, DMA : 1>
    # CHECK: }
    # CHECK: %tile_5_0 = AIE.tile(5, 0)
    # CHECK: %switchbox_5_0 = AIE.switchbox(%tile_5_0) {
    # CHECK:   AIE.connect<East : 0, West : 0>
    # CHECK:   AIE.connect<East : 0, North : 0>
    # CHECK: }
    # CHECK: %switchbox_6_0 = AIE.switchbox(%tile_6_0) {
    # CHECK:   AIE.connect<South : 3, West : 0>
    # CHECK:   AIE.connect<South : 3, West : 1>
    # CHECK:   AIE.connect<South : 3, North : 0>
    # CHECK:   AIE.connect<South : 3, East : 0>
    # CHECK: }
    # CHECK: %shimmux_6_0 = AIE.shimmux(%tile_6_0) {
    # CHECK:   AIE.connect<DMA : 0, North : 3>
    # CHECK: }
    # CHECK: %switchbox_7_0 = AIE.switchbox(%tile_7_0) {
    # CHECK:   AIE.connect<West : 0, North : 0>
    # CHECK: }
    # CHECK: %switchbox_7_3 = AIE.switchbox(%tile_7_3) {
    # CHECK:   AIE.connect<South : 0, East : 0>
    # CHECK: }
    # CHECK: %switchbox_8_3 = AIE.switchbox(%tile_8_3) {
    # CHECK:   AIE.connect<West : 0, DMA : 1>
    # CHECK: }
    print(mlir_module)


# CHECK-LABEL: TEST: test_flow_test_1
@run
def test_flow_test_1(ctx):
    mlir_module = Module.parse(
        open(Path(THIS_FILE).parent.parent / "create-flows" / "flow_test_1.mlir").read()
    )
    r = Router()
    pass_ = create_pathfinder_flows_with_python_pass(r)
    pm = PassManager()
    pass_manager_add_owned_pass(pm, pass_)
    device = mlir_module.body.operations[0]
    pm.run(device.operation)

    for src, paths in r.routing_solution.items():
        print(src)
        pprint(paths)

    print(mlir_module)


# CHECK-LABEL: TEST: test_flow_test_2
@run
def test_flow_test_2(ctx):
    mlir_module = Module.parse(
        open(Path(THIS_FILE).parent.parent / "create-flows" / "flow_test_2.mlir").read()
    )
    r = Router()
    pass_ = create_pathfinder_flows_with_python_pass(r)
    pm = PassManager()
    pass_manager_add_owned_pass(pm, pass_)
    device = mlir_module.body.operations[0]
    pm.run(device.operation)

    for src, paths in r.routing_solution.items():
        print(src)
        pprint(paths)

    print(mlir_module)


# CHECK-LABEL: TEST: test_flow_test_3
@run
def test_flow_test_3(ctx):
    mlir_module = Module.parse(
        open(Path(THIS_FILE).parent.parent / "create-flows" / "flow_test_3.mlir").read()
    )
    r = Router()
    pass_ = create_pathfinder_flows_with_python_pass(r)
    pm = PassManager()
    pass_manager_add_owned_pass(pm, pass_)
    device = mlir_module.body.operations[0]
    pm.run(device.operation)

    for src, paths in r.routing_solution.items():
        print(src)
        pprint(paths)

    print(mlir_module)


# CHECK-LABEL: TEST: test_many_flows
@run
def test_many_flows(ctx):
    mlir_module = Module.parse(
        open(Path(THIS_FILE).parent.parent / "create-flows" / "many_flows.mlir").read()
    )
    r = Router()
    pass_ = create_pathfinder_flows_with_python_pass(r)
    pm = PassManager()
    pass_manager_add_owned_pass(pm, pass_)
    device = mlir_module.body.operations[0]
    pm.run(device.operation)

    for src, paths in r.routing_solution.items():
        print(src)
        pprint(paths)

    print(mlir_module)


# CHECK-LABEL: TEST: test_many_flows2
@run
def test_many_flows2(ctx):
    mlir_module = Module.parse(
        open(Path(THIS_FILE).parent.parent / "create-flows" / "many_flows2.mlir").read()
    )
    r = Router()
    pass_ = create_pathfinder_flows_with_python_pass(r)
    pm = PassManager()
    pass_manager_add_owned_pass(pm, pass_)
    device = mlir_module.body.operations[0]
    pm.run(device.operation)

    for src, paths in r.routing_solution.items():
        print(src)
        pprint(paths)

    print(mlir_module)


# CHECK-LABEL: TEST: test_memtile
@run
def test_memtile(ctx):
    mlir_module = Module.parse(
        open(Path(THIS_FILE).parent.parent / "create-flows" / "memtile.mlir").read()
    )
    r = Router()
    pass_ = create_pathfinder_flows_with_python_pass(r)
    pm = PassManager()
    pass_manager_add_owned_pass(pm, pass_)
    device = mlir_module.body.operations[0]
    pm.run(device.operation)

    for src, paths in r.routing_solution.items():
        print(src)
        pprint(paths)

    print(mlir_module)


# CHECK-LABEL: TEST: test_memtile_routing_constraints
@run
def test_memtile_routing_constraints(ctx):
    mlir_module = Module.parse(
        open(
            Path(THIS_FILE).parent.parent
            / "create-flows"
            / "memtile_routing_constraints.mlir"
        ).read()
    )
    r = Router()
    pass_ = create_pathfinder_flows_with_python_pass(r)
    pm = PassManager()
    pass_manager_add_owned_pass(pm, pass_)
    device = mlir_module.body.operations[0]
    pm.run(device.operation)

    for src, paths in r.routing_solution.items():
        print(src)
        pprint(paths)

    print(mlir_module)


# CHECK-LABEL: TEST: test_mmult
@run
def test_mmult(ctx):
    mlir_module = Module.parse(
        open(Path(THIS_FILE).parent.parent / "create-flows" / "mmult.mlir").read()
    )
    r = Router()
    pass_ = create_pathfinder_flows_with_python_pass(r)
    pm = PassManager()
    pass_manager_add_owned_pass(pm, pass_)
    device = mlir_module.body.operations[0]
    pm.run(device.operation)

    for src, paths in r.routing_solution.items():
        print(src)
        pprint(paths)

    print(mlir_module)


# CHECK-LABEL: TEST: test_more_flows_shim
@run
def test_more_flows_shim(ctx):
    mlir_module = Module.parse(
        open(
            Path(THIS_FILE).parent.parent / "create-flows" / "more_flows_shim.mlir"
        ).read()
    )
    r = Router()
    pass_ = create_pathfinder_flows_with_python_pass(r)
    pm = PassManager()
    pass_manager_add_owned_pass(pm, pass_)
    device = mlir_module.body.operations[0]
    pm.run(device.operation)

    for src, paths in r.routing_solution.items():
        print(src)
        pprint(paths)

    print(mlir_module)


# CHECK-LABEL: TEST: test_over_flows
@run
def test_over_flows(ctx):
    mlir_module = Module.parse(
        open(Path(THIS_FILE).parent.parent / "create-flows" / "over_flows.mlir").read()
    )
    r = Router()
    pass_ = create_pathfinder_flows_with_python_pass(r)
    pm = PassManager()
    pass_manager_add_owned_pass(pm, pass_)
    device = mlir_module.body.operations[0]
    pm.run(device.operation)

    for src, paths in r.routing_solution.items():
        print(src)
        pprint(paths)

    print(mlir_module)


# CHECK-LABEL: TEST: test_routed_herd_3x1
@run
def test_routed_herd_3x1(ctx):
    mlir_module = Module.parse(
        open(
            Path(THIS_FILE).parent.parent / "create-flows" / "routed_herd_3x1.mlir"
        ).read()
    )
    r = Router()
    pass_ = create_pathfinder_flows_with_python_pass(r)
    pm = PassManager()
    pass_manager_add_owned_pass(pm, pass_)
    device = mlir_module.body.operations[0]
    pm.run(device.operation)

    for src, paths in r.routing_solution.items():
        print(src)
        pprint(paths)

    print(mlir_module)


# CHECK-LABEL: TEST: test_routed_herd_3x2
@run
def test_routed_herd_3x2(ctx):
    mlir_module = Module.parse(
        open(
            Path(THIS_FILE).parent.parent / "create-flows" / "routed_herd_3x2.mlir"
        ).read()
    )
    r = Router()
    pass_ = create_pathfinder_flows_with_python_pass(r)
    pm = PassManager()
    pass_manager_add_owned_pass(pm, pass_)
    device = mlir_module.body.operations[0]
    pm.run(device.operation)

    for src, paths in r.routing_solution.items():
        print(src)
        pprint(paths)

    print(mlir_module)


# CHECK-LABEL: TEST: test_simple
@run
def test_simple(ctx):
    mlir_module = Module.parse(
        open(Path(THIS_FILE).parent.parent / "create-flows" / "simple.mlir").read()
    )
    r = Router()
    pass_ = create_pathfinder_flows_with_python_pass(r)
    pm = PassManager()
    pass_manager_add_owned_pass(pm, pass_)
    device = mlir_module.body.operations[0]
    pm.run(device.operation)

    # CHECK: PathEndPoint(Switchbox(0, 1): (DMA: 0))
    # CHECK: OrderedDict([(Switchbox(0, 1), (DMA: 0) -> {(N: 0)}),
    # CHECK:              (Switchbox(0, 2), (S: 0) -> {(E: 0)}),
    # CHECK:              (Switchbox(1, 2), (W: 0) -> {(Core: 1)})])
    for src, paths in r.routing_solution.items():
        print(src)
        pprint(paths)

    # CHECK: %switchbox_0_1 = AIE.switchbox(%tile_0_1) {
    # CHECK:   AIE.connect<DMA : 0, North : 0>
    # CHECK: }
    # CHECK: %switchbox_0_2 = AIE.switchbox(%tile_0_2) {
    # CHECK:   AIE.connect<South : 0, East : 0>
    # CHECK: }
    # CHECK: %switchbox_1_2 = AIE.switchbox(%tile_1_2) {
    # CHECK:   AIE.connect<West : 0, Core : 1>
    # CHECK: }
    # CHECK: AIE.packet_flow(16) {
    # CHECK:   AIE.packet_source<%tile_0_1, Core : 0>
    # CHECK:   AIE.packet_dest<%tile_1_2, Core : 0>
    # CHECK:   AIE.packet_dest<%tile_0_2, DMA : 1>
    # CHECK: }
    print(mlir_module)


# CHECK-LABEL: TEST: test_simple2
@run
def test_simple2(ctx):
    mlir_module = Module.parse(
        open(Path(THIS_FILE).parent.parent / "create-flows" / "simple2.mlir").read()
    )
    r = Router()
    pass_ = create_pathfinder_flows_with_python_pass(r)
    pm = PassManager()
    pass_manager_add_owned_pass(pm, pass_)
    device = mlir_module.body.operations[0]
    pm.run(device.operation)

    # CHECK: PathEndPoint(Switchbox(2, 3): (Core: 1))
    # CHECK: OrderedDict([(Switchbox(2, 3), (Core: 1) -> {(S: 0)}),
    # CHECK:              (Switchbox(2, 2), (N: 0) -> {(E: 0)}),
    # CHECK:              (Switchbox(3, 2), (W: 0) -> {(DMA: 0)})])
    for src, paths in r.routing_solution.items():
        print(src)
        pprint(paths)

    # CHECK: %switchbox_2_2 = AIE.switchbox(%tile_2_2) {
    # CHECK:   AIE.connect<North : 0, East : 0>
    # CHECK: }
    # CHECK: %switchbox_2_3 = AIE.switchbox(%tile_2_3) {
    # CHECK:   AIE.connect<Core : 1, South : 0>
    # CHECK: }
    # CHECK: %switchbox_3_2 = AIE.switchbox(%tile_3_2) {
    # CHECK:   AIE.connect<West : 0, DMA : 0>
    # CHECK: }
    print(mlir_module)


# CHECK-LABEL: TEST: test_simple_flows
@run
def test_simple_flows(ctx):
    mlir_module = Module.parse(
        open(
            Path(THIS_FILE).parent.parent / "create-flows" / "simple_flows.mlir"
        ).read()
    )
    r = Router()
    pass_ = create_pathfinder_flows_with_python_pass(r)
    pm = PassManager()
    pass_manager_add_owned_pass(pm, pass_)
    device = mlir_module.body.operations[0]
    pm.run(device.operation)

    for src, paths in r.routing_solution.items():
        print(src)
        pprint(paths)

    print(mlir_module)


# CHECK-LABEL: TEST: test_simple_flows2
@run
def test_simple_flows2(ctx):
    mlir_module = Module.parse(
        open(
            Path(THIS_FILE).parent.parent / "create-flows" / "simple_flows2.mlir"
        ).read()
    )
    r = Router()
    pass_ = create_pathfinder_flows_with_python_pass(r)
    pm = PassManager()
    pass_manager_add_owned_pass(pm, pass_)
    device = mlir_module.body.operations[0]
    pm.run(device.operation)

    # CHECK: PathEndPoint(Switchbox(2, 3): (Core: 0))
    # CHECK: OrderedDict([(Switchbox(2, 3), (Core: 0) -> {(S: 0)}),
    # CHECK:              (Switchbox(2, 2), (N: 0) -> {(Core: 1)})])
    # CHECK: PathEndPoint(Switchbox(2, 2): (Core: 0))
    # CHECK: OrderedDict([(Switchbox(2, 2), (Core: 0) -> {(W: 0)}),
    # CHECK:              (Switchbox(1, 2), (E: 0) -> {(S: 0)}),
    # CHECK:              (Switchbox(1, 1), (N: 0) -> {(Core: 0)})])
    for src, paths in r.routing_solution.items():
        print(src)
        pprint(paths)

    # CHECK: %switchbox_2_2 = AIE.switchbox(%tile_2_2) {
    # CHECK:   AIE.connect<North : 0, Core : 1>
    # CHECK:   AIE.connect<Core : 0, West : 0>
    # CHECK: }
    # CHECK: %switchbox_2_3 = AIE.switchbox(%tile_2_3) {
    # CHECK:   AIE.connect<Core : 0, South : 0>
    # CHECK: }
    # CHECK: %switchbox_1_1 = AIE.switchbox(%tile_1_1) {
    # CHECK:   AIE.connect<North : 0, Core : 0>
    # CHECK: }
    # CHECK: %tile_1_2 = AIE.tile(1, 2)
    # CHECK: %switchbox_1_2 = AIE.switchbox(%tile_1_2) {
    # CHECK:   AIE.connect<East : 0, South : 0>
    # CHECK: }
    print(mlir_module)


# CHECK-LABEL: TEST: test_simple_flows_shim
@run
def test_simple_flows_shim(ctx):
    mlir_module = Module.parse(
        open(
            Path(THIS_FILE).parent.parent / "create-flows" / "simple_flows_shim.mlir"
        ).read()
    )
    r = Router()
    pass_ = create_pathfinder_flows_with_python_pass(r)
    pm = PassManager()
    pass_manager_add_owned_pass(pm, pass_)
    device = mlir_module.body.operations[0]
    pm.run(device.operation)

    for src, paths in r.routing_solution.items():
        print(src)
        pprint(paths)

    print(mlir_module)


# @run
def test_vecmul_4x4(ctx):
    mlir_module = Module.parse(
        open(Path(THIS_FILE).parent.parent / "create-flows" / "vecmul_4x4.mlir").read()
    )
    r = Router()
    pass_ = create_pathfinder_flows_with_python_pass(r)
    pm = PassManager()
    pass_manager_add_owned_pass(pm, pass_)
    device = mlir_module.body.operations[0]
    pm.run(device.operation)

    for src, paths in r.routing_solution.items():
        print(src)
        pprint(paths)

    print(mlir_module)
