# Copyright (C) 2022, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %PYTHON %s | FileCheck %s
# REQUIRES: python_passes


from pprint import pprint

from aie._mlir_libs._aie_python_passes import register_pathfinder_flows_with_python
from aie.dialects.aie import register_dialect
from aie.ir import Context, Location, Module
from aie.passmanager import PassManager
from aie.util import build_graph, route_using_cp, get_routing_solution


def run(f):
    with Context() as ctx, Location.unknown():
        register_dialect(ctx)
        print("\nTEST:", f.__name__)
        f(ctx)


# CHECK-LABEL: TEST: testPathfinderFlowsWithPython
@run
def testPathfinderFlowsWithPython(ctx):
    def find_paths(max_cols, max_rows, target_model, flows, fixed_connections):
        DG = build_graph(max_cols, max_rows, target_model)
        flow_paths = route_using_cp(DG, flows)

        # plot_src_paths(DG, flow_paths)
        routing_solution = get_routing_solution(DG, flow_paths)
        # CHECK: PathEndPoint(Switchbox(5, 2): (DMA: 0))
        # CHECK: OrderedDict([(Switchbox(5, 2), (DMA: 0) -> {(S: 0), (W: 0), (N: 0), (E: 0)}),
        # CHECK:              (Switchbox(5, 1), (N: 0) -> {(W: 0)}),
        # CHECK:              (Switchbox(4, 1), (E: 0) -> {(W: 0)}),
        # CHECK:              (Switchbox(3, 1), (E: 0) -> {(W: 0)}),
        # CHECK:              (Switchbox(2, 1), (E: 0) -> {(W: 0)}),
        # CHECK:              (Switchbox(1, 1), (E: 0) -> {(W: 0)}),
        # CHECK:              (Switchbox(0, 1), (E: 0) -> {(DMA: 0)}),
        # CHECK:              (Switchbox(4, 2), (E: 0) -> {(W: 0)}),
        # CHECK:              (Switchbox(3, 2), (E: 0) -> {(W: 0)}),
        # CHECK:              (Switchbox(2, 2), (E: 0) -> {(W: 0)}),
        # CHECK:              (Switchbox(1, 2), (E: 0) -> {(W: 0)}),
        # CHECK:              (Switchbox(0, 2), (E: 0) -> {(DMA: 0)}),
        # CHECK:              (Switchbox(5, 3), (S: 0) -> {(W: 0)}),
        # CHECK:              (Switchbox(4, 3), (E: 0) -> {(W: 0)}),
        # CHECK:              (Switchbox(3, 3), (E: 0) -> {(W: 0)}),
        # CHECK:              (Switchbox(2, 3), (E: 0) -> {(W: 0)}),
        # CHECK:              (Switchbox(1, 3), (E: 0) -> {(W: 0)}),
        # CHECK:              (Switchbox(0, 3), (E: 0) -> {(DMA: 0)}),
        # CHECK:              (Switchbox(6, 2), (W: 0) -> {(E: 0)}),
        # CHECK:              (Switchbox(7, 2), (W: 0) -> {(E: 0)}),
        # CHECK:              (Switchbox(8, 2), (W: 0) -> {(DMA: 0)})])
        # CHECK: PathEndPoint(Switchbox(8, 0): (DMA: 0))
        # CHECK: OrderedDict([(Switchbox(8, 0), (DMA: 0) -> {(W: 0), (N: 0)}),
        # CHECK:              (Switchbox(7, 0), (E: 0) -> {(W: 0)}),
        # CHECK:              (Switchbox(6, 0), (E: 0) -> {(W: 0)}),
        # CHECK:              (Switchbox(5, 0), (E: 0) -> {(W: 0)}),
        # CHECK:              (Switchbox(4, 0), (E: 0) -> {(W: 0)}),
        # CHECK:              (Switchbox(3, 0), (E: 0) -> {(W: 0)}),
        # CHECK:              (Switchbox(2, 0), (E: 0) -> {(W: 0)}),
        # CHECK:              (Switchbox(1, 0), (E: 0) -> {(W: 0)}),
        # CHECK:              (Switchbox(0, 0), (E: 0) -> {(DMA: 0)}),
        # CHECK:              (Switchbox(8, 1), (S: 0) -> {(N: 0)}),
        # CHECK:              (Switchbox(8, 2), (S: 0) -> {(N: 0)}),
        # CHECK:              (Switchbox(8, 3), (S: 0) -> {(DMA: 0)})])
        for src, paths in routing_solution.items():
            print(src)
            pprint(paths)

        return routing_solution

    module = """
      AIE.device(xcvc1902) {
        %tile_0_0 = AIE.tile(0, 0)
        %tile_0_1 = AIE.tile(0, 1)
        %tile_0_2 = AIE.tile(0, 2)
        %tile_0_3 = AIE.tile(0, 3)
        %tile_1_0 = AIE.tile(1, 0)
        %tile_1_1 = AIE.tile(1, 1)
        %tile_1_3 = AIE.tile(1, 3)
        %tile_2_0 = AIE.tile(2, 0)
        %tile_3_0 = AIE.tile(3, 0)
        %tile_2_2 = AIE.tile(2, 2)
        %tile_3_1 = AIE.tile(3, 1)
        %tile_5_0 = AIE.tile(5, 0)
        %tile_5_2 = AIE.tile(5, 2)
        %tile_6_0 = AIE.tile(6, 0)
        %tile_7_0 = AIE.tile(7, 0)
        %tile_7_1 = AIE.tile(7, 1)
        %tile_7_2 = AIE.tile(7, 2)
        %tile_7_3 = AIE.tile(7, 3)
        %tile_8_0 = AIE.tile(8, 0)
        %tile_8_1 = AIE.tile(8, 1)
        %tile_8_2 = AIE.tile(8, 2)
        %tile_8_3 = AIE.tile(8, 3)
        
        AIE.flow(%tile_5_2, DMA : 0, %tile_0_1, DMA : 0)
        AIE.flow(%tile_5_2, DMA : 0, %tile_0_2, DMA : 0)
        AIE.flow(%tile_5_2, DMA : 0, %tile_0_3, DMA : 0)
        AIE.flow(%tile_5_2, DMA : 0, %tile_8_2, DMA : 0)
        
        AIE.flow(%tile_8_0, DMA : 0, %tile_0_0, DMA : 0)
        AIE.flow(%tile_8_0, DMA : 0, %tile_8_3, DMA : 0)
      }
    """

    register_pathfinder_flows_with_python(find_paths)
    mlir_module = Module.parse(module)
    device = mlir_module.body.operations[0]
    pm = PassManager.parse("AIE.device(aie-create-pathfinder-flows-with-python)")
    # pm.enable_ir_printing()
    pm.run(device.operation)

    # CHECK: %switchbox_0_1 = AIE.switchbox(%tile_0_1) {
    # CHECK:   AIE.connect<East : 0, DMA : 0>
    # CHECK: }
    # CHECK: %switchbox_0_2 = AIE.switchbox(%tile_0_2) {
    # CHECK:   AIE.connect<East : 0, DMA : 0>
    # CHECK: }
    # CHECK: %switchbox_0_3 = AIE.switchbox(%tile_0_3) {
    # CHECK:   AIE.connect<East : 0, DMA : 0>
    # CHECK: }
    # CHECK: %switchbox_1_1 = AIE.switchbox(%tile_1_1) {
    # CHECK:   AIE.connect<East : 0, West : 0>
    # CHECK: }
    # CHECK: %tile_1_2 = AIE.tile(1, 2)
    # CHECK: %switchbox_1_2 = AIE.switchbox(%tile_1_2) {
    # CHECK:   AIE.connect<East : 0, West : 0>
    # CHECK: }
    # CHECK: %switchbox_1_3 = AIE.switchbox(%tile_1_3) {
    # CHECK:   AIE.connect<East : 0, West : 0>
    # CHECK: }
    # CHECK: %tile_2_1 = AIE.tile(2, 1)
    # CHECK: %switchbox_2_1 = AIE.switchbox(%tile_2_1) {
    # CHECK:   AIE.connect<East : 0, West : 0>
    # CHECK: }
    # CHECK: %switchbox_2_2 = AIE.switchbox(%tile_2_2) {
    # CHECK:   AIE.connect<East : 0, West : 0>
    # CHECK: }
    # CHECK: %tile_2_3 = AIE.tile(2, 3)
    # CHECK: %switchbox_2_3 = AIE.switchbox(%tile_2_3) {
    # CHECK:   AIE.connect<East : 0, West : 0>
    # CHECK: }
    # CHECK: %switchbox_3_1 = AIE.switchbox(%tile_3_1) {
    # CHECK:   AIE.connect<East : 0, West : 0>
    # CHECK: }
    # CHECK: %tile_3_2 = AIE.tile(3, 2)
    # CHECK: %switchbox_3_2 = AIE.switchbox(%tile_3_2) {
    # CHECK:   AIE.connect<East : 0, West : 0>
    # CHECK: }
    # CHECK: %tile_3_3 = AIE.tile(3, 3)
    # CHECK: %switchbox_3_3 = AIE.switchbox(%tile_3_3) {
    # CHECK:   AIE.connect<East : 0, West : 0>
    # CHECK: }
    # CHECK: %tile_4_1 = AIE.tile(4, 1)
    # CHECK: %switchbox_4_1 = AIE.switchbox(%tile_4_1) {
    # CHECK:   AIE.connect<East : 0, West : 0>
    # CHECK: }
    # CHECK: %tile_4_2 = AIE.tile(4, 2)
    # CHECK: %switchbox_4_2 = AIE.switchbox(%tile_4_2) {
    # CHECK:   AIE.connect<East : 0, West : 0>
    # CHECK: }
    # CHECK: %tile_4_3 = AIE.tile(4, 3)
    # CHECK: %switchbox_4_3 = AIE.switchbox(%tile_4_3) {
    # CHECK:   AIE.connect<East : 0, West : 0>
    # CHECK: }
    # CHECK: %tile_5_1 = AIE.tile(5, 1)
    # CHECK: %switchbox_5_1 = AIE.switchbox(%tile_5_1) {
    # CHECK:   AIE.connect<North : 0, West : 0>
    # CHECK: }
    # CHECK: %switchbox_5_2 = AIE.switchbox(%tile_5_2) {
    # CHECK:   AIE.connect<DMA : 0, South : 0>
    # CHECK:   AIE.connect<DMA : 0, West : 0>
    # CHECK:   AIE.connect<DMA : 0, North : 0>
    # CHECK:   AIE.connect<DMA : 0, East : 0>
    # CHECK: }
    # CHECK: %tile_5_3 = AIE.tile(5, 3)
    # CHECK: %switchbox_5_3 = AIE.switchbox(%tile_5_3) {
    # CHECK:   AIE.connect<South : 0, West : 0>
    # CHECK: }
    # CHECK: %tile_6_2 = AIE.tile(6, 2)
    # CHECK: %switchbox_6_2 = AIE.switchbox(%tile_6_2) {
    # CHECK:   AIE.connect<West : 0, East : 0>
    # CHECK: }
    # CHECK: %switchbox_7_2 = AIE.switchbox(%tile_7_2) {
    # CHECK:   AIE.connect<West : 0, East : 0>
    # CHECK: }
    # CHECK: %switchbox_8_2 = AIE.switchbox(%tile_8_2) {
    # CHECK:   AIE.connect<West : 0, DMA : 0>
    # CHECK:   AIE.connect<South : 0, North : 0>
    # CHECK: }
    # CHECK: %switchbox_0_0 = AIE.switchbox(%tile_0_0) {
    # CHECK:   AIE.connect<East : 0, South : 0>
    # CHECK: }
    # CHECK: %switchbox_1_0 = AIE.switchbox(%tile_1_0) {
    # CHECK:   AIE.connect<East : 0, West : 0>
    # CHECK: }
    # CHECK: %switchbox_2_0 = AIE.switchbox(%tile_2_0) {
    # CHECK:   AIE.connect<East : 0, West : 0>
    # CHECK: }
    # CHECK: %switchbox_3_0 = AIE.switchbox(%tile_3_0) {
    # CHECK:   AIE.connect<East : 0, West : 0>
    # CHECK: }
    # CHECK: %tile_4_0 = AIE.tile(4, 0)
    # CHECK: %switchbox_4_0 = AIE.switchbox(%tile_4_0) {
    # CHECK:   AIE.connect<East : 0, West : 0>
    # CHECK: }
    # CHECK: %switchbox_5_0 = AIE.switchbox(%tile_5_0) {
    # CHECK:   AIE.connect<East : 0, West : 0>
    # CHECK: }
    # CHECK: %switchbox_6_0 = AIE.switchbox(%tile_6_0) {
    # CHECK:   AIE.connect<East : 0, West : 0>
    # CHECK: }
    # CHECK: %switchbox_7_0 = AIE.switchbox(%tile_7_0) {
    # CHECK:   AIE.connect<East : 0, West : 0>
    # CHECK: }
    # CHECK: %switchbox_8_0 = AIE.switchbox(%tile_8_0) {
    # CHECK:   AIE.connect<South : 0, West : 0>
    # CHECK:   AIE.connect<South : 0, North : 0>
    # CHECK: }
    # CHECK: %switchbox_8_1 = AIE.switchbox(%tile_8_1) {
    # CHECK:   AIE.connect<South : 0, North : 0>
    # CHECK: }
    # CHECK: %switchbox_8_3 = AIE.switchbox(%tile_8_3) {
    # CHECK:   AIE.connect<South : 0, DMA : 0>
    # CHECK: }
    print(mlir_module)
