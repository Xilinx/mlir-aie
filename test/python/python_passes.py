# Copyright (C) 2022, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %PYTHON %s | FileCheck %s
# REQUIRES: python_passes


import aie
from aie.ir import *
from aie.dialects.aie import *
from aie.passmanager import PassManager
from aie._mlir_libs._aie_python_passes import (
    register_pathfinder_flows_with_python,
    WireBundle,
    SwitchSetting,
    Switchbox,
    Port,
    get_connecting_bundle,
)
import matplotlib.pyplot as plt
from collections import defaultdict, OrderedDict

import matplotlib as mpl


import gurobipy as gp
from gurobipy import GRB
import networkx as nx


def build_graph(max_cols, max_rows, target_model):
    DG = nx.DiGraph()
    for c in range(max_cols + 1):
        for r in range(max_rows + 1):
            this_switchbox = (c, r)
            DG.add_node(this_switchbox)
            if r > 0:
                southern_neighbor = (c, r - 1)
                if max_capacity := target_model.get_num_source_switchbox_connections(
                    c, r, WireBundle.South
                ):
                    DG.add_edge(
                        southern_neighbor,
                        this_switchbox,
                        bundle=WireBundle.North,
                        capacity=max_capacity,
                    )
                if max_capacity := target_model.get_num_dest_switchbox_connections(
                    c, r, WireBundle.South
                ):
                    DG.add_edge(
                        this_switchbox,
                        southern_neighbor,
                        bundle=WireBundle.South,
                        capacity=max_capacity,
                    )
            if c > 0:
                western_neighbor = (c - 1, r)
                if max_capacity := target_model.get_num_source_switchbox_connections(
                    c, r, WireBundle.West
                ):
                    DG.add_edge(
                        western_neighbor,
                        this_switchbox,
                        bundle=WireBundle.East,
                        capacity=max_capacity,
                    )
                if max_capacity := target_model.get_num_dest_switchbox_connections(
                    c, r, WireBundle.West
                ):
                    DG.add_edge(
                        this_switchbox,
                        western_neighbor,
                        bundle=WireBundle.West,
                        capacity=max_capacity,
                    )

    return DG


def route_using_ilp(DG, flows):
    # Create model object
    m = gp.Model()

    # Create variable for each edge, for each path
    flow_vars = {}
    flat_flow_vars = []
    for flow in flows:
        flow_var = m.addVars(DG.edges, vtype=GRB.BINARY, name="flow")
        flow_vars[flow] = flow_var
        flat_flow_vars.append(flow_var)

    # Add flow-balance constraints at all nodes (besides sources and targets)
    for (src, tgt), flow_var in zip(flows, flat_flow_vars):
        src = src.sb.col, src.sb.row
        tgt = tgt.sb.col, tgt.sb.row

        for n in DG.nodes:
            if n in {src, tgt}:
                continue

            # what goes in must come out
            m.addConstr(
                gp.quicksum(flow_var[e] for e in DG.in_edges(nbunch=n))
                == gp.quicksum(flow_var[e] for e in DG.out_edges(nbunch=n))
            )

            # flow must leave src, and must not enter src
            m.addConstr(gp.quicksum(flow_var[src, j] for j in DG.neighbors(src)) == 1)
            m.addConstr(gp.quicksum(flow_var[i, src] for i in DG.neighbors(src)) == 0)

            # flow must enter tgt, and must not leave tgt
            m.addConstr(gp.quicksum(flow_var[tgt, j] for j in DG.neighbors(tgt)) == 0)
            m.addConstr(gp.quicksum(flow_var[i, tgt] for i in DG.neighbors(tgt)) == 1)

    # Create demand variables
    total_demand = m.addVars(DG.edges)
    overlapping_demands = m.addVars(DG.edges)

    # Add demand/flow relationship
    for i, j, attrs in DG.edges(data=True):
        m.addConstr(total_demand[i, j] == gp.quicksum(f[i, j] for f in flat_flow_vars))
        m.addConstr(total_demand[i, j] <= attrs["capacity"])
        m.addConstr(
            overlapping_demands[i, j]
            == gp.quicksum(
                (f1[i, j] * f2[i, j])
                for k, f1 in enumerate(flat_flow_vars)
                for f2 in flat_flow_vars[k + 1 :]
            )
        )

    # Objective function: total system time
    m.setObjective(
        # 1,
        gp.quicksum(
            total_demand[i, j] + overlapping_demands[i, j] for i, j in DG.edges
        ),
        GRB.MINIMIZE,
    )

    # Solve
    m.optimize()

    flow_paths = {}
    for flow, flow_varss in flow_vars.items():
        flow_paths[flow] = [(i, j) for i, j in DG.edges if flow_varss[i, j].x > 0.5]

    return flow_paths


def rgb2hex(r, g, b, a):
    return "#{:02x}{:02x}{:02x}".format(
        int(r * 255), int(g * 255), int(b * 255), int(a * 255)
    )


def plot_paths(DG, src, paths):
    pos = dict((n, n) for n in DG.nodes())
    labels = dict(((i, j), f"{i},{j}") for i, j in DG.nodes())

    fig, ax = plt.subplots()
    nx.draw(
        DG,
        with_labels=True,
        edge_color="white",
        node_color=["green" if n == src else "gray" for n in DG.nodes],
        pos=pos,
        labels=labels,
        ax=ax,
    )

    colors = lambda x: mpl.colormaps["prism"](x / len(paths))
    for j, path in enumerate(paths):
        nx.draw_networkx_edges(
            DG, pos=pos, edgelist=path, edge_color=rgb2hex(*colors(j)), width=4, ax=ax
        )

    plt.show()


def plot_src_paths(DG, flow_paths):
    src_paths = defaultdict(list)
    for (src, _), path in flow_paths.items():
        src_paths[src.sb.col, src.sb.row].append(path)

    for src, paths in src_paths.items():
        plot_paths(DG, src, paths)


def get_routing_solution(DG, flow_paths):
    routing_solution = defaultdict(OrderedDict)
    # OrderedDict just for debugging aid, i.e., src stays to the far left in
    # repr and such.
    for flow, path in flow_paths.items():
        src, tgt = flow
        switch_settings = routing_solution[src]

        if src.sb not in switch_settings:
            switch_settings[src.sb] = SwitchSetting(src.port)
            routing_solution[src] = switch_settings
        else:
            assert switch_settings[src.sb].src == src.port

        path_subgraph = DG.edge_subgraph(path)
        prev = src.sb.col, src.sb.row
        while curr := path_subgraph.successors(prev):
            curr = list(curr)
            if not curr:
                assert prev == (tgt.sb.col, tgt.sb.row)
                break
            assert len(curr) == 1, "multiple curressors not supported"
            curr = curr[0]
            data = path_subgraph.get_edge_data(prev, curr)

            # Add the successor Switchbox to the destinations of curr
            switch_settings[Switchbox(*prev)].dsts.add(Port(data["bundle"], 0))

            # Add the entrance port for next Switchbox
            # TODO(max): used_capacity isn't being recovered from ILP
            # Why does capacity mean the same thing as channel?
            # TODO(max): channel isn't used - stub it.
            switch_settings[Switchbox(*curr)] = SwitchSetting(
                Port(get_connecting_bundle(data["bundle"]), 0)
            )
            prev = curr

    for src, switch_settings in routing_solution.items():
        assert len(switch_settings) == len(
            set(
                [
                    sb
                    for ((s, t), fp) in flow_paths.items()
                    for e in fp
                    for sb in e
                    if s == src
                ]
            )
        )
        routing_solution[src] = dict(switch_settings)

    return dict(routing_solution)


def testPathfinderFlowsWithPython():
    print("\nTEST: testPathfinderFlowsWithPython")

    def find_paths(max_cols, max_rows, target_model, flows, fixed_connections):
        DG = build_graph(max_cols, max_rows, target_model)
        flow_paths = route_using_ilp(DG, flows)

        # plot_src_paths(DG, flow_paths)
        routing_solution = get_routing_solution(DG, flow_paths)

        return routing_solution

    module = """
      AIE.device(xcvc1902) {
        %tile_0_3 = AIE.tile(0, 3)
        %tile_0_2 = AIE.tile(0, 2)
        %tile_0_0 = AIE.tile(0, 0)
        %tile_1_3 = AIE.tile(1, 3)
        %tile_1_1 = AIE.tile(1, 1)
        %tile_1_0 = AIE.tile(1, 0)
        %tile_2_0 = AIE.tile(2, 0)
        %tile_3_0 = AIE.tile(3, 0)
        %tile_2_2 = AIE.tile(2, 2)
        %tile_3_1 = AIE.tile(3, 1)
        %tile_6_0 = AIE.tile(6, 0)
        %tile_7_0 = AIE.tile(7, 0)
        %tile_7_1 = AIE.tile(7, 1)
        %tile_7_2 = AIE.tile(7, 2)
        %tile_7_3 = AIE.tile(7, 3)
        %tile_8_0 = AIE.tile(8, 0)
        %tile_8_2 = AIE.tile(8, 2)
        %tile_8_3 = AIE.tile(8, 3)
        
        AIE.flow(%tile_2_0, DMA : 0, %tile_1_3, DMA : 0)
        AIE.flow(%tile_2_0, DMA : 0, %tile_3_1, DMA : 0)
        AIE.flow(%tile_2_0, DMA : 0, %tile_7_1, DMA : 0)
        AIE.flow(%tile_2_0, DMA : 0, %tile_8_2, DMA : 0)
        AIE.flow(%tile_6_0, DMA : 0, %tile_0_2, DMA : 1)
        AIE.flow(%tile_6_0, DMA : 0, %tile_8_3, DMA : 1)
        AIE.flow(%tile_6_0, DMA : 0, %tile_2_2, DMA : 1)
        AIE.flow(%tile_6_0, DMA : 0, %tile_3_1, DMA : 1)
      }
    """

    with Context() as ctx, Location.unknown():
        aie.dialects.aie.register_dialect(ctx)
        register_pathfinder_flows_with_python(find_paths)
        mlir_module = Module.parse(module)
        device = mlir_module.body.operations[0]
        PassManager.parse("AIE.device(aie-create-pathfinder-flows-with-python)").run(
            device.operation
        )

        print(mlir_module)


# CHECK-LABEL: TEST: testPathfinderFlowsWithPython
# CHECK: module {
# CHECK:   AIE.device(xcvc1902) {
# CHECK:     %tile_0_3 = AIE.tile(0, 3)
# CHECK:     %tile_0_2 = AIE.tile(0, 2)
# CHECK:     %tile_0_0 = AIE.tile(0, 0)
# CHECK:     %tile_1_3 = AIE.tile(1, 3)
# CHECK:     %tile_1_1 = AIE.tile(1, 1)
# CHECK:     %tile_1_0 = AIE.tile(1, 0)
# CHECK:     %tile_2_0 = AIE.tile(2, 0)
# CHECK:     %tile_3_0 = AIE.tile(3, 0)
# CHECK:     %tile_2_2 = AIE.tile(2, 2)
# CHECK:     %tile_3_1 = AIE.tile(3, 1)
# CHECK:     %tile_6_0 = AIE.tile(6, 0)
# CHECK:     %tile_7_0 = AIE.tile(7, 0)
# CHECK:     %tile_7_1 = AIE.tile(7, 1)
# CHECK:     %tile_7_2 = AIE.tile(7, 2)
# CHECK:     %tile_7_3 = AIE.tile(7, 3)
# CHECK:     %tile_8_0 = AIE.tile(8, 0)
# CHECK:     %tile_8_2 = AIE.tile(8, 2)
# CHECK:     %tile_8_3 = AIE.tile(8, 3)
# CHECK:     %switchbox_1_0 = AIE.switchbox(%tile_1_0) {
# CHECK:       AIE.connect<East : 0, North : 0>
# CHECK:     }
# CHECK:     %switchbox_1_1 = AIE.switchbox(%tile_1_1) {
# CHECK:       AIE.connect<South : 0, North : 0>
# CHECK:       AIE.connect<East : 0, West : 0>
# CHECK:     }
# CHECK:     %tile_1_2 = AIE.tile(1, 2)
# CHECK:     %switchbox_1_2 = AIE.switchbox(%tile_1_2) {
# CHECK:       AIE.connect<South : 0, North : 0>
# CHECK:     }
# CHECK:     %switchbox_1_3 = AIE.switchbox(%tile_1_3) {
# CHECK:     }
# CHECK:     %switchbox_2_0 = AIE.switchbox(%tile_2_0) {
# CHECK:       AIE.connect<South : 3, West : 0>
# CHECK:       AIE.connect<South : 3, North : 0>
# CHECK:       AIE.connect<South : 3, East : 0>
# CHECK:     }
# CHECK:     %shimmux_2_0 = AIE.shimmux(%tile_2_0) {
# CHECK:       AIE.connect<DMA : 0, North : 3>
# CHECK:     }
# CHECK:     %tile_2_1 = AIE.tile(2, 1)
# CHECK:     %switchbox_2_1 = AIE.switchbox(%tile_2_1) {
# CHECK:       AIE.connect<South : 0, North : 0>
# CHECK:       AIE.connect<East : 0, West : 0>
# CHECK:     }
# CHECK:     %switchbox_2_2 = AIE.switchbox(%tile_2_2) {
# CHECK:       AIE.connect<South : 0, East : 0>
# CHECK:     }
# CHECK:     %switchbox_3_0 = AIE.switchbox(%tile_3_0) {
# CHECK:       AIE.connect<West : 0, East : 0>
# CHECK:       AIE.connect<East : 0, North : 0>
# CHECK:     }
# CHECK:     %switchbox_3_1 = AIE.switchbox(%tile_3_1) {
# CHECK:     }
# CHECK:     %tile_3_2 = AIE.tile(3, 2)
# CHECK:     %switchbox_3_2 = AIE.switchbox(%tile_3_2) {
# CHECK:       AIE.connect<West : 0, East : 0>
# CHECK:       AIE.connect<East : 0, West : 0>
# CHECK:     }
# CHECK:     %tile_4_0 = AIE.tile(4, 0)
# CHECK:     %switchbox_4_0 = AIE.switchbox(%tile_4_0) {
# CHECK:       AIE.connect<West : 0, North : 0>
# CHECK:       AIE.connect<East : 0, West : 0>
# CHECK:     }
# CHECK:     %tile_4_1 = AIE.tile(4, 1)
# CHECK:     %switchbox_4_1 = AIE.switchbox(%tile_4_1) {
# CHECK:       AIE.connect<South : 0, East : 0>
# CHECK:       AIE.connect<East : 0, West : 0>
# CHECK:     }
# CHECK:     %tile_4_2 = AIE.tile(4, 2)
# CHECK:     %switchbox_4_2 = AIE.switchbox(%tile_4_2) {
# CHECK:       AIE.connect<West : 0, East : 0>
# CHECK:       AIE.connect<East : 0, West : 0>
# CHECK:     }
# CHECK:     %tile_5_1 = AIE.tile(5, 1)
# CHECK:     %switchbox_5_1 = AIE.switchbox(%tile_5_1) {
# CHECK:       AIE.connect<West : 0, East : 0>
# CHECK:       AIE.connect<East : 0, West : 0>
# CHECK:     }
# CHECK:     %tile_5_2 = AIE.tile(5, 2)
# CHECK:     %switchbox_5_2 = AIE.switchbox(%tile_5_2) {
# CHECK:       AIE.connect<West : 0, East : 0>
# CHECK:       AIE.connect<South : 0, West : 0>
# CHECK:     }
# CHECK:     %tile_6_1 = AIE.tile(6, 1)
# CHECK:     %switchbox_6_1 = AIE.switchbox(%tile_6_1) {
# CHECK:       AIE.connect<West : 0, East : 0>
# CHECK:       AIE.connect<South : 0, West : 0>
# CHECK:     }
# CHECK:     %tile_6_2 = AIE.tile(6, 2)
# CHECK:     %switchbox_6_2 = AIE.switchbox(%tile_6_2) {
# CHECK:       AIE.connect<West : 0, East : 0>
# CHECK:     }
# CHECK:     %switchbox_7_1 = AIE.switchbox(%tile_7_1) {
# CHECK:       AIE.connect<South : 0, North : 0>
# CHECK:     }
# CHECK:     %switchbox_7_2 = AIE.switchbox(%tile_7_2) {
# CHECK:       AIE.connect<West : 0, East : 0>
# CHECK:       AIE.connect<South : 0, North : 0>
# CHECK:     }
# CHECK:     %switchbox_8_2 = AIE.switchbox(%tile_8_2) {
# CHECK:     }
# CHECK:     %tile_0_1 = AIE.tile(0, 1)
# CHECK:     %switchbox_0_1 = AIE.switchbox(%tile_0_1) {
# CHECK:       AIE.connect<East : 0, North : 0>
# CHECK:     }
# CHECK:     %switchbox_0_2 = AIE.switchbox(%tile_0_2) {
# CHECK:     }
# CHECK:     %tile_5_0 = AIE.tile(5, 0)
# CHECK:     %switchbox_5_0 = AIE.switchbox(%tile_5_0) {
# CHECK:       AIE.connect<East : 0, North : 0>
# CHECK:     }
# CHECK:     %switchbox_6_0 = AIE.switchbox(%tile_6_0) {
# CHECK:       AIE.connect<South : 3, West : 0>
# CHECK:       AIE.connect<South : 3, North : 0>
# CHECK:       AIE.connect<South : 3, East : 0>
# CHECK:     }
# CHECK:     %shimmux_6_0 = AIE.shimmux(%tile_6_0) {
# CHECK:       AIE.connect<DMA : 0, North : 3>
# CHECK:     }
# CHECK:     %switchbox_7_0 = AIE.switchbox(%tile_7_0) {
# CHECK:       AIE.connect<West : 0, North : 0>
# CHECK:     }
# CHECK:     %switchbox_7_3 = AIE.switchbox(%tile_7_3) {
# CHECK:       AIE.connect<South : 0, East : 0>
# CHECK:     }
# CHECK:     %switchbox_8_3 = AIE.switchbox(%tile_8_3) {
# CHECK:     }
# CHECK:     AIE.wire(%tile_0_1 : Core, %switchbox_0_1 : Core)
# CHECK:     AIE.wire(%tile_0_1 : DMA, %switchbox_0_1 : DMA)
# CHECK:     AIE.wire(%tile_0_2 : Core, %switchbox_0_2 : Core)
# CHECK:     AIE.wire(%tile_0_2 : DMA, %switchbox_0_2 : DMA)
# CHECK:     AIE.wire(%switchbox_0_1 : North, %switchbox_0_2 : South)
# CHECK:     AIE.wire(%switchbox_0_1 : East, %switchbox_1_1 : West)
# CHECK:     AIE.wire(%tile_1_1 : Core, %switchbox_1_1 : Core)
# CHECK:     AIE.wire(%tile_1_1 : DMA, %switchbox_1_1 : DMA)
# CHECK:     AIE.wire(%switchbox_1_0 : North, %switchbox_1_1 : South)
# CHECK:     AIE.wire(%switchbox_0_2 : East, %switchbox_1_2 : West)
# CHECK:     AIE.wire(%tile_1_2 : Core, %switchbox_1_2 : Core)
# CHECK:     AIE.wire(%tile_1_2 : DMA, %switchbox_1_2 : DMA)
# CHECK:     AIE.wire(%switchbox_1_1 : North, %switchbox_1_2 : South)
# CHECK:     AIE.wire(%tile_1_3 : Core, %switchbox_1_3 : Core)
# CHECK:     AIE.wire(%tile_1_3 : DMA, %switchbox_1_3 : DMA)
# CHECK:     AIE.wire(%switchbox_1_2 : North, %switchbox_1_3 : South)
# CHECK:     AIE.wire(%switchbox_1_0 : East, %switchbox_2_0 : West)
# CHECK:     AIE.wire(%shimmux_2_0 : North, %switchbox_2_0 : South)
# CHECK:     AIE.wire(%tile_2_0 : DMA, %shimmux_2_0 : DMA)
# CHECK:     AIE.wire(%switchbox_1_1 : East, %switchbox_2_1 : West)
# CHECK:     AIE.wire(%tile_2_1 : Core, %switchbox_2_1 : Core)
# CHECK:     AIE.wire(%tile_2_1 : DMA, %switchbox_2_1 : DMA)
# CHECK:     AIE.wire(%switchbox_2_0 : North, %switchbox_2_1 : South)
# CHECK:     AIE.wire(%switchbox_1_2 : East, %switchbox_2_2 : West)
# CHECK:     AIE.wire(%tile_2_2 : Core, %switchbox_2_2 : Core)
# CHECK:     AIE.wire(%tile_2_2 : DMA, %switchbox_2_2 : DMA)
# CHECK:     AIE.wire(%switchbox_2_1 : North, %switchbox_2_2 : South)
# CHECK:     AIE.wire(%switchbox_2_0 : East, %switchbox_3_0 : West)
# CHECK:     AIE.wire(%switchbox_2_1 : East, %switchbox_3_1 : West)
# CHECK:     AIE.wire(%tile_3_1 : Core, %switchbox_3_1 : Core)
# CHECK:     AIE.wire(%tile_3_1 : DMA, %switchbox_3_1 : DMA)
# CHECK:     AIE.wire(%switchbox_3_0 : North, %switchbox_3_1 : South)
# CHECK:     AIE.wire(%switchbox_2_2 : East, %switchbox_3_2 : West)
# CHECK:     AIE.wire(%tile_3_2 : Core, %switchbox_3_2 : Core)
# CHECK:     AIE.wire(%tile_3_2 : DMA, %switchbox_3_2 : DMA)
# CHECK:     AIE.wire(%switchbox_3_1 : North, %switchbox_3_2 : South)
# CHECK:     AIE.wire(%switchbox_3_0 : East, %switchbox_4_0 : West)
# CHECK:     AIE.wire(%switchbox_3_1 : East, %switchbox_4_1 : West)
# CHECK:     AIE.wire(%tile_4_1 : Core, %switchbox_4_1 : Core)
# CHECK:     AIE.wire(%tile_4_1 : DMA, %switchbox_4_1 : DMA)
# CHECK:     AIE.wire(%switchbox_4_0 : North, %switchbox_4_1 : South)
# CHECK:     AIE.wire(%switchbox_3_2 : East, %switchbox_4_2 : West)
# CHECK:     AIE.wire(%tile_4_2 : Core, %switchbox_4_2 : Core)
# CHECK:     AIE.wire(%tile_4_2 : DMA, %switchbox_4_2 : DMA)
# CHECK:     AIE.wire(%switchbox_4_1 : North, %switchbox_4_2 : South)
# CHECK:     AIE.wire(%switchbox_4_0 : East, %switchbox_5_0 : West)
# CHECK:     AIE.wire(%switchbox_4_1 : East, %switchbox_5_1 : West)
# CHECK:     AIE.wire(%tile_5_1 : Core, %switchbox_5_1 : Core)
# CHECK:     AIE.wire(%tile_5_1 : DMA, %switchbox_5_1 : DMA)
# CHECK:     AIE.wire(%switchbox_5_0 : North, %switchbox_5_1 : South)
# CHECK:     AIE.wire(%switchbox_4_2 : East, %switchbox_5_2 : West)
# CHECK:     AIE.wire(%tile_5_2 : Core, %switchbox_5_2 : Core)
# CHECK:     AIE.wire(%tile_5_2 : DMA, %switchbox_5_2 : DMA)
# CHECK:     AIE.wire(%switchbox_5_1 : North, %switchbox_5_2 : South)
# CHECK:     AIE.wire(%switchbox_5_0 : East, %switchbox_6_0 : West)
# CHECK:     AIE.wire(%shimmux_6_0 : North, %switchbox_6_0 : South)
# CHECK:     AIE.wire(%tile_6_0 : DMA, %shimmux_6_0 : DMA)
# CHECK:     AIE.wire(%switchbox_5_1 : East, %switchbox_6_1 : West)
# CHECK:     AIE.wire(%tile_6_1 : Core, %switchbox_6_1 : Core)
# CHECK:     AIE.wire(%tile_6_1 : DMA, %switchbox_6_1 : DMA)
# CHECK:     AIE.wire(%switchbox_6_0 : North, %switchbox_6_1 : South)
# CHECK:     AIE.wire(%switchbox_5_2 : East, %switchbox_6_2 : West)
# CHECK:     AIE.wire(%tile_6_2 : Core, %switchbox_6_2 : Core)
# CHECK:     AIE.wire(%tile_6_2 : DMA, %switchbox_6_2 : DMA)
# CHECK:     AIE.wire(%switchbox_6_1 : North, %switchbox_6_2 : South)
# CHECK:     AIE.wire(%switchbox_6_0 : East, %switchbox_7_0 : West)
# CHECK:     AIE.wire(%switchbox_6_1 : East, %switchbox_7_1 : West)
# CHECK:     AIE.wire(%tile_7_1 : Core, %switchbox_7_1 : Core)
# CHECK:     AIE.wire(%tile_7_1 : DMA, %switchbox_7_1 : DMA)
# CHECK:     AIE.wire(%switchbox_7_0 : North, %switchbox_7_1 : South)
# CHECK:     AIE.wire(%switchbox_6_2 : East, %switchbox_7_2 : West)
# CHECK:     AIE.wire(%tile_7_2 : Core, %switchbox_7_2 : Core)
# CHECK:     AIE.wire(%tile_7_2 : DMA, %switchbox_7_2 : DMA)
# CHECK:     AIE.wire(%switchbox_7_1 : North, %switchbox_7_2 : South)
# CHECK:     AIE.wire(%tile_7_3 : Core, %switchbox_7_3 : Core)
# CHECK:     AIE.wire(%tile_7_3 : DMA, %switchbox_7_3 : DMA)
# CHECK:     AIE.wire(%switchbox_7_2 : North, %switchbox_7_3 : South)
# CHECK:     AIE.wire(%switchbox_7_2 : East, %switchbox_8_2 : West)
# CHECK:     AIE.wire(%tile_8_2 : Core, %switchbox_8_2 : Core)
# CHECK:     AIE.wire(%tile_8_2 : DMA, %switchbox_8_2 : DMA)
# CHECK:     AIE.wire(%switchbox_7_3 : East, %switchbox_8_3 : West)
# CHECK:     AIE.wire(%tile_8_3 : Core, %switchbox_8_3 : Core)
# CHECK:     AIE.wire(%tile_8_3 : DMA, %switchbox_8_3 : DMA)
# CHECK:     AIE.wire(%switchbox_8_2 : North, %switchbox_8_3 : South)
# CHECK:   }
# CHECK: }
testPathfinderFlowsWithPython()
