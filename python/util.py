# Copyright (C) 2022, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import multiprocessing
import numbers
import os
from collections import defaultdict
from typing import List, Tuple, Dict, Set
import numpy as np
from numpy.lib.stride_tricks import as_strided


def build_graph(max_cols, max_rows, target_model):
    import networkx as nx
    from ._mlir_libs._aie_python_passes import WireBundle, Switchbox

    DG = nx.DiGraph()
    for c in range(max_cols + 1):
        for r in range(max_rows + 1):
            this_switchbox = Switchbox(c, r)
            DG.add_node(this_switchbox)
            if r > 0:
                southern_neighbor = Switchbox(c, r - 1)
                # Get the number of outgoing connections on the south side - outgoing
                # because these correspond to rhs of a connect op.
                if max_capacity := target_model.get_num_dest_switchbox_connections(
                    c, r, WireBundle.South
                ):
                    DG.add_edge(
                        this_switchbox,
                        southern_neighbor,
                        bundle=WireBundle.South,
                        capacity=max_capacity,
                    )
                # Get the number of incoming connections on the south side - incoming
                # because they correspond to connections on the southside that are then
                # routed using internal connect ops through the switchbox (i.e., lhs of
                # connect ops).
                if max_capacity := target_model.get_num_source_switchbox_connections(
                    c, r, WireBundle.South
                ):
                    DG.add_edge(
                        southern_neighbor,
                        this_switchbox,
                        bundle=WireBundle.North,
                        capacity=max_capacity,
                    )
            if c > 0:
                western_neighbor = Switchbox(c - 1, r)
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


def route_using_cp(
    DG,
    flows,
    min_edges=False,
    seed=42,
    num_workers=multiprocessing.cpu_count() // 2,
    timeout=600,
):
    from ortools.sat.python import cp_model

    # Create model object
    model = cp_model.CpModel()
    solver = cp_model.CpSolver()
    # For determinism
    solver.parameters.random_seed = seed
    solver.parameters.num_workers = num_workers
    solver.parameters.max_time_in_seconds = timeout

    # Create variable for each edge, for each path
    flow_vars = {
        flow: {(i, j): model.NewIntVar(0, 1, "") for i, j in DG.edges} for flow in flows
    }
    flat_flow_vars = list(flow_vars.values())

    # Add flow-balance constraints at all nodes (besides sources and targets)
    for (src, tgt), flow_var in zip(flows, flat_flow_vars):
        src, tgt = src.sb, tgt.sb

        # flow must leave src, and must not enter src
        model.Add(sum(flow_var[src, j] for j in DG.neighbors(src)) == 1)
        model.Add(sum(flow_var[i, src] for i in DG.neighbors(src)) == 0)

        # flow must enter tgt, and must not leave tgt
        model.Add(sum(flow_var[tgt, j] for j in DG.neighbors(tgt)) == 0)
        model.Add(sum(flow_var[i, tgt] for i in DG.neighbors(tgt)) == 1)

        for n in DG.nodes:
            if n in {src, tgt}:
                continue

            # what goes in must come out
            model.Add(
                sum(flow_var[e] for e in DG.in_edges(nbunch=n))
                == sum(flow_var[e] for e in DG.out_edges(nbunch=n))
            )

    total_demand = {
        (i, j): model.NewIntVar(0, len(flat_flow_vars), "") for i, j in DG.edges
    }
    overlapping_demands = {
        (i, j): model.NewIntVar(0, len(flat_flow_vars), "") for i, j in DG.edges
    }
    used_edges = {(i, j): model.NewIntVar(0, 1, "") for i, j in DG.edges}

    for i, j, attrs in DG.edges(data=True):
        model.Add(total_demand[i, j] == sum(f[i, j] for f in flat_flow_vars))
        model.Add(total_demand[i, j] <= attrs["capacity"])

        if min_edges:
            # counts whether an edge is used by any flow
            model.AddMaxEquality(used_edges[i, j], [f[i, j] for f in flat_flow_vars])
        else:
            # counts the number of overlapping flows
            overlapping_flows = {}
            for k, f1 in enumerate(flat_flow_vars):
                for l, f2 in enumerate(flat_flow_vars[k + 1 :], start=k + 1):
                    # for each pair of flows k, l (at edge i,j)
                    # overlapping_flows[k, l] is 1 if both flows use edge i,j
                    # note this could also by a MinEquality i.e.,
                    # model.AddMinEquality(overlapping_flows[k, l], [f1[i, j], f2[i, j]])
                    overlapping_flows[k, l] = model.NewIntVar(
                        0, len(flat_flow_vars), ""
                    )
                    model.AddMultiplicationEquality(
                        overlapping_flows[k, l], [f1[i, j], f2[i, j]]
                    )

            # overlapping demands counts up all overlapping flows on the edge i,j
            # by summing across all overlapping_flows[k, j], which as written above, count whether flows k, j
            # overlap on edge i,j
            model.Add(
                overlapping_demands[i, j]
                == sum(
                    overlapping_flows[k, l]
                    for k, f1 in enumerate(flat_flow_vars)
                    for l, f2 in enumerate(flat_flow_vars[k + 1 :], start=k + 1)
                )
            )

    obj = sum(total_demand[i, j] for i, j in DG.edges)
    if min_edges:
        obj += sum(used_edges[i, j] for i, j in DG.edges)
    else:
        obj += sum(overlapping_demands[i, j] for i, j in DG.edges)
    model.Minimize(obj)

    status = solver.Solve(model)
    if status in {cp_model.OPTIMAL, cp_model.FEASIBLE}:
        flow_paths = {}
        for flow, flow_varss in flow_vars.items():
            flow_paths[flow] = [
                # solver.Value > 0.5 means the edge is used
                (i, j)
                for i, j in DG.edges
                if solver.Value(flow_varss[i, j]) > 0.5
            ]

        return flow_paths

    raise RuntimeError("Couldn't route.")


def route_using_ilp(
    DG,
    flows,
    timeout=600,
):
    import gurobipy as gp
    from gurobipy import GRB

    m = gp.Model()
    m.setParam("TimeLimit", timeout)

    flow_vars = {
        flow: m.addVars(DG.edges, vtype=GRB.BINARY, name="flow") for flow in flows
    }
    flat_flow_vars = list(flow_vars.values())

    for (src, tgt), flow_var in zip(flows, flat_flow_vars):
        src, tgt = src.sb, tgt.sb
        # flow must leave src, and must not enter src
        m.addConstr(gp.quicksum(flow_var[src, j] for j in DG.neighbors(src)) == 1)
        m.addConstr(gp.quicksum(flow_var[i, src] for i in DG.neighbors(src)) == 0)

        # flow must enter tgt, and must not leave tgt
        m.addConstr(gp.quicksum(flow_var[tgt, j] for j in DG.neighbors(tgt)) == 0)
        m.addConstr(gp.quicksum(flow_var[i, tgt] for i in DG.neighbors(tgt)) == 1)

        for n in DG.nodes:
            if n in {src, tgt}:
                continue

            # what goes in must come out
            m.addConstr(
                gp.quicksum(flow_var[e] for e in DG.in_edges(nbunch=n))
                == gp.quicksum(flow_var[e] for e in DG.out_edges(nbunch=n))
            )

    # Create demand variables
    total_demand = m.addVars(DG.edges)
    overlapping_demands = m.addVars(DG.edges)

    # Add demand/flow relationship
    for i, j, attrs in DG.edges(data=True):
        m.addConstr(total_demand[i, j] == gp.quicksum(f[i, j] for f in flat_flow_vars))
        m.addConstr(total_demand[i, j] <= attrs["capacity"])
        # See above for this counts up overlapping demands (gurobi just has a nicer API).
        m.addConstr(
            overlapping_demands[i, j]
            == gp.quicksum(
                (f1[i, j] * f2[i, j])
                for k, f1 in enumerate(flat_flow_vars)
                for f2 in flat_flow_vars[k + 1 :]
            )
        )

    m.setObjective(
        gp.quicksum(
            total_demand[i, j] + overlapping_demands[i, j] for i, j in DG.edges
        ),
        GRB.MINIMIZE,
    )

    # Solve
    m.optimize()

    if m.Status == GRB.INFEASIBLE:
        raise RuntimeError("Couldn't route.")

    flow_paths = {}
    for flow, flow_varss in flow_vars.items():
        # x > 0.5 means the edge is used
        flow_paths[flow] = [(i, j) for i, j in DG.edges if flow_varss[i, j].x > 0.5]

    return flow_paths


def rgb2hex(r, g, b, a):
    return f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}{int(a * 255):02x}"


def plot_paths(DG, src, paths):
    import networkx as nx
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    pos = dict((n, (n.col, n.row)) for n in DG.nodes())
    labels = dict((n, f"{n.col},{n.row}") for n in DG.nodes())

    _fig, ax = plt.subplots()
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


MAX_NUM_CHANNELS = 12


def get_routing_solution(DG, flow_paths, used_channels):
    from ._mlir_libs._aie_python_passes import (
        SwitchSetting,
        Port,
        get_connecting_bundle,
    )

    def get_pred(path, curr_sb):
        path_subgraph = DG.edge_subgraph(path)
        pred_sb = list(path_subgraph.predecessors(curr_sb))
        assert len(pred_sb) == 1
        pred_sb = pred_sb[0]
        incoming_edge = pred_sb, curr_sb
        # outgoing bundle of pred, corresponding to rhs of connect op
        outgoing_bundle = path_subgraph.get_edge_data(*incoming_edge)["bundle"]
        return pred_sb, outgoing_bundle

    flow_dsts = defaultdict(list)
    for flow, path in flow_paths.items():
        src, tgt = flow
        flow_dsts[src].append((tgt, path))
        # endpoints function as "fixed connections" i.e., already
        # assigned channels on some bundle
        used_channels[tgt.sb, tgt.port.bundle].add(tgt.port.channel)

    def get_next_avail_channel(sb, bundle):
        i = 0
        while i in used_channels[sb, bundle]:
            i += 1
        used_channels[sb, bundle].add(i)
        return i

    routing_solution = {}
    for src, dsts in flow_dsts.items():
        switch_settings = defaultdict(SwitchSetting)
        processed = {src.sb}

        # Trace backwards until a vertex already processed is reached
        for end_point, path in dsts:
            curr_sb = end_point.sb
            pred_sb, _ = get_pred(path, curr_sb)
            switch_settings[curr_sb].dsts.add(end_point.port)

            while curr_sb not in processed:
                pred_sb, outgoing_bundle = get_pred(path, curr_sb)
                # connecting bundle on curr, lhs of connect op
                incoming_bundle = get_connecting_bundle(outgoing_bundle)
                # next avail channel on outgoing side on pred
                matching_channel = get_next_avail_channel(pred_sb, outgoing_bundle)

                switch_settings[curr_sb].src = Port(incoming_bundle, matching_channel)
                switch_settings[pred_sb].dsts.add(
                    Port(outgoing_bundle, matching_channel)
                )

                processed.add(curr_sb)
                curr_sb = pred_sb

        switch_settings[src.sb].src = src.port
        routing_solution[src] = list(reversed(switch_settings.items()))

    return routing_solution


def pythonize_bool(value):
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, numbers.Number):
        return value != 0
    if isinstance(value, str):
        if value.lower() in ("1", "true", "on", "yes"):
            return True
        if value.lower() in ("", "0", "false", "off", "no"):
            return False
    raise ValueError(f'"{value}" is not a valid boolean')


class Router:
    max_col: int
    max_row: int
    timeout: int
    use_gurobi: bool = False
    # Don't use actual binding here to prevent a blow up since class bodies are executed
    # at module load time.
    target_model: "AIETargetModel"
    flows: List[Tuple["PathEndPoint", "PathEndPoint"]]
    used_channels: Dict[Tuple["Switchbox", "Switchbox"], Set[int]]
    routing_solution: Dict["PathEndPoint", "SwitchSettings"]

    def __init__(self, use_gurobi=False, timeout=600):
        self.flows = []
        self.routing_solution = None
        self.use_gurobi = use_gurobi or pythonize_bool(
            os.getenv("ROUTER_USE_GUROBI", "False")
        )
        self.timeout = timeout
        self.used_channels = defaultdict(set)

    def initialize(self, max_col, max_row, target_model):
        self.max_col = max_col
        self.max_row = max_row
        self.target_model = target_model
        self.DG = build_graph(self.max_col, self.max_row, self.target_model)

    def add_flow(self, src: "PathEndPoint", tgt: "PathEndPoint"):
        self.flows.append((src, tgt))

    def add_fixed_connection(self, connect_op):
        from ._mlir_libs._aie_python_passes import get_connecting_bundle

        tileid = connect_op.get_switchbox().get_tileid()
        lhs_port = connect_op.get_src_port()
        rhs_port = connect_op.get_dst_port()

        # find the correct Channel and indicate the fixed direction

        # outgoing connection
        matching_outgoing_edges = [
            (u, v, e)
            for u, v, e in self.DG.edges(data=True)
            if e["bundle"] == rhs_port.bundle
            and u == tileid  # i.e., this tile is the source
        ]
        if len(matching_outgoing_edges):
            assert len(matching_outgoing_edges) == 1
            u, v, e = matching_outgoing_edges[0]
            e["capacity"] -= 1
            self.used_channels[u, rhs_port.bundle].add(rhs_port.channel)
            return True

        # incoming connection
        matching_incoming_edges = [
            (u, v, e)
            for u, v, e in self.DG.edges(data=True)
            if e["bundle"] == get_connecting_bundle(lhs_port.bundle)
            and v == tileid  # i.e., this tile is the target
        ]
        if len(matching_incoming_edges):
            assert len(matching_incoming_edges) == 1
            u, v, e = matching_incoming_edges[0]
            e["capacity"] -= 1
            # this is where the assumption that connection ports across
            # tiles use the same channel comes in
            assert e["bundle"] == get_connecting_bundle(lhs_port.bundle)
            self.used_channels[u, e["bundle"]].add(lhs_port.channel)
            return True

        return False

    def find_paths(self):
        if self.routing_solution is None:
            if self.use_gurobi:
                flow_paths = route_using_ilp(self.DG, self.flows, timeout=self.timeout)
            else:
                flow_paths = route_using_cp(
                    self.DG, self.flows, num_workers=10, timeout=self.timeout
                )

            self.routing_solution = get_routing_solution(
                self.DG, flow_paths, self.used_channels
            )

        return {k: dict(v) for k, v in self.routing_solution.items()}

    def is_legal(self):
        return True


def tiling_calculator_tile_sizes(*matrix_dims, tile_n_cols=4, tile_n_rows=4):
    rows, cols = matrix_dims
    n_tiles_row = rows // tile_n_rows
    n_tiles_col = cols // tile_n_cols

    sizes_strides = [
        [n_tiles_row, cols * tile_n_rows],
        [n_tiles_col, tile_n_cols],
        [tile_n_rows, cols],
        [tile_n_cols, 1],
    ]

    return sizes_strides


def tiling_calculator_n_tiles(*matrix_dims, n_tile_rows=4, n_tile_cols=4):
    rows, cols = matrix_dims
    tile_n_rows = rows // n_tile_rows
    tile_n_cols = cols // n_tile_cols

    sizes_strides = [
        [n_tile_rows, cols * tile_n_rows],
        [n_tile_cols, tile_n_cols],
        [tile_n_rows, cols],
        [tile_n_cols, 1],
    ]

    return sizes_strides


def _to_js(sizes_strides):
    # plug into https://andreroesti.com/data-layout-viz/data_layout.html
    return f"""
    set_transforms([
        {sizes_strides}
    ])
    """


# based on https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/feature_extraction/image.py#L290
def extract_patches(arr, patch_shape=8, extraction_step=1):
    arr_ndim = arr.ndim

    if isinstance(patch_shape, numbers.Number):
        patch_shape = tuple([patch_shape] * arr_ndim)
    if isinstance(extraction_step, numbers.Number):
        extraction_step = tuple([extraction_step] * arr_ndim)

    patch_strides = arr.strides

    slices = tuple(slice(None, None, st) for st in extraction_step)
    indexing_strides = arr[slices].strides

    patch_indices_shape = (
        (np.array(arr.shape) - np.array(patch_shape)) // np.array(extraction_step)
    ) + 1

    shape = tuple(list(patch_indices_shape) + list(patch_shape))
    strides = tuple(list(indexing_strides) + list(patch_strides))

    patches = as_strided(arr, shape=shape, strides=strides)
    return patches
