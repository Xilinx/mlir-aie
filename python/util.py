import multiprocessing
import numbers
import os
import warnings
from collections import defaultdict
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from typing import List, Tuple, Dict
from typing import Optional
from typing import Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from .extras import types as T
from .ir import (
    F32Type,
    F64Type,
    IntegerType,
    RankedTensorType,
    Context,
    Module,
    Location,
    InsertionPoint,
)

_np_dtype_to_mlir_type_ctor = {
    np.int8: T.i8,
    np.int16: T.i16,
    np.int32: T.i32,  # windows
    np.intc: T.i32,
    np.int64: T.i64,  # is technically wrong i guess but numpy by default casts python scalars to this
    # so to support passing lists of ints we map to index type
    np.longlong: T.index,
    np.uintp: T.index,
    np.float16: T.f16,
    np.float32: T.f32,
    np.float64: T.f64,
}


def np_dtype_to_mlir_type(np_dtype):
    if typ := _np_dtype_to_mlir_type_ctor.get(np_dtype):
        return typ()


def infer_mlir_type(
    py_val: Union[int, float, bool, np.ndarray]
) -> Union[IntegerType, F32Type, F64Type, RankedTensorType]:
    """Infer MLIR type (`ir.Type`) from supported python values.

    Note ints and floats are mapped to 64-bit types.

    Args:
      py_val: Python value that's either a numerical value or numpy array.

    Returns:
      MLIR type corresponding to py_val.
    """
    if isinstance(py_val, bool):
        return T.bool()

    if isinstance(py_val, int):
        # no clue why but black can't decide which it wants the **
        # fmt: off
        if -(2 ** 31) <= py_val < 2 ** 31:
            return T.i32()
        elif 2 ** 31 <= py_val < 2 ** 32:
            return T.ui32()
        elif -(2 ** 63) <= py_val < 2 ** 63:
            return T.i64()
        elif 2 ** 63 <= py_val < 2 ** 64:
            return T.ui64()
        raise RuntimeError(f"Nonrepresentable integer {py_val}.")
        # fmt: on

    if isinstance(py_val, float):
        if (
            abs(py_val) == float("inf")
            or abs(py_val) == 0.0
            or py_val != py_val  # NaN
            or np.finfo(np.float32).min <= abs(py_val) <= np.finfo(np.float32).max
        ):
            return T.f32()
        return T.f64()

    if isinstance(py_val, np.ndarray):
        dtype = np_dtype_to_mlir_type(py_val.dtype.type)
        return RankedTensorType.get(py_val.shape, dtype)

    raise NotImplementedError(
        f"Unsupported Python value {py_val=} with type {type(py_val)}"
    )


def mlir_type_to_np_dtype(mlir_type):
    _mlir_type_to_np_dtype = {v(): k for k, v in _np_dtype_to_mlir_type_ctor.items()}
    return _mlir_type_to_np_dtype.get(mlir_type)


@dataclass
class MLIRContext:
    context: Context
    module: Module

    def __str__(self):
        return str(self.module)


@contextmanager
def mlir_mod_ctx(
    src: Optional[str] = None,
    context: Context = None,
    location: Location = None,
    allow_unregistered_dialects=False,
) -> MLIRContext:
    if context is None:
        context = Context()
    if allow_unregistered_dialects:
        context.allow_unregistered_dialects = True
    with ExitStack() as stack:
        stack.enter_context(context)
        if location is None:
            location = Location.unknown()
        stack.enter_context(location)
        if src is not None:
            module = Module.parse(src)
        else:
            module = Module.create()
        ip = InsertionPoint(module.body)
        stack.enter_context(ip)
        yield MLIRContext(context, module)


def build_graph(max_cols, max_rows, target_model):
    from ._mlir_libs._aie_python_passes import WireBundle, Switchbox

    DG = nx.DiGraph()
    for c in range(max_cols + 1):
        for r in range(max_rows + 1):
            this_switchbox = Switchbox(c, r)
            DG.add_node(this_switchbox)
            if r > 0:
                southern_neighbor = Switchbox(c, r - 1)
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
    seed=10,
    num_workers=multiprocessing.cpu_count() // 2,
    max_time_in_seconds=600,
):
    from ortools.sat.python import cp_model

    # Create model object
    model = cp_model.CpModel()
    solver = cp_model.CpSolver()
    # For determinism
    solver.parameters.random_seed = seed
    solver.parameters.num_workers = num_workers
    solver.parameters.max_time_in_seconds = max_time_in_seconds

    # Create variable for each edge, for each path
    flow_vars = {}
    flat_flow_vars = []
    for flow in flows:
        flow_var = {(i, j): model.NewIntVar(0, 1, "") for i, j in DG.edges}
        flow_vars[flow] = flow_var
        flat_flow_vars.append(flow_var)

    # Add flow-balance constraints at all nodes (besides sources and targets)
    for (src, tgt), flow_var in zip(flows, flat_flow_vars):
        src, tgt = src.sb, tgt.sb

        for n in DG.nodes:
            if n in {src, tgt}:
                continue

            # what goes in must come out
            model.Add(
                sum(flow_var[e] for e in DG.in_edges(nbunch=n))
                == sum(flow_var[e] for e in DG.out_edges(nbunch=n))
            )

            # flow must leave src, and must not enter src
            model.Add(sum(flow_var[src, j] for j in DG.neighbors(src)) == 1)
            model.Add(sum(flow_var[i, src] for i in DG.neighbors(src)) == 0)

            # flow must enter tgt, and must not leave tgt
            model.Add(sum(flow_var[tgt, j] for j in DG.neighbors(tgt)) == 0)
            model.Add(sum(flow_var[i, tgt] for i in DG.neighbors(tgt)) == 1)

    # Create demand variables
    total_demand = {
        (i, j): model.NewIntVar(0, len(flat_flow_vars), "") for i, j in DG.edges
    }
    overlapping_demands = {
        (i, j): model.NewIntVar(0, len(flat_flow_vars), "") for i, j in DG.edges
    }
    used_edges = {(i, j): model.NewIntVar(0, 1, "") for i, j in DG.edges}

    # Add demand/flow relationship
    for i, j, attrs in DG.edges(data=True):
        model.Add(total_demand[i, j] == sum(f[i, j] for f in flat_flow_vars))
        model.Add(total_demand[i, j] <= attrs["capacity"])

        if min_edges:
            model.AddMaxEquality(used_edges[i, j], [f[i, j] for f in flat_flow_vars])
        else:
            overlapping_flows = {}
            for k, f1 in enumerate(flat_flow_vars):
                for l, f2 in enumerate(flat_flow_vars[k + 1 :], start=k + 1):
                    overlapping_flows[k, l] = model.NewIntVar(
                        0, len(flat_flow_vars), ""
                    )
                    model.AddMultiplicationEquality(
                        overlapping_flows[k, l], [f1[i, j], f2[i, j]]
                    )

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
                (i, j) for i, j in DG.edges if solver.Value(flow_varss[i, j]) > 0.5
            ]

        return flow_paths

    warnings.warn("Couldn't route.")


def route_using_ilp(DG, flows):
    import gurobipy as gp
    from gurobipy import GRB

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
        src, tgt = src.sb, tgt.sb
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

    m.setObjective(
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
    return f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}{int(a * 255):02x}"


def plot_paths(DG, src, paths):
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


def get_routing_solution(DG, flow_paths):
    from ._mlir_libs._aie_python_passes import (
        SwitchSetting,
        Port,
        get_connecting_bundle,
    )

    flow_dsts = defaultdict(list)
    for flow, path in flow_paths.items():
        src, tgt = flow
        flow_dsts[src].append((tgt, path))

    # I don't know what's going on here but AIECreatePathFindFlows has some assumptions
    # hard-coded about matching channels on both the source and target port of a connection.
    # So keep track on a "per-edge" basis and use the same channel for both port.
    used_channel = defaultdict(lambda: 0)

    routing_solution = {}
    for src, dsts in flow_dsts.items():
        switch_settings = defaultdict(SwitchSetting)
        switch_settings[src.sb].src = src.port
        processed = {src.sb}

        # Trace backwards until a vertex already processed is reached
        for end_point, path in dsts:
            path_subgraph = DG.edge_subgraph(path)
            curr_sb = end_point.sb
            switch_settings[curr_sb].dsts.add(end_point.port)

            while curr_sb not in processed:
                pred_sb = list(path_subgraph.predecessors(curr_sb))
                assert len(pred_sb) == 1
                pred_sb = pred_sb[0]
                edge = pred_sb, curr_sb
                bundle = path_subgraph.get_edge_data(*edge)["bundle"]

                # add the entrance port for this Switchbox
                switch_settings[curr_sb].src = Port(
                    get_connecting_bundle(bundle), used_channel[edge]
                )
                # add the current Switchbox to the map of the predecessor
                switch_settings[pred_sb].dsts.add(Port(bundle, used_channel[edge]))
                used_channel[edge] += 1

                processed.add(curr_sb)
                curr_sb = pred_sb

        routing_solution[src] = dict(switch_settings)

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
    use_gurobi: bool = False
    # Don't use actual binding here to prevent a blow up since class bodies are executed
    # at module load time.
    target_model: "AIETargetModel"
    flows: List[Tuple["PathEndPoint", "PathEndPoint"]]
    fixed_connections: List[Tuple["TileID", "Port"]]
    routing_solution: Dict["PathEndPoint", "SwitchSettings"]

    def __init__(self, use_gurobi=False):
        self.flows = []
        self.fixed_connections = []
        self.routing_solution = None
        self.use_gurobi = use_gurobi or pythonize_bool(
            os.getenv("ROUTER_USE_GUROBI", "False")
        )

    def initialize(self, max_col, max_row, target_model):
        self.max_col = max_col
        self.max_row = max_row
        self.target_model = target_model

    def add_flow(self, src: "PathEndPoint", tgt: "PathEndPoint"):
        self.flows.append((src, tgt))

    def add_fixed_connection(self, connect_op):
        raise NotImplementedError("adding fixed connections not implemented yet.")

    def find_paths(self):
        if self.routing_solution is None:
            DG = build_graph(self.max_col, self.max_row, self.target_model)
            if self.use_gurobi:
                flow_paths = route_using_ilp(DG, self.flows)
            else:
                flow_paths = route_using_cp(DG, self.flows, num_workers=10)

            self.routing_solution = get_routing_solution(DG, flow_paths)
        return self.routing_solution

    def is_legal(self):
        return True
