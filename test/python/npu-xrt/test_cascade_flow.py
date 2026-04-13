# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 AMD Inc.

# RUN: %pytest %s
# REQUIRES: ryzen_ai

import pytest
import numpy as np

from aie.iron import CascadeFlow, ObjectFifo, Program, Runtime, Worker
from aie.iron.device import NPU1Col2, Tile
from aie.iron.dataflow.cascadeflow import CascadeFlow as _CascadeFlow
from aie.iron.placers import SequentialPlacer
from aie.iron.controlflow import range_


def test_cascade_flow_construction():
    """CascadeFlow can be constructed with two Worker-like objects."""

    class _FakeTile:
        def __init__(self, col, row):
            self.op = None

    class _FakeWorker:
        def __init__(self, col, row):
            self.tile = _FakeTile(col, row)

    src = _FakeWorker(0, 2)
    dst = _FakeWorker(1, 2)
    cf = _CascadeFlow(src, dst)
    assert cf._src is src
    assert cf._dst is dst


def test_runtime_cascade_flow_registration():
    """Runtime.cascade_flow() accumulates CascadeFlow objects."""
    n_ty = np.ndarray[(64,), np.dtype[np.int32]]
    of_in = ObjectFifo(n_ty, name="in")
    of_out = ObjectFifo(n_ty, name="out")

    def noop(a, b):
        pass

    worker_a = Worker(noop, fn_args=[of_in.cons(), of_out.prod()], placement=Tile(0, 2))
    worker_b = Worker(noop, fn_args=[], placement=Tile(1, 2))

    rt = Runtime()
    with rt.sequence(n_ty, n_ty) as (A, B):
        rt.start(worker_a, worker_b)
        rt.cascade_flow(worker_a, worker_b)
        rt.fill(of_in.prod(), A)
        rt.drain(of_out.cons(), B, wait=True)

    assert len(rt._cascade_flows) == 1
    cf = rt._cascade_flows[0]
    assert isinstance(cf, _CascadeFlow)
    assert cf._src is worker_a
    assert cf._dst is worker_b


def test_cascade_flow_mlir_resolve():
    """CascadeFlow ops appear in the resolved MLIR module."""
    n_ty = np.ndarray[(64,), np.dtype[np.int32]]
    of_in = ObjectFifo(n_ty, name="cf_in")
    of_out = ObjectFifo(n_ty, name="cf_out")

    def noop(a, b):
        pass

    worker_a = Worker(noop, fn_args=[of_in.cons(), of_out.prod()], placement=Tile(0, 2))
    worker_b = Worker(noop, fn_args=[], placement=Tile(1, 2))

    rt = Runtime()
    with rt.sequence(n_ty, n_ty) as (A, B):
        rt.start(worker_a, worker_b)
        rt.cascade_flow(worker_a, worker_b)
        rt.fill(of_in.prod(), A)
        rt.drain(of_out.cons(), B, wait=True)

    module = Program(NPU1Col2(), rt).resolve_program()
    mlir_str = str(module)
    assert "cascade_flow" in mlir_str


def test_top_level_cascade_flow_import():
    """CascadeFlow is importable from aie.iron top-level."""
    from aie.iron import CascadeFlow as _CF

    assert _CF is _CascadeFlow
