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
from aie.iron.controlflow import range_


def test_cascade_flow_construction():
    """CascadeFlow can be constructed with two Worker-like objects."""

    class _FakeTile:
        def __init__(self, col, row):
            self.op = None

    class _FakeWorker:
        def __init__(self, col, row):
            self.tile = _FakeTile(col, row)
            self._outgoing_cascades = []

    src = _FakeWorker(0, 2)
    dst = _FakeWorker(1, 2)
    cf = _CascadeFlow(src, dst)
    assert cf._src is src
    assert cf._dst is dst
    assert src._outgoing_cascades == [cf]


def test_runtime_cascade_flow_registration():
    """CascadeFlow self-registers on its source Worker."""
    n_ty = np.ndarray[(64,), np.dtype[np.int32]]
    of_in = ObjectFifo(n_ty, name="in")
    of_out = ObjectFifo(n_ty, name="out")

    def noop(*args):
        pass

    worker_a = Worker(noop, fn_args=[of_in.cons(), of_out.prod()], tile=Tile(0, 2))
    worker_b = Worker(noop, fn_args=[], tile=Tile(1, 2))

    cf = CascadeFlow(worker_a, worker_b)

    rt = Runtime()
    with rt.sequence(n_ty, n_ty) as (A, B):
        rt.start(worker_a, worker_b)
        rt.fill(of_in.prod(), A)
        rt.drain(of_out.cons(), B, wait=True)

    assert worker_a._outgoing_cascades == [cf]
    assert isinstance(cf, _CascadeFlow)
    assert cf._src is worker_a
    assert cf._dst is worker_b


def test_cascade_flow_mlir_resolve():
    """CascadeFlow ops appear in the resolved MLIR module."""
    n_ty = np.ndarray[(64,), np.dtype[np.int32]]
    of_in = ObjectFifo(n_ty, name="cf_in")
    of_out = ObjectFifo(n_ty, name="cf_out")

    def noop(*args):
        pass

    worker_a = Worker(noop, fn_args=[of_in.cons(), of_out.prod()], tile=Tile(0, 2))
    worker_b = Worker(noop, fn_args=[], tile=Tile(1, 2))

    CascadeFlow(worker_a, worker_b)

    rt = Runtime()
    with rt.sequence(n_ty, n_ty) as (A, B):
        rt.start(worker_a, worker_b)
        rt.fill(of_in.prod(), A)
        rt.drain(of_out.cons(), B, wait=True)

    module = Program(NPU1Col2(), rt).resolve_program()
    mlir_str = str(module)
    assert "cascade_flow" in mlir_str


def test_top_level_cascade_flow_import():
    """CascadeFlow is importable from aie.iron top-level."""
    from aie.iron import CascadeFlow as _CF

    assert _CF is _CascadeFlow
