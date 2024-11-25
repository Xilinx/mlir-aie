# runtime.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 Advanced Micro Devices, Inc.

from __future__ import annotations

from contextlib import contextmanager
import numpy as np


from ... import ir  # type: ignore

from ...dialects.aiex import runtime_sequence
from ...dialects._aiex_ops_gen import dma_free_task
from ...helpers.taplib import TensorAccessPattern
from ..dataflow.objectfifo import ObjectFifoHandle
from ..phys.tile import PlacementTile, AnyShimTile, Tile
from ..resolvable import Resolvable
from .dmatask import DMATask
from .runtimedata import RuntimeData
from .runtimeendpoint import RuntimeEndpoint
from ..worker import Worker


class Runtime(Resolvable):
    def __init__(
        self,
    ) -> Runtime:
        self._rt_data = []
        self._ops = []
        self._fifos = set()
        self._workers = []

    @contextmanager
    def sequence(self, *input_types: type[np.ndarray]):
        self._rt_data = list(map(RuntimeData, input_types))
        yield tuple(self._rt_data)

    def fill(
        self,
        in_fifo: ObjectFifoHandle,
        tap: TensorAccessPattern,
        source: RuntimeData,
        wait: bool = False,
        placement: PlacementTile = AnyShimTile,
    ) -> None:
        assert source in self._rt_data
        rt_endpoint = RuntimeEndpoint(placement)

        # There can only be one runtime endpoint in an ObjectFIFO
        if in_fifo in self._fifos:
            existing_endpoints = in_fifo.get_endpoint()
            for ep in existing_endpoints:
                if isinstance(ep, RuntimeEndpoint):
                    if ep.tile != placement:
                        if ep.tile == AnyShimTile:
                            in_fifo.replace_endpoint(ep, rt_endpoint)
                        else:
                            raise ValueError(
                                f"ObjectFIFO can only have one RuntimeEndpoint: has {ep}, trying to set: {rt_endpoint}"
                            )
        else:
            in_fifo.set_endpoint(rt_endpoint)
            self._fifos.add(in_fifo)
        self._ops.append(DMATask(in_fifo, source, tap, wait))

    def drain(
        self,
        out_fifo: ObjectFifoHandle,
        tap: TensorAccessPattern,
        dest: RuntimeData,
        wait: bool = False,
        placement: PlacementTile = AnyShimTile,
    ) -> None:
        assert dest in self._rt_data
        rt_endpoint = RuntimeEndpoint(placement)

        if out_fifo in self._fifos:
            existing_endpoints = out_fifo.get_endpoint()
            for ep in existing_endpoints:
                if isinstance(ep, RuntimeEndpoint):
                    if ep.tile != placement:
                        ep.place(placement)
                        if ep.tile == AnyShimTile:
                            ep.place(placement)
                        else:
                            raise ValueError(
                                f"ObjectFIFO can only have one RuntimeEndpoint: has {ep}, trying to set: {rt_endpoint}"
                            )
        else:
            out_fifo.set_endpoint(rt_endpoint)
            self._fifos.add(out_fifo)
        self._ops.append(DMATask(out_fifo, dest, tap, wait))

    def start(self, *args: Worker):
        for worker in args:
            if not isinstance(worker, Worker):
                raise ValueError("Runtime can only start Worker objects")
            self._workers.append(worker)

    def get_workers(self) -> list[Worker]:
        return self._workers.copy()

    def get_fifos(self) -> list[ObjectFifoHandle]:
        return self._fifos.copy()

    def place_tasks(self, shim_tiles: list[Tile]) -> None:
        # TODO: move to placer?
        for op in self._ops:
            ofe = op.fifo.get_endpoint()
            ofe = ofe[0]  # un-listify
            assert isinstance(
                ofe, RuntimeEndpoint
            ), f"Expected RuntimeEndpoint, but found {type(ofe)}"
            if ofe.tile == AnyShimTile:
                ofe.place(shim_tiles[0])

    def resolve(
        self,
        loc: ir.Location | None = None,
        ip: ir.InsertionPoint | None = None,
    ) -> None:
        rt_dtypes = [rt_data.dtype for rt_data in self._rt_data]

        @runtime_sequence(*rt_dtypes)
        def sequence(*args):

            for rt_data, rt_data_val in zip(self._rt_data, args):
                rt_data.op = rt_data_val

            no_waits = []
            for dma_task in self._ops:
                dma_task.resolve()
                if dma_task.will_wait():
                    for t in no_waits:
                        dma_free_task(t.task)
                    no_waits = []
                else:
                    no_waits.append(dma_task)
