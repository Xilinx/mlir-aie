# placers.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 Advanced Micro Devices, Inc.

from abc import ABCMeta, abstractmethod

from .phys.device import Device
from .runtime import Runtime
from .runtime.runtimeendpoint import RuntimeEndpoint
from .worker import Worker
from .phys.tile import AnyComputeTile, AnyMemTile
from .dataflow.objectfifo import ObjectFifoHandle


class Placer(metaclass=ABCMeta):

    @abstractmethod
    def make_placement(
        self,
        device: Device,
        rt: Runtime,
        workers: list[Worker],
        object_fifos: list[ObjectFifoHandle],
    ): ...


class SequentialPlacer(Placer):

    def __init__(self):
        super().__init__()

    def make_placement(
        self,
        device: Device,
        rt: Runtime,
        workers: list[Worker],
        object_fifos: list[ObjectFifoHandle],
    ):
        shims = device.get_shim_tiles()

        mems = device.get_mem_tiles()
        mem_idx = 0  # Loop over memtiles

        computes = device.get_compute_tiles()
        compute_idx = 0  # Will not loop over core tiles

        for worker in workers:
            if worker.tile == AnyComputeTile:
                assert compute_idx < len(
                    computes
                ), "Ran out of compute tiles for placement!"
                worker.place(computes[compute_idx])
                compute_idx += 1

        for of in object_fifos:
            of_endpoints = of.get_all_endpoints()
            # RuntimeEndpoints are placed by the Runtime
            of_endpoints = [
                of for of in of_endpoints if not isinstance(of, RuntimeEndpoint)
            ]
            for ofe in of_endpoints:
                if ofe.tile == AnyMemTile:
                    ofe.place(mems[mem_idx])
                    mem_idx = (mem_idx + 1) % len(mems)
                elif ofe.tile == AnyComputeTile:
                    assert compute_idx < len(
                        computes
                    ), "Ran out of compute tiles for placement!"
                    ofe.place(computes[compute_idx])
                    compute_idx += 1

        rt.place_tasks(shims)
