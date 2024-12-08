# program.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 Advanced Micro Devices, Inc.

from ..extras.context import mlir_mod_ctx  # type: ignore
from ..helpers.dialects.ext.func import FuncBase
from ..dialects.aie import device

from .device import Device
from .runtime import Runtime
from .placers import Placer
from .resolvable import Resolvable


class Program:
    def __init__(
        self,
        device: Device,
        rt: Runtime,
    ):
        self._device = device
        self._rt = rt

    def resolve_program(self, placer: Placer | None = None):
        with mlir_mod_ctx() as ctx:

            @device(self._device.resolve())
            def device_body():
                # Collect all fifos
                all_fifos = set()
                all_fifos.update(self._rt.fifos)
                for w in self._rt.workers:
                    all_fifos.update(w.fifos)

                if placer:
                    # TODO: should maybe just take runtime?
                    placer.make_placement(
                        self._device, self._rt, self._rt.workers, all_fifos
                    )

                # Collect all tiles
                all_tiles = []
                for w in self._rt.workers:
                    all_tiles.append(w.tile)
                for f in all_fifos:
                    all_tiles.extend([e.tile for e in f.all_of_endpoints()])

                # Resolve tiles
                for t in all_tiles:
                    self._device.resolve_tile(t)

                # Generate fifos
                for f in all_fifos:
                    f.resolve()

                # generate functions - this may call resolve() more than once on the same fifo, but that's ok
                for w in self._rt.workers:
                    for arg in w.fn_args:
                        if isinstance(arg, FuncBase):
                            arg.emit()
                        elif isinstance(arg, Resolvable):
                            arg.resolve()

                # Generate core programs
                for w in self._rt.workers:
                    w.resolve()

                # In/Out Sequence
                self._rt.resolve()

            self._print_verify(ctx)
            return ctx.module

    def _print_verify(self, ctx):
        verify = ctx.module.operation.verify()
        if verify != True:
            print(verify)
