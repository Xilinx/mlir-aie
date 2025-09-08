# program.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 Advanced Micro Devices, Inc.

from ..extras.context import mlir_mod_ctx  # type: ignore
from ..helpers.dialects.ext.func import FuncBase
from ..dialects.aie import device, tile

from .device import Device
from .runtime import Runtime
from .placers import Placer
from .resolvable import Resolvable

# import aie.utils.trace as trace_utils
from ..utils import trace as trace_utils

import contextvars

CurrentDeviceOp = contextvars.ContextVar("CurrentDeviceOp", default=None)

CurrentModule = contextvars.ContextVar("CurrentModule", default=None)


class Program:
    def __init__(
        self,
        device: Device,
        rt: Runtime,
    ):
        """A Program represents all design information needed to run the design on a device.

        ctx.module.operation.verify() is called within this function to verify the correctness
        of the MLIR module.

        Args:
            device (Device): The device used to generate the final MLIR for the design.
            rt (Runtime): The runtime object for the design.
        """
        self._device = device
        self._rt = rt

    def resolve_program(self, placer: Placer | None = None, device_name = "main"):
        """This method resolves the program components in order to generate MLIR.

        Args:
            placer (Placer | None, optional): The placer that will assign placement to unplaced components.
                If a placer is not given, all components must be fully placed. Defaults to None.

        Returns:
            module (Module): The module containing the MLIR context information.
        """
        with mlir_mod_ctx() as ctx:
            CurrentModule.set(ctx.module)

            # Create a fresh device instance of the same type to avoid stale MLIR operations
            # This preserves the device configuration while ensuring clean state
            device_type = type(self._device)
            # For dynamically created device classes, the constructor takes no arguments
            self._device = device_type()

            @device(self._device.resolve(), sym_name = device_name)
            def device_body():
                # Collect all fifos
                all_fifos = set()
                all_fifos.update(self._rt.fifos)
                for w in self._rt.workers:
                    all_fifos.update(w.fifos)

                # Sort fifos for deterministic resolve
                all_fifos = sorted(all_fifos, key=lambda obj: obj.name)

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

                # Generate trace routes
                # TODO Need to iterate over all tiles or workers & fifos to make list of tiles to trace
                #      Alternatively, we merge the mechanism for packet routed objfifos so we use unique
                #      route IDs for trace as well

                # Scan workers and build list of tiles to trace
                tiles_to_trace = []
                if self._rt._trace_workers is not None:
                    for w in self._rt._trace_workers:
                        tiles_to_trace.append(w.tile.op)
                else:
                    for w in self._rt._workers:
                        if w.trace is not None:
                            tiles_to_trace.append(w.tile.op)
                if self._rt._trace_size is not None:
                    trace_shim_tile = self._rt.get_first_cons_shimtile()
                    trace_utils.configure_packet_tracing_flow(
                        tiles_to_trace, trace_shim_tile
                    )

                # In/Out Sequence
                self._rt.resolve()

            device_op = device_body
            CurrentDeviceOp.set(device_op)

            self._print_verify(ctx)
            return ctx.module

    def _print_verify(self, ctx):
        verify = ctx.module.operation.verify()
        if verify != True:
            print(verify)
