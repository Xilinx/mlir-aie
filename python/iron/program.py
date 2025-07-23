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

    def _create_fresh_components(self):
        """Create fresh instances of device and runtime to avoid state sharing.

        Returns:
            tuple: (fresh_device, fresh_runtime) with copied state but fresh MLIR contexts
        """
        # Create a fresh device instance of the same type
        device_class = type(self._device)
        fresh_device = device_class()

        # Create a fresh runtime instance
        fresh_rt = Runtime()

        # Copy the runtime state that doesn't involve MLIR operations
        fresh_rt._trace_size = self._rt._trace_size
        fresh_rt._trace_offset = self._rt._trace_offset
        fresh_rt._trace_workers = self._rt._trace_workers
        fresh_rt.ddr_id = self._rt.ddr_id

        # Copy workers (they will be re-resolved with fresh MLIR context)
        fresh_rt._workers = self._rt._workers.copy()
        fresh_rt._fifos = self._rt._fifos.copy()
        fresh_rt._tasks = self._rt._tasks.copy()
        fresh_rt._open_task_groups = self._rt._open_task_groups.copy()

        # Copy the runtime data list
        fresh_rt._rt_data = self._rt._rt_data.copy()

        return fresh_device, fresh_rt

    def resolve_program(self, placer: Placer | None = None):
        """This method resolves the program components in order to generate MLIR.

        Args:
            placer (Placer | None, optional): The placer that will assign placement to unplaced components.
                If a placer is not given, all components must be fully placed. Defaults to None.

        Returns:
            module (Module): The module containing the MLIR context information.
        """
        # Create fresh components to avoid state sharing issues
        fresh_device, fresh_rt = self._create_fresh_components()

        with mlir_mod_ctx() as ctx:

            @device(fresh_device.resolve())
            def device_body():
                # Collect all fifos
                all_fifos = set()
                all_fifos.update(fresh_rt.fifos)
                for w in fresh_rt.workers:
                    all_fifos.update(w.fifos)

                if placer:
                    # TODO: should maybe just take runtime?
                    placer.make_placement(
                        fresh_device, fresh_rt, fresh_rt.workers, all_fifos
                    )

                # Collect all tiles
                all_tiles = []
                for w in fresh_rt.workers:
                    all_tiles.append(w.tile)
                for f in all_fifos:
                    all_tiles.extend([e.tile for e in f.all_of_endpoints()])

                # Resolve tiles
                for t in all_tiles:
                    fresh_device.resolve_tile(t)

                # Generate fifos
                for f in all_fifos:
                    f.resolve()

                # generate functions - this may call resolve() more than once on the same fifo, but that's ok
                for w in fresh_rt.workers:
                    for arg in w.fn_args:
                        if isinstance(arg, FuncBase):
                            arg.emit()
                        elif isinstance(arg, Resolvable):
                            arg.resolve()

                # Generate core programs
                for w in fresh_rt.workers:
                    w.resolve()

                # Generate trace routes
                # TODO Need to iterate over all tiles or workers & fifos to make list of tiles to trace
                #      Alternatively, we merge the mechanism for packet routed objfifos so we use unique
                #      route IDs for trace as well

                # Scan workers and build list of tiles to trace
                tiles_to_trace = []
                if fresh_rt._trace_workers is not None:
                    for w in fresh_rt._trace_workers:
                        tiles_to_trace.append(w.tile.op)
                else:
                    for w in fresh_rt._workers:
                        if w.trace is not None:
                            tiles_to_trace.append(w.tile.op)
                if fresh_rt._trace_size is not None:
                    trace_shim_tile = fresh_rt.get_first_cons_shimtile()
                    trace_utils.configure_packet_tracing_flow(
                        tiles_to_trace, trace_shim_tile
                    )

                # In/Out Sequence
                fresh_rt.resolve()

            self._print_verify(ctx)
            return ctx.module

    def _print_verify(self, ctx):
        verify = ctx.module.operation.verify()
        if verify != True:
            print(verify)
