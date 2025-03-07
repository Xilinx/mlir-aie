# worker.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 Advanced Micro Devices, Inc.
import contextvars
import sys
from typing import Callable

from .. import ir  # type: ignore
from ..dialects.aie import core
from ..helpers.dialects.ext.scf import _for as range_
from .device import PlacementTile, AnyComputeTile, Tile
from .dataflow.objectfifo import ObjectFifoHandle, ObjectFifo
from .dataflow.endpoint import ObjectFifoEndpoint
from .kernel import Kernel
from .globalbuffer import GlobalBuffer


class Worker(ObjectFifoEndpoint):
    """_summary_
    Worker is an object that takes a `core_fn` and a set of arguments.
    A Worker must be placed on a Compute Core.
    """

    """This variable is the current core if resolving() within the Worker, or None otherwise."""
    current_core_placement = contextvars.ContextVar(
        "current_core_placement", default=None
    )

    def __init__(
        self,
        core_fn: Callable | None,
        fn_args: list = [],
        placement: PlacementTile | None = AnyComputeTile,
        while_true: bool = True,
        stack_size: int = None
    ):
        """Construct a Worker

        Args:
            core_fn (Callable | None): The task to run on a core. If None, a busy-loop (`while(true): pass`) core will be generated.
            fn_args (list, optional): Pointers to arguments, which should include all context the core_fn needs to run. Defaults to [].
            placement (PlacementTile | None, optional): The placement for the Worker. Defaults to AnyComputeTile.
            while_true (bool, optional): If true, will wrap the core_fn in a while(true) loop to ensure it runs until reconfiguration. Defaults to True.
            stack_size (int, optional): The stack_size in bytes to be allocated for the worker. Defaults to 1024 bytes.

        Raises:
            ValueError: Parameters are validated.
        """
        self._tile = placement
        self._while_true = while_true
        self.stack_size = stack_size

        # If no core_fn is given, make a simple while(true) loop.
        if core_fn is None:

            def do_nothing_core_fun(*args) -> None:
                for _ in range_(sys.maxsize):
                    pass

            self.core_fn = do_nothing_core_fun
        else:
            self.core_fn = core_fn
        self.link_with: str | None = None
        self.fn_args = fn_args
        bin_names = set()
        self._fifos = []
        self._buffers = []

        # Check arguments to the core. Some information is saved for resolution.
        for arg in self.fn_args:
            if isinstance(arg, Kernel):
                bin_names.add(arg.bin_name)
            elif isinstance(arg, ObjectFifoHandle):
                arg.endpoint = self
                self._fifos.append(arg)
            elif isinstance(arg, GlobalBuffer):
                self._buffers.append(arg)
            elif isinstance(arg, ObjectFifo):
                # This is an easy error to make, so we catch it early
                raise ValueError(
                    "Cannot give an ObjectFifo directly to a worker; "
                    "must give an ObjectFifoHandle obtained through "
                    "ObjectFifo.prod() or ObjectFifo.cons()"
                )
            # We assume other arguments are metaprogramming (e.g, Python args)
            # This could allow some errors to sink through, but we allow it for now.
            # TODO: this could be cleaned up through creation of a MetaArgs struct, so you
            # could access values through meta.my_var within the function.

        if len(bin_names) > 1:
            raise ValueError(
                f"Currently, only one binary per works is supported. Found: {bin_names}"
            )
        if len(bin_names) == 1:
            self.link_with = list(bin_names)[0]

    def place(self, tile: Tile) -> None:
        """Set the placement of the Worker.

        Args:
            tile (Tile): The placement location.
        """
        ObjectFifoEndpoint.place(self, tile)

    @property
    def fifos(self) -> list[ObjectFifoHandle]:
        """Returns a list of ObjectFifoHandles given to the Worker via fn_args.

        Returns:
            list[ObjectFifoHandle]: ObjectFifoHandles used by the Worker.
        """
        return self._fifos.copy()

    @property
    def buffers(self) -> list[GlobalBuffer]:
        """Returns a list of GlobalBuffers given to the Worker via fn_args.

        Returns:
            list[GlobalBuffer]: GlobalBuffers used by the Worker.
        """
        return self._buffers.copy()

    def resolve(
        self,
        loc: ir.Location | None = None,
        ip: ir.InsertionPoint | None = None,
    ) -> None:
        if not self._tile:
            raise ValueError("Must place Worker before it can be resolved.")
        my_tile = self._tile.op
        my_link = self.link_with

        # Set the current_core_placement context variable to the current placement.
        # If there are objects within a core_fn that that need a placement, they can
        # query this value, e.g., Worker.current_core_placement.get()
        self.current_core_placement.set(my_tile)

        @core(my_tile, link_with=my_link, stack_size=self.stack_size)
        def core_body():
            for _ in range_(sys.maxsize) if self._while_true else range(1):
                self.core_fn(*self.fn_args)

        # Once we are done resolving the core, remove the placement context information
        self.current_core_placement.set(None)
