"""
TODO: 
* docs
* types
* logical?
"""

from ..extras.context import mlir_mod_ctx
from ..extras.dialects.ext.func import FuncBase
from ..dialects.aie import device

from .worker import Worker
from .phys.device import Device
from .io.coordinator import IOCoordinator


class Program:
    def __init__(
        self,
        device: Device,
        worker_programs: list[Worker],
        io_coordinator: IOCoordinator,
    ):
        self.__device = device
        self.__worker_programs = worker_programs
        self.__io_coordinator = io_coordinator

    def resolve_program(self, generate_placement: bool = False):
        with mlir_mod_ctx() as ctx:

            @device(self.__device.resolve())
            def device_body():
                # Collect all fifos
                all_fifos = set()
                all_fifos.update(self.__io_coordinator.get_fifos())
                for w in self.__worker_programs:
                    all_fifos.update(w.get_fifos())

                # Collect all tiles
                all_tiles = set()
                for w in self.__worker_programs:
                    all_tiles.add(w.tile)
                for f in all_fifos:
                    all_tiles.add(f.end1_tile())
                    all_tiles.update(f.end2_tiles())

                # Resolve tiles
                for t in all_tiles:
                    self.__device.resolve_tile(t)
                    self._print_verify(ctx)

                # Generate fifos
                for f in all_fifos:
                    f.resolve()
                    self._print_verify(ctx)

                # generate functions - this may call resolve() more than once on the same fifo, but that's ok
                for w in self.__worker_programs:
                    for arg in w.fn_args:
                        if isinstance(arg, FuncBase):
                            arg.emit()
                        else:
                            arg.resolve()
                        self._print_verify(ctx)

                # In/Out Sequence
                self.__io_coordinator.resolve()
                self._print_verify(ctx)

                # Generate core programs
                for w in self.__worker_programs:
                    w.resolve()
                    self._print_verify(ctx)

            print(ctx.module)

    def _print_verify(self, ctx):
        verify = ctx.module.operation.verify()
        if verify != True:
            print(verify)
