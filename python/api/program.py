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
from .io.iocoordinator import IOCoordinator
from .placers import Placer


class Program:
    def __init__(
        self,
        device: Device,
        io_coordinator: IOCoordinator,
        workers: list[Worker] = [],
    ):
        self._device = device
        self._workers = workers
        self._io_coordinator = io_coordinator

    def resolve_program(self, placer: Placer | None = None):
        with mlir_mod_ctx() as ctx:

            @device(self._device.resolve())
            def device_body():
                # Collect all fifos
                all_fifos = set()
                all_fifos.update(self._io_coordinator.get_fifos())
                for w in self._workers:
                    all_fifos.update(w.get_fifos())

                if placer:
                    placer.make_placement(
                        self._device, self._io_coordinator, self._workers, all_fifos
                    )

                # Collect all tiles
                all_tiles = []
                for w in self._workers:
                    all_tiles.append(w.tile)
                for f in all_fifos:
                    all_tiles.append(f.end1_tile())
                    all_tiles.extend(f.end2_tiles())

                # Resolve tiles
                for t in all_tiles:
                    self._device.resolve_tile(t)
                    self._print_verify(ctx)

                # Generate fifos
                for f in all_fifos:
                    f.resolve()
                    self._print_verify(ctx)

                # generate functions - this may call resolve() more than once on the same fifo, but that's ok
                for w in self._workers:
                    for arg in w.fn_args:
                        if isinstance(arg, FuncBase):
                            arg.emit()
                        else:
                            arg.resolve()
                        self._print_verify(ctx)

                # Generate core programs
                for w in self._workers:
                    w.resolve()
                    self._print_verify(ctx)

                # In/Out Sequence
                self._io_coordinator.resolve()
                self._print_verify(ctx)

            print(ctx.module)

    def _print_verify(self, ctx):
        verify = ctx.module.operation.verify()
        if verify != True:
            print(verify)
