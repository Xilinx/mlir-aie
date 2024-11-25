"""
TODO: 
* docs
* types
* logical?
"""

from ..extras.context import mlir_mod_ctx
from ..extras.dialects.ext.func import FuncBase
from ..dialects.aie import device

from .phys.device import Device
from .runtime import Runtime
from .placers import Placer


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
                # In/Out Sequence
                self._rt.resolve()

                # Collect all fifos
                all_fifos = set()
                all_fifos.update(self._rt.get_fifos())
                for w in self._rt.get_workers():
                    all_fifos.update(w.get_fifos())

                workers = self._rt.get_workers()
                if placer:
                    # TODO: should maybe just take runtime?
                    placer.make_placement(self._device, self._rt, workers, all_fifos)

                # Collect all tiles
                all_tiles = []
                for w in workers:
                    all_tiles.append(w.tile)
                for f in all_fifos:
                    all_tiles.append(f.end_prod_tile())
                    all_tiles.extend(f.end_cons_tiles())

                # Resolve tiles
                for t in all_tiles:
                    self._device.resolve_tile(t)

                # Generate fifos
                for f in all_fifos:
                    f.resolve()

                # generate functions - this may call resolve() more than once on the same fifo, but that's ok
                for w in workers:
                    for arg in w.fn_args:
                        if isinstance(arg, FuncBase):
                            arg.emit()
                        else:
                            arg.resolve()

                # Generate core programs
                for w in workers:
                    w.resolve()

            self._print_verify(ctx)
            return ctx.module

    def _print_verify(self, ctx):
        verify = ctx.module.operation.verify()
        if verify != True:
            print(verify)
