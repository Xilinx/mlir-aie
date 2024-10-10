"""
TODO: 
* docs
* types
* logical?
"""

from ..extras.context import mlir_mod_ctx
from ..extras.dialects.ext.func import FuncBase
from ..dialects.aie import device

from .worker import MyWorker
from .phys.device import MyDevice
from .dataflow.inout.inout import InOutSequence
from .dataflow.objectfifolink import MyObjectFifoLink


class MyProgram:
    def __init__(
        self,
        device: MyDevice,
        worker_programs: list[MyWorker],
        inout_sequence: InOutSequence,
        links: list[MyObjectFifoLink] = [],
    ):
        self.__device = device
        self.__worker_programs = worker_programs
        self.__inout_sequence = inout_sequence
        self.__links = links

    def resolve_program(self):
        with mlir_mod_ctx() as ctx:

            @device(self.__device.resolve())
            def device_body():
                # Collect all fifos
                all_fifos = self.__inout_sequence.get_fifos()
                for w in self.__worker_programs:
                    all_fifos.extend(w.get_fifos())

                # Collect all tiles
                my_tiles = set()
                for w in self.__worker_programs:
                    my_tiles.add(w.tile)
                for l in self.__links:
                    my_tiles.add(l.tile)
                for f in all_fifos:
                    my_tiles.add(f.end1_tile())
                    my_tiles.update(f.end2_tiles())

                # Resolve tiles
                for t in my_tiles:
                    self.__device.resolve_tile(t)
                    self._print_verify(ctx)

                # Generate fifos
                for f in all_fifos:
                    f.resolve()
                    self._print_verify(ctx)

                # Generate object fifo links
                for l in self.__links:
                    l.resolve()
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
                self.__inout_sequence.resolve()
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
