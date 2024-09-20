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
from .dataflow.inout.inout import InOutProgram
from .dataflow.objectfifolink import MyObjectFifoLink


class MyProgram:
    def __init__(
        self,
        device: MyDevice,
        worker_programs: list[MyWorker],
        inout_program: InOutProgram,
        links: list[MyObjectFifoLink] = [],
    ):
        self.__device = device
        self.__worker_programs = worker_programs
        self.__inout_program = inout_program
        self.__links = links

    def resolve_program(self):
        with mlir_mod_ctx() as ctx:

            @device(self.__device.resolve())
            def device_body():
                # generate tiles
                for w in self.__worker_programs:
                    self.__device.resolve_tile(w.tile)
                for l in self.__links:
                    self.__device.resolve_tile(l.tile)
                self._print_verify(ctx)

                self.__device.resolve_tile(self.__inout_program.tile)
                self._print_verify(ctx)

                # generate fifos (and external functions)
                for w in self.__worker_programs:
                    for arg in w.fn_args:
                        if isinstance(arg, FuncBase):
                            arg.emit()
                        else:
                            arg.resolve()
                        self._print_verify(ctx)
                for f in self.__inout_program.get_fifos():
                    f.resolve()
                    self._print_verify(ctx)

                # Generate object fifo links
                for l in self.__links:
                    l.resolve()
                    self._print_verify(ctx)

                # Generate core programs
                for w in self.__worker_programs:
                    w.resolve()
                    self._print_verify(ctx)

                # Host program
                self.__inout_program.resolve()

                self._print_verify(ctx)

            print(ctx.module)

    def _print_verify(self, ctx):
        verify = ctx.module.operation.verify()
        if verify != True:
            print(verify)
