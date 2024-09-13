"""
TODO: 
* docs
* types
* logical?
"""

from ..extras.context import mlir_mod_ctx
from ..dialects.aie import device
from .worker import MyWorker
from .phys.device import MyDevice
from .dataflow.inout.inout import InOutProgram


class MyProgram:
    def __init__(
        self,
        device: MyDevice,
        worker_programs: list[MyWorker],
        inout_program: InOutProgram,
    ):
        assert isinstance(device, MyDevice)
        assert worker_programs != None and len(worker_programs) >= 1
        for w in worker_programs:
            assert isinstance(w, MyWorker)
        assert isinstance(inout_program, InOutProgram)
        self.__device = device
        self.__worker_programs = worker_programs
        self.__inout_program = inout_program

    def resolve_program(self):
        with mlir_mod_ctx() as ctx:

            @device(self.__device.resolve())
            def device_body():
                # generate tiles
                for w in self.__worker_programs:
                    w.tile.resolve()  # TODO: should maybe be part of endpoint trait?
                self._print_verify(ctx)

                self.__inout_program.tile.resolve()
                self._print_verify(ctx)

                # generate fifos (and external functions)
                for w in self.__worker_programs:
                    for arg in w.fn_args:
                        arg.resolve()
                        # self._print_verify(ctx)

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
