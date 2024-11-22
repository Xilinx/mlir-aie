from abc import ABCMeta, abstractmethod

from .phys.device import Device
from .io.iocoordinator import IOCoordinator
from .io.ioendpoint import IOEndpoint
from .worker import Worker
from .phys.tile import AnyComputeTile, AnyMemTile
from .dataflow.objectfifo import ObjectFifoHandle


class Placer(metaclass=ABCMeta):

    @abstractmethod
    def make_placement(
        self,
        device: Device,
        io: IOCoordinator,
        workers: list[Worker],
        object_fifos: list[ObjectFifoHandle],
    ): ...


class SequentialPlacer(Placer):

    def __init__(self):
        super().__init__()

    def make_placement(
        self,
        device: Device,
        io: IOCoordinator,
        workers: list[Worker],
        object_fifos: list[ObjectFifoHandle],
    ):
        shims = device.get_shim_tiles()

        mems = device.get_mem_tiles()
        mem_idx = 0  # Loop over memtiles

        computes = device.get_compute_tiles()
        compute_idx = 0  # Will not loop over core tiles

        for worker in workers:
            if worker.tile == AnyComputeTile:
                assert compute_idx < len(
                    computes
                ), "Ran out of compute tiles for placement!"
                worker.place(computes[compute_idx])
                compute_idx += 1

        for of in object_fifos:
            of_endpoints = of.get_all_endpoints()
            # IOEndpoints are placed by the IOCoordinator
            of_endpoints = [of for of in of_endpoints if not isinstance(of, IOEndpoint)]
            for ofe in of_endpoints:
                if ofe.tile == AnyMemTile:
                    ofe.place(mems[mem_idx])
                    mem_idx = (mem_idx + 1) % len(mems)
                elif ofe.tile == AnyComputeTile:
                    assert compute_idx < len(
                        computes
                    ), "Ran out of compute tiles for placement!"
                    ofe.place(computes[compute_idx])
                    compute_idx += 1

        io.place_tasks(shims)
