from abc import ABCMeta, abstractmethod

from .phys.device import Device
from .io.iocoordinator import IOCoordinator
from .worker import Worker
from .phys.tile import AnyComputeTile, AnyShimTile, AnyMemTile
from .dataflow.objectfifo import ObjectFifoHandle


class Placer(meta=ABCMeta):
    @abstractmethod
    def make_placement(
        self,
        device: Device,
        io: IOCoordinator,
        workers: list[Worker],
        object_fifos: list[ObjectFifoHandle],
    ): ...


class SequentialPlacer(Placer):

    def make_placement(
        self,
        device: Device,
        io: IOCoordinator,
        workers: list[Worker],
        object_fifos: list[ObjectFifoHandle],
    ):
        shims = self.__device.get_shim_tiles()

        mems = self.__device.get_mem_tiles()
        mem_idx = 0  # Loop over memtiles

        cores = self.__device.get_core_tiles()
        core_idx = 0  # Will not loop over core tiles

        io.place_tasks(shims)
        for worker in workers:
            if worker.tile == AnyComputeTile:
                assert core_idx < len(cores), "Ran out of cores for placement!"
                worker.place(cores[core_idx])
                core_idx += 1

        for of in object_fifos:
            of_endpoints = [of.end1] + of.end2
            for ofe in of_endpoints:
                if ofe.tile == AnyMemTile:
                    ofe.place(mems[mem_idx])
                    mem_idx = (mem_idx + 1) % len(mems)
                elif ofe.tile == AnyComputeTile:
                    assert core_idx < len(cores), "Ran out of cores for placement!"
                    ofe.place(cores[core_idx])
                    core_idx += 1
