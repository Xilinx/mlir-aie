"""
TODO: 
* docs
* types
* producer/consumer
* join
"""

from typing import Optional

from ... import ir
from ..phys.tile import MyTile
from .objectfifo import ObjectFifoHandle
from .endpoint import MyObjectFifoEndpoint
from ...dialects.aiex import NpuDmaMemcpyNd


class MyNpuDmaMemcpyNd(MyObjectFifoEndpoint):
    def __init__(
        self,
        of: ObjectFifoHandle,
        bd_id,
        mem,
        coords: tuple[int, int] = (0, 0),
        offsets: list[int] = None,
        sizes: list[int] = None,
        strides: list[int] = None,
        issue_token: Optional[bool] = None,
    ):
        self.__of = of
        self.__bd_id = bd_id
        self.__mem = mem
        self.__tile = MyTile(*coords)
        self.__offsets = offsets
        self.__sizes = sizes
        self.__strides = strides
        self.__issue_token = issue_token
        self.__op: Optional[NpuDmaMemcpyNd] = None
        self.__of.set_endpoint(self.__tile)

    @property
    def tile(self) -> MyTile:
        return self.__tile

    def resolve(
        self,
        loc: ir.Location = None,
        ip: ir.InsertionPoint = None,
    ) -> None:
        if self.__op == None:
            self.__op = NpuDmaMemcpyNd(
                self.__of.name,
                self.__bd_id,
                self.__mem,
                self.__offsets,
                self.__sizes,
                self.__strides,
                self.__issue_token,
            )
