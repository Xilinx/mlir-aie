"""
TODO: 
* docs
* types
* join/distribute
"""

# Address circular dependency between ObjectFifo and ObjectFifoHandle
from __future__ import annotations
import numpy as np

from ... import ir  # type: ignore
from ...dialects._aie_enum_gen import ObjectFifoPort  # type: ignore
from ...dialects._aie_ops_gen import ObjectFifoCreateOp  # type: ignore
from ...dialects.aie import object_fifo, object_fifo_link
from ...helpers.util import np_ndarray_type_to_memref_type, single_elem_or_list_to_list

from ..resolvable import Resolvable
from .endpoint import ObjectFifoEndpoint
from ..phys.tile import Tile


class ObjectFifo(Resolvable):
    __of_index = 0

    def __init__(
        self,
        depth: int,
        obj_type: type[np.ndarray],
        name: str | None = None,
        dimensionsToStream=None,
        dimensionsFromStreamPerConsumer=None,
    ):
        self.__depth = depth
        self.__obj_type = obj_type
        self.end1 = None
        self.end2 = []
        self.dimensionToStream = dimensionsToStream
        self.dimensionsFromStreamPerConsumer = dimensionsFromStreamPerConsumer

        if name is None:
            self.name = f"of{ObjectFifo.__get_index()}"
        else:
            self.name = name
        self.__op: ObjectFifoCreateOp | None = None
        self.__first: ObjectFifoHandle = ObjectFifoHandle(self, True)
        self.__second: ObjectFifoHandle = ObjectFifoHandle(self, False)

    @classmethod
    def __get_index(cls) -> int:
        idx = cls.__of_index
        cls.__of_index += 1
        return idx

    @property
    def depth(self) -> int:
        return self.__depth

    @property
    def op(self) -> ObjectFifoCreateOp:
        assert self.__op != None
        return self.__op

    @property
    def first(self) -> ObjectFifoHandle:
        return self.__first

    @property
    def second(self) -> ObjectFifoHandle:
        return self.__second

    def end1_tile(self) -> Tile | None:
        if self.end1 == None:
            return None
        return self.end1.tile

    def end2_tiles(self) -> list[Tile | None] | None:
        if self.end2 == []:
            return None
        return [e.tile for e in self.end2]

    @property
    def obj_type(self) -> type[np.ndarray]:
        return self.__obj_type

    def resolve(
        self,
        loc: ir.Location | None = None,
        ip: ir.InsertionPoint | None = None,
    ) -> None:
        if self.__op == None:
            tile1 = self.end1_tile()
            tiles2 = self.end2_tiles()
            assert tile1 != None
            assert tiles2 != None and len(tiles2) >= 1
            for t in tiles2:
                assert t != None

            self.__op = object_fifo(
                self.name,
                tile1.op,
                [t.op for t in tiles2],
                self.__depth,
                np_ndarray_type_to_memref_type(self.__obj_type),
                dimensionsToStream=self.dimensionToStream,
                dimensionsFromStreamPerConsumer=self.dimensionsFromStreamPerConsumer,
                loc=loc,
                ip=ip,
            )

            if isinstance(self.end1, ObjectFifoLink):
                self.end1.resolve()
            for e in self.end2:
                if isinstance(self.end2, ObjectFifoLink):
                    e.resolve()

    def _set_endpoint(self, endpoint: ObjectFifoEndpoint, first: bool = True) -> None:
        if first:
            assert (
                self.end1 == None or self.end1 == endpoint
            ), f"ObjectFifo already assigned endpoint 1 ({self.end1})"
            self.end1 = endpoint
        else:
            # TODO: need rules about shim tiles here
            self.end2.append(endpoint)

    def _acquire(
        self,
        port: ObjectFifoPort,
        num_elem: int,
        loc: ir.Location | None = None,
        ip: ir.InsertionPoint | None = None,
    ):
        assert num_elem > 0, "Must consume at least one element"
        assert (
            num_elem <= self.__depth
        ), "Cannot consume elements to exceed ObjectFifo depth"
        return self.op.acquire(port, num_elem)

    def _release(
        self,
        port: ObjectFifoPort,
        num_elem: int,
        loc: ir.Location | None = None,
        ip: ir.InsertionPoint | None = None,
    ):
        assert num_elem > 0, "Must consume at least one element"
        assert (
            num_elem <= self.__depth
        ), "Cannot consume elements to exceed ObjectFifo depth"
        self.op.release(port, num_elem)


class ObjectFifoHandle(Resolvable):
    def __init__(self, of: ObjectFifo, is_first: bool):
        self.__port: ObjectFifoPort = (
            ObjectFifoPort.Produce if is_first else ObjectFifoPort.Consume
        )
        self.__is_first = is_first
        self.__object_fifo = of

    def acquire(
        self,
        num_elem: int,
        loc: ir.Location | None = None,
        ip: ir.InsertionPoint | None = None,
    ):
        return self.__object_fifo._acquire(self.__port, num_elem, loc=loc, ip=ip)

    def release(
        self,
        num_elem: int,
        loc: ir.Location | None = None,
        ip: ir.InsertionPoint | None = None,
    ):
        return self.__object_fifo._release(self.__port, num_elem, loc=loc, ip=ip)

    @property
    def name(self) -> str:
        return self.__object_fifo.name

    @property
    def op(self) -> ObjectFifoCreateOp:
        return self.__object_fifo.op

    @property
    def obj_type(self) -> type[np.ndarray]:
        return self.__object_fifo.obj_type

    def end1_tile(self) -> Tile | None:
        return self.__object_fifo.end1_tile()

    def end2_tiles(self) -> list[Tile | None] | None:
        return self.__object_fifo.end2_tiles()

    def set_endpoint(self, endpoint: ObjectFifoEndpoint) -> None:
        self.__object_fifo._set_endpoint(endpoint, first=self.__is_first)

    def join(
        self,
        offsets: list[int],
        coords: tuple[int, int],
        depths: list[int] | None = None,
        types: list[type[np.ndarray]] = None,
        dimensions=None,
    ) -> list[ObjectFifo]:
        num_subfifos = len(offsets)
        if depths is None:
            depths = [self.__object_fifo.depth] * num_subfifos
        if isinstance(depths, list):
            assert len(depths) == len(offsets)

        if types is None:
            types = [self.__object_fifo.obj_type] * num_subfifos
        if isinstance(types, list):
            assert len(types) == len(offsets)

        # Create subfifos
        subfifos = []
        for i in range(num_subfifos):
            subfifos.append(
                ObjectFifo(
                    depths[i],
                    types[i],
                    name=self.__object_fifo.name + f"_join{i}",
                )
            )

        # Create link and set it as endpoints
        link = ObjectFifoLink(subfifos, self.__object_fifo, coords, offsets, [])
        self.set_endpoint(link)
        for i in range(num_subfifos):
            subfifos[i].second.set_endpoint(link)
        return subfifos

    def split(
        self,
        offsets: list[int],
        coords: tuple[int, int],
        depths: list[int] | None = None,
        types: list[type[np.ndarray]] = None,
        dimensions=None,
    ) -> list[ObjectFifo]:
        num_subfifos = len(offsets)
        if depths is None:
            depths = [self.__object_fifo.depth] * num_subfifos
        if isinstance(depths, list):
            assert len(depths) == len(offsets)

        if types is None:
            types = [self.__object_fifo.obj_type] * num_subfifos
        if isinstance(types, list):
            assert len(types) == len(offsets)

        # Create subfifos
        subfifos = []
        for i in range(num_subfifos):
            subfifos.append(
                ObjectFifo(
                    depths[i],
                    types[i],
                    name=self.__object_fifo.name + f"_split{i}",
                )
            )

        # Create link and set it as endpoints
        link = ObjectFifoLink(self.__object_fifo, subfifos, coords, [], offsets)
        self.set_endpoint(link)
        for i in range(num_subfifos):
            subfifos[i].first.set_endpoint(link)
        return subfifos

    def forward(self, coords: tuple[int, int], depth: int | None = None) -> ObjectFifo:
        assert not self.__is_first
        if depth is None:
            depth = self.__object_fifo.depth
        forward_fifo = ObjectFifo(
            depth, self.__object_fifo.obj_type, name=self.__object_fifo.name + f"_fwd"
        )
        link = ObjectFifoLink(self.__object_fifo, forward_fifo, coords, [], [])
        self.set_endpoint(link)
        forward_fifo.first.set_endpoint(link)
        return forward_fifo

    def resolve(
        self,
        loc: ir.Location | None = None,
        ip: ir.InsertionPoint | None = None,
    ) -> None:
        self.__object_fifo.resolve(loc=loc, ip=ip)


class ObjectFifoLink(ObjectFifoEndpoint, Resolvable):
    def __init__(
        self,
        srcs: list[ObjectFifoHandle] | ObjectFifoHandle,
        dsts: list[ObjectFifoHandle] | ObjectFifoHandle,
        coords: tuple[int, int],
        src_offsets: list[int] = [],
        dst_offsets: list[int] = [],
    ):
        self.__srcs = single_elem_or_list_to_list(srcs)
        self.__dsts = single_elem_or_list_to_list(dsts)
        self.__src_offsets = src_offsets
        self.__dst_offsets = dst_offsets

        assert len(self.__srcs) > 0 and len(self.__dsts) > 0
        assert len(self.__srcs) == 1 or len(self.__dsts) == 1
        if len(self.__src_offsets) > 0:
            assert len(self.__src_offsets) == len(self.__srcs)
        if len(self.__dst_offsets) > 0:
            assert len(self.__dst_offsets) == len(self.__dsts)

        self.__tile = None
        if coords:
            column, row = coords
            self.__tile = Tile(column, row)

        self.__op = None

    @property
    def tile(self) -> Tile | None:
        return self.__tile

    def resolve(
        self,
        loc: ir.Location | None = None,
        ip: ir.InsertionPoint | None = None,
    ) -> None:
        if self.__op == None:
            for s in self.__srcs:
                s.resolve()
            for d in self.__dsts:
                d.resolve()
            src_ops = [s.op for s in self.__srcs]
            dst_ops = [d.op for d in self.__dsts]
            self.__op = object_fifo_link(
                src_ops, dst_ops, self.__src_offsets, self.__dst_offsets
            )
