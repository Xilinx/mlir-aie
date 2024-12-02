# objectfifo.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 Advanced Micro Devices, Inc.
from __future__ import annotations
import numpy as np
from typing import Sequence

from ... import ir  # type: ignore
from ...dialects._aie_enum_gen import ObjectFifoPort  # type: ignore
from ...dialects._aie_ops_gen import ObjectFifoCreateOp  # type: ignore
from ...dialects.aie import object_fifo, object_fifo_link
from ...helpers.util import (
    np_ndarray_type_to_memref_type,
    single_elem_or_list_to_list,
    np_ndarray_type_get_dtype,
    np_ndarray_type_get_shape,
)

from ..resolvable import Resolvable
from .endpoint import ObjectFifoEndpoint
from ..phys.tile import PlacementTile, AnyMemTile, Tile


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
        self._depth = depth
        self._obj_type = obj_type
        self.end_prod = None
        self.end_cons = []
        self.dimensionToStream = dimensionsToStream
        self.dimensionsFromStreamPerConsumer = dimensionsFromStreamPerConsumer

        if name is None:
            self.name = f"of{ObjectFifo.__get_index()}"
        else:
            self.name = name
        self._op: ObjectFifoCreateOp | None = None
        self._prod: ObjectFifoHandle = ObjectFifoHandle(self, True)
        self._cons: ObjectFifoHandle = ObjectFifoHandle(self, False)

    @classmethod
    def __get_index(cls) -> int:
        idx = cls.__of_index
        cls.__of_index += 1
        return idx

    @property
    def depth(self) -> int:
        return self._depth

    @property
    def op(self) -> ObjectFifoCreateOp:
        assert self._op != None
        return self._op

    @property
    def shape(self) -> Sequence[int]:
        return np_ndarray_type_get_shape(self._obj_type)

    @property
    def dtype(self) -> np.dtype:
        return np_ndarray_type_get_dtype(self._obj_type)

    @property
    def prod(self) -> ObjectFifoHandle:
        return self._prod

    @property
    def cons(self) -> ObjectFifoHandle:
        return self._cons

    def end_prod_tile(self) -> PlacementTile | None:
        if self.end_prod == None:
            return None
        return self.end_prod.tile

    def end_cons_tiles(self) -> list[PlacementTile | None] | None:
        if self.end_cons == []:
            return None
        return [e.tile for e in self.end_cons]

    def _get_endpoint(self, is_first: bool) -> list[ObjectFifoEndpoint]:
        if is_first:
            return [self.end_prod]
        else:
            return self.end_cons.copy()

    @property
    def obj_type(self) -> type[np.ndarray]:
        return self._obj_type

    def resolve(
        self,
        loc: ir.Location | None = None,
        ip: ir.InsertionPoint | None = None,
    ) -> None:
        if self._op == None:
            tile1 = self.end_prod_tile()
            tiles2 = self.end_cons_tiles()
            assert tile1 != None
            assert tiles2 != None and len(tiles2) >= 1
            for t in tiles2:
                assert t != None

            self._op = object_fifo(
                self.name,
                tile1.op,
                [t.op for t in tiles2],
                self._depth,
                np_ndarray_type_to_memref_type(self._obj_type),
                dimensionsToStream=self.dimensionToStream,
                dimensionsFromStreamPerConsumer=self.dimensionsFromStreamPerConsumer,
                loc=loc,
                ip=ip,
            )

            if isinstance(self.end_prod, ObjectFifoLink):
                self.end_prod.resolve()
            for e in self.end_cons:
                if isinstance(self.end_cons, ObjectFifoLink):
                    e.resolve()

    def _set_endpoint(self, endpoint: ObjectFifoEndpoint, first: bool = True) -> None:
        if first:
            assert (
                self.end_prod == None or self.end_prod == endpoint
            ), f"ObjectFifo already assigned endpoint 1 ({self.end_prod})"
            self.end_prod = endpoint
        else:
            # TODO: need rules about shim tiles here
            self.end_cons.append(endpoint)

    def _acquire(
        self,
        port: ObjectFifoPort,
        num_elem: int,
        loc: ir.Location | None = None,
        ip: ir.InsertionPoint | None = None,
    ):
        assert num_elem > 0, "Must consume at least one element"
        assert (
            num_elem <= self._depth
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
            num_elem <= self._depth
        ), "Cannot consume elements to exceed ObjectFifo depth"
        self.op.release(port, num_elem)


class ObjectFifoHandle(Resolvable):
    def __init__(self, of: ObjectFifo, is_first: bool):
        self._port: ObjectFifoPort = (
            ObjectFifoPort.Produce if is_first else ObjectFifoPort.Consume
        )
        self._is_first = is_first
        self._object_fifo = of

    def acquire(
        self,
        num_elem: int,
        loc: ir.Location | None = None,
        ip: ir.InsertionPoint | None = None,
    ):
        return self._object_fifo._acquire(self._port, num_elem, loc=loc, ip=ip)

    def release(
        self,
        num_elem: int,
        loc: ir.Location | None = None,
        ip: ir.InsertionPoint | None = None,
    ):
        return self._object_fifo._release(self._port, num_elem, loc=loc, ip=ip)

    @property
    def name(self) -> str:
        return self._object_fifo.name

    @property
    def op(self) -> ObjectFifoCreateOp:
        return self._object_fifo.op

    @property
    def obj_type(self) -> type[np.ndarray]:
        return self._object_fifo.obj_type

    @property
    def shape(self) -> Sequence[int]:
        return self._object_fifo.shape

    @property
    def dtype(self) -> np.dtype:
        return self._oject_fifo.dtype

    def end_prod_tile(self) -> Tile | None:
        return self._object_fifo.end_prod_tile()

    def end_cons_tiles(self) -> list[Tile | None] | None:
        return self._object_fifo.end_cons_tiles()

    def set_endpoint(self, endpoint: ObjectFifoEndpoint) -> None:
        self._object_fifo._set_endpoint(endpoint, first=self._is_first)

    def get_endpoint(self) -> list[ObjectFifoEndpoint]:
        return self._object_fifo._get_endpoint(is_first=self._is_first)

    def get_all_endpoints(self) -> list[ObjectFifoEndpoint]:
        return self._object_fifo._get_endpoint(
            is_first=True
        ) + self._object_fifo._get_endpoint(is_first=False)

    def join(
        self,
        offsets: list[int],
        placement: PlacementTile = AnyMemTile,
        depths: list[int] | None = None,
        types: list[type[np.ndarray]] = None,
        dimensions=None,
    ) -> list[ObjectFifo]:
        num_subfifos = len(offsets)
        if depths is None:
            depths = [self._object_fifo.depth] * num_subfifos
        if isinstance(depths, list):
            assert len(depths) == len(offsets)

        if types is None:
            types = [self._object_fifo.obj_type] * num_subfifos
        if isinstance(types, list):
            assert len(types) == len(offsets)

        # Create subfifos
        subfifos = []
        for i in range(num_subfifos):
            subfifos.append(
                ObjectFifo(
                    depths[i],
                    types[i],
                    name=self._object_fifo.name + f"_join{i}",
                )
            )

        # Create link and set it as endpoints
        link = ObjectFifoLink(subfifos, self._object_fifo, placement, offsets, [])
        self.set_endpoint(link)
        for i in range(num_subfifos):
            subfifos[i].cons.set_endpoint(link)
        return subfifos

    def split(
        self,
        offsets: list[int],
        placement: PlacementTile = AnyMemTile,
        depths: list[int] | None = None,
        types: list[type[np.ndarray]] = None,
        dimensions=None,
    ) -> list[ObjectFifo]:
        num_subfifos = len(offsets)
        if depths is None:
            depths = [self._object_fifo.depth] * num_subfifos
        if isinstance(depths, list):
            assert len(depths) == len(offsets)

        if types is None:
            types = [self._object_fifo.obj_type] * num_subfifos
        if isinstance(types, list):
            assert len(types) == len(offsets)

        # Create subfifos
        subfifos = []
        for i in range(num_subfifos):
            subfifos.append(
                ObjectFifo(
                    depths[i],
                    types[i],
                    name=self._object_fifo.name + f"_split{i}",
                )
            )

        # Create link and set it as endpoints
        link = ObjectFifoLink(self._object_fifo, subfifos, placement, [], offsets)
        self.set_endpoint(link)
        for i in range(num_subfifos):
            subfifos[i].prod.set_endpoint(link)
        return subfifos

    def forward(
        self,
        placement: PlacementTile = AnyMemTile,
        obj_type: type[np.ndarray] | None = None,
        depth: int | None = None,
    ) -> ObjectFifo:
        assert not self._is_first
        if obj_type is None:
            obj_type = self._object_fifo.obj_type
        if depth is None:
            depth = self._object_fifo.depth
        forward_fifo = ObjectFifo(
            depth, obj_type, name=self._object_fifo.name + f"_fwd"
        )
        link = ObjectFifoLink(self._object_fifo, forward_fifo, placement, [], [])
        self.set_endpoint(link)
        forward_fifo.prod.set_endpoint(link)
        return forward_fifo

    def resolve(
        self,
        loc: ir.Location | None = None,
        ip: ir.InsertionPoint | None = None,
    ) -> None:
        self._object_fifo.resolve(loc=loc, ip=ip)


class ObjectFifoLink(ObjectFifoEndpoint, Resolvable):
    def __init__(
        self,
        srcs: list[ObjectFifoHandle] | ObjectFifoHandle,
        dsts: list[ObjectFifoHandle] | ObjectFifoHandle,
        placement: PlacementTile = AnyMemTile,
        src_offsets: list[int] = [],
        dst_offsets: list[int] = [],
    ):
        self._srcs = single_elem_or_list_to_list(srcs)
        self._dsts = single_elem_or_list_to_list(dsts)
        self._src_offsets = src_offsets
        self._dst_offsets = dst_offsets

        assert len(self._srcs) > 0 and len(self._dsts) > 0
        assert len(self._srcs) == 1 or len(self._dsts) == 1
        if len(self._src_offsets) > 0:
            assert len(self._src_offsets) == len(self._srcs)
        if len(self._dst_offsets) > 0:
            assert len(self._dst_offsets) == len(self._dsts)

        self._tile = placement
        self._op = None

    @property
    def tile(self) -> PlacementTile:
        return self._tile

    def place(self, tile: Tile) -> None:
        assert not isinstance(
            self._tile, Tile
        ), f"Worker already placed at {self._tile}, cannot place {tile}"
        self._tile = tile

    def resolve(
        self,
        loc: ir.Location | None = None,
        ip: ir.InsertionPoint | None = None,
    ) -> None:
        if self._op == None:
            for s in self._srcs:
                s.resolve()
            for d in self._dsts:
                d.resolve()
            src_ops = [s.op for s in self._srcs]
            dst_ops = [d.op for d in self._dsts]
            self._op = object_fifo_link(
                src_ops, dst_ops, self._src_offsets, self._dst_offsets
            )
