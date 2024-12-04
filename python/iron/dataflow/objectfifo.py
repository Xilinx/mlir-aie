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

from ..resolvable import Resolvable, NotResolvedError
from .endpoint import ObjectFifoEndpoint
from ..phys.tile import PlacementTile, AnyMemTile, Tile


class ObjectFifo(Resolvable):
    __of_index = 0

    def __init__(
        self,
        obj_type: type[np.ndarray],
        default_depth: int | None = 2,
        name: str | None = None,
        dimensionsToStream=None,
        dimensionsFromStreamPerConsumer=None,
    ):
        self._default_depth = default_depth
        if isinstance(self._default_depth, int) and self._default_depth < 1:
            raise ValueError(
                f"Default ObjectFifo depth must be > 0, but got {self._default_depth}"
            )
        self._obj_type = obj_type
        self._dimensionToStream = dimensionsToStream
        self._dimensionsFromStreamPerConsumer = dimensionsFromStreamPerConsumer

        if name is None:
            self.name = f"of{ObjectFifo.__get_index()}"
        else:
            self.name = name
        self._op: ObjectFifoCreateOp | None = None
        self._prod: ObjectFifoHandle | None = None
        self._cons: list[ObjectFifoHandle] = []

    @classmethod
    def __get_index(cls) -> int:
        idx = cls.__of_index
        cls.__of_index += 1
        return idx

    @property
    def default_depth(self) -> int:
        return self._default_depth

    @property
    def op(self) -> ObjectFifoCreateOp:
        if self._op is None:
            raise NotResolvedError()
        return self._op

    @property
    def shape(self) -> Sequence[int]:
        return np_ndarray_type_get_shape(self._obj_type)

    @property
    def dtype(self) -> np.dtype:
        return np_ndarray_type_get_dtype(self._obj_type)

    def prod(self, depth: int | None = None) -> ObjectFifoHandle:
        if self._prod:
            if depth is None:
                if self._default_depth is None:
                    raise ValueError(
                        f"If default_depth is None, then depth must be specified."
                    )
                else:
                    depth = self._default_depth
            elif depth < 1:
                raise ValueError(f"Depth must be > 1, but got {depth}")
        else:
            self._prod = ObjectFifoHandle(self, True, depth)
        return self._prod

    def cons(self, depth: int | None = None) -> ObjectFifoHandle:
        if depth is None:
            if self._default_depth is None:
                raise ValueError(
                    f"If default_depth is None, then depth must be specified."
                )
            else:
                depth = self._default_depth
        self._cons.append(ObjectFifoHandle(self, False, depth))
        return self._cons[-1]

    def tiles(self) -> list[PlacementTile]:
        if self._prod == None:
            raise ValueError("Cannot return prod.tile.op because prod was not created.")
        if self._cons == []:
            raise ValueError("Cannot return cons.tile.op because prod was not created.")
        return [self._prod.tile] + [cons.tile for cons in self._cons]

    def _prod_tile_op(self) -> Tile:
        if self._prod == None:
            raise ValueError("Cannot return prod.tile.op because prod was not created.")
        return self._prod.get_endpoint().tile.op

    def _cons_tiles_ops(self) -> list[Tile]:
        if self._cons == []:
            raise ValueError("Cannot return cons.tile.op because prod was not created.")
        return [cons.get_endpoint().tile.op for cons in self._cons]

    def _get_depths(self) -> int | list[int]:
        if not self._prod:
            raise ValueError("Cannot return depths is producer is not created.")
        if len(self._cons) == 0:
            raise ValueError("Cannot return depths if no consumers are created.")
        depths = [self._prod.depth] + [con.depth for con in self._cons]
        if len(set(depths)) == 1:
            return depths[0]
        return depths

    def _get_endpoint(self, is_prod: bool) -> list[ObjectFifoEndpoint]:
        if is_prod:
            if self._prod:
                return [self._prod.get_endpoint()]
            else:
                raise ValueError(f"Prod endpoint not set for {self}")
        else:
            return [con.get_endpoint() for con in self._cons]

    @property
    def obj_type(self) -> type[np.ndarray]:
        return self._obj_type

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self._obj_type}, default_depth={self.default_depth}, name='{self.name}')"

    def resolve(
        self,
        loc: ir.Location | None = None,
        ip: ir.InsertionPoint | None = None,
    ) -> None:
        if self._op == None:
            self._op = object_fifo(
                self.name,
                self._prod_tile_op(),
                self._cons_tiles_ops(),
                self._get_depths(),
                np_ndarray_type_to_memref_type(self._obj_type),
                dimensionsToStream=self._dimensionToStream,
                dimensionsFromStreamPerConsumer=self._dimensionsFromStreamPerConsumer,
                loc=loc,
                ip=ip,
            )

            prod_endpoint = self._prod.get_endpoint()
            if isinstance(prod_endpoint, ObjectFifoLink):
                prod_endpoint.resolve()
            for con in self._cons:
                con_endpoint = con.get_endpoint()
                if isinstance(con_endpoint, ObjectFifoLink):
                    con_endpoint.resolve()

    def _acquire(
        self,
        port: ObjectFifoPort,
        num_elem: int,
        loc: ir.Location | None = None,
        ip: ir.InsertionPoint | None = None,
    ):
        if num_elem < 1:
            raise ValueError("Must consume at least one element")
        return self.op.acquire(port, num_elem)

    def _release(
        self,
        port: ObjectFifoPort,
        num_elem: int,
        loc: ir.Location | None = None,
        ip: ir.InsertionPoint | None = None,
    ):
        if num_elem < 1:
            raise ValueError("Must produce at least one element")
        self.op.release(port, num_elem)


class ObjectFifoHandle(Resolvable):
    def __init__(self, of: ObjectFifo, is_prod: bool, depth: int | None = None):
        if depth is None:
            if of.default_depth:
                depth = of.default_depth
            else:
                raise ValueError(
                    "Must specify either ObjectFifoHandle depth or ObjectFifo default depth; both are None."
                )
        if depth < 1:
            raise ValueError(f"Depth must be > 0 but is {depth}")
        self._port: ObjectFifoPort = (
            ObjectFifoPort.Produce if is_prod else ObjectFifoPort.Consume
        )
        self._is_prod = is_prod
        self._object_fifo = of
        self._depth = depth
        self._endpoint = None

    def acquire(
        self,
        num_elem: int,
        loc: ir.Location | None = None,
        ip: ir.InsertionPoint | None = None,
    ):
        if self._depth < num_elem:
            raise ValueError(
                f"Number of elements to acquire {num_elem} must be smaller than depth {self._depth}"
            )
        return self._object_fifo._acquire(self._port, num_elem, loc=loc, ip=ip)

    def release(
        self,
        num_elem: int,
        loc: ir.Location | None = None,
        ip: ir.InsertionPoint | None = None,
    ):
        if self._depth < num_elem:
            raise ValueError(
                f"Number of elements to release {num_elem} must be smaller than depth {self._depth}"
            )
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

    @property
    def depth(self) -> int:
        return self._depth

    def set_endpoint(self, endpoint: ObjectFifoEndpoint) -> None:
        if self._endpoint and self._endpoint != endpoint:
            raise ValueError("Endpoint already set for ObjectFifoEndpoint")
        self._endpoint = endpoint

    def get_endpoint(self) -> ObjectFifoEndpoint | None:
        return self._endpoint

    def get_all_endpoints(self) -> list[ObjectFifoEndpoint]:
        return self._object_fifo._get_endpoint(
            is_prod=True
        ) + self._object_fifo._get_endpoint(is_prod=False)

    def join(
        self,
        offsets: list[int],
        placement: PlacementTile = AnyMemTile,
        depths: list[int] | None = None,
        types: list[type[np.ndarray]] = None,
        names: list[str] | None = None,
    ) -> list[ObjectFifo]:
        num_subfifos = len(offsets)
        if depths is None:
            depths = [self.depth] * num_subfifos
        elif len(depths) != num_subfifos:
            raise ValueError("Number of depths does not match number of offsets")

        if types is None:
            types = [self._object_fifo.obj_type] * num_subfifos
        elif len(types) != num_subfifos:
            raise ValueError("Number of types does not match number of offsets")

        if names is None:
            names = [self._object_fifo.name + f"_join{i}" for i in range(num_subfifos)]
        elif len(names) != num_subfifos:
            raise ValueError("Number of names does not match number of offsets")

        # Create subfifos
        subfifos = []
        for i in range(num_subfifos):
            subfifos.append(
                ObjectFifo(
                    types[i],
                    name=names[i],
                    default_depth=depths[i],
                )
            )

        # Create link and set it as endpoints
        link = ObjectFifoLink(subfifos, self._object_fifo, placement, offsets, [])
        self.set_endpoint(link)
        for i in range(num_subfifos):
            subfifos[i].cons().set_endpoint(link)
        return subfifos

    def split(
        self,
        offsets: list[int],
        placement: PlacementTile = AnyMemTile,
        depths: list[int] | None = None,
        types: list[type[np.ndarray]] = None,
        names: list[str] | None = None,
    ) -> list[ObjectFifo]:
        num_subfifos = len(offsets)
        if depths is None:
            depths = [self.depth] * num_subfifos
        elif len(depths) != num_subfifos:
            raise ValueError("Number of depths does not match number of offsets")

        if types is None:
            types = [self._object_fifo.obj_type] * num_subfifos
        elif len(types) != num_subfifos:
            raise ValueError("Number of types does not match number of offsets")

        if names is None:
            names = [self._object_fifo.name + f"_split{i}" for i in range(num_subfifos)]
        elif len(names) != num_subfifos:
            raise ValueError("Number of names does not match number of offsets")

        # Create subfifos
        subfifos = []
        for i in range(num_subfifos):
            subfifos.append(
                ObjectFifo(
                    types[i],
                    name=names[i],
                    default_depth=depths[i],
                )
            )

        # Create link and set it as endpoints
        link = ObjectFifoLink(self._object_fifo, subfifos, placement, [], offsets)
        self.set_endpoint(link)
        for i in range(num_subfifos):
            subfifos[i].prod().set_endpoint(link)
        return subfifos

    def forward(
        self,
        placement: PlacementTile = AnyMemTile,
        obj_type: type[np.ndarray] | None = None,
        depth: int | None = None,
        name: str | None = None,
    ) -> ObjectFifo:
        if self._is_prod:
            raise ValueError("Cannot forward a producer ObjectFifoHandle")
        if obj_type is None:
            obj_type = self._object_fifo.obj_type
        if depth is None:
            if self._object_fifo.default_depth is None:
                raise ValueError(
                    f"Must provide depth since ObjectFifo default_depth={self._object_fifo.default_depth}"
                )
            depth = self._object_fifo.default_depth
        if not name:
            name = self._object_fifo.name + "_fwd"
        forward_fifo = ObjectFifo(obj_type, name=name, default_depth=depth)
        link = ObjectFifoLink(self._object_fifo, forward_fifo, placement, [], [])
        self.set_endpoint(link)
        forward_fifo.prod().set_endpoint(link)
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
        self._resolving = False

        if len(self._srcs) < 1:
            raise ValueError("An ObjectFifoLink must have at least one source")
        if len(self._dsts) < 1:
            raise ValueError("An ObjectFifoLink must have at least one destination")
        if len(self._srcs) != 1 and len(self._dsts) != 1:
            raise ValueError(
                "An ObjectFifoLink may only have > 1 of either sources or destinations, but not both"
            )
        if len(self._src_offsets) > 0 and len(self._src_offsets) != len(self._srcs):
            raise ValueError(
                "Then number of source offsets does not match the number of sources"
            )
        if len(self._dst_offsets) > 0 and len(self._dst_offsets) != len(self._dsts):
            raise ValueError(
                "Then number of destination offsets does not match the number of destinations"
            )
        self._op = None
        super(ObjectFifoEndpoint, self).__init__(placement)

    def resolve(
        self,
        loc: ir.Location | None = None,
        ip: ir.InsertionPoint | None = None,
    ) -> None:
        if not self._resolving:
            self._resolving = True

            # This function may be re-entrant as resolving sources/destinations
            # may call resolve on the object fifo endpoints, e.g., this function

            # We solve this be marking as _resolving BEFORE calling resolve on
            # sources or destinations.

            for s in self._srcs:
                s.resolve()
            for d in self._dsts:
                d.resolve()
            src_ops = [s.op for s in self._srcs]
            dst_ops = [d.op for d in self._dsts]
            self._op = object_fifo_link(
                src_ops, dst_ops, self._src_offsets, self._dst_offsets
            )
