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
    np_ndarray_type_get_dtype,
    np_ndarray_type_get_shape,
)
from ...util import single_elem_or_list_to_list

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
        dims_to_stream: list[Sequence[int]] | None = None,
        default_dims_from_stream_per_cons: list[Sequence[int]] | None = None,
    ):
        self._default_depth = default_depth
        if isinstance(self._default_depth, int) and self._default_depth < 1:
            raise ValueError(
                f"Default ObjectFifo depth must be > 0, but got {self._default_depth}"
            )
        self._obj_type = obj_type
        self._dims_to_stream = dims_to_stream
        self._default_dims_from_stream_per_cons = default_dims_from_stream_per_cons

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
    def default_dims_from_stream_per_cons(self) -> list[Sequence[int]]:
        return self._default_dims_from_stream_per_cons

    @property
    def dims_to_stream(self) -> list[Sequence[int]]:
        return self._dims_to_stream

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

    @property
    def obj_type(self) -> type[np.ndarray]:
        return self._obj_type

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__}({self._obj_type}, "
            f"default_depth={self.default_depth}, name='{self.name}', "
            f"prod={self._prod}, cons={self._cons})"
        )

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

    def cons(
        self,
        depth: int | None = None,
        dims_from_stream: list[Sequence[int]] | None = None,
    ) -> ObjectFifoHandle:
        if depth is None:
            if self._default_depth is None:
                raise ValueError(
                    f"If default_depth is None, then depth must be specified."
                )
            else:
                depth = self._default_depth

        if dims_from_stream is None:
            dims_from_stream = self._default_dims_from_stream_per_cons
        self._cons.append(
            ObjectFifoHandle(
                self, is_prod=False, depth=depth, dims_from_stream=dims_from_stream
            )
        )
        return self._cons[-1]

    def tiles(self) -> list[PlacementTile]:
        if self._prod == None:
            raise ValueError("Cannot return prod.tile.op because prod was not created.")
        if self._cons == []:
            raise ValueError("Cannot return cons.tile.op because prod was not created.")
        return [self._prod.tile] + [cons.tile for cons in self._cons]

    def _prod_tile_op(self) -> Tile:
        if self._prod == None:
            raise ValueError(
                f"Cannot return prod.tile.op for ObjectFifo {self.name} because prod was not created."
            )
        return self._prod.endpoint.tile.op

    def _cons_tiles_ops(self) -> list[Tile]:
        if len(self._cons) < 1:
            raise ValueError(
                f"Cannot return cons.tile.op for ObjectFifo {self.name} because no consumers were not created."
            )
        return [cons.endpoint.tile.op for cons in self._cons]

    def _get_depths(self) -> int | list[int]:
        if not self._prod:
            raise ValueError(
                "Cannot return depths since prod ObjectFifoHandle is not created."
            )
        if len(self._cons) == 0:
            raise ValueError(
                "Cannot return depths since no cons ObjectFifoHandles are created."
            )
        depths = [self._prod.depth] + [con.depth for con in self._cons]
        if len(set(depths)) == 1:
            return depths[0]
        return depths

    def _get_endpoint(self, is_prod: bool) -> list[ObjectFifoEndpoint]:
        if is_prod:
            if self._prod:
                return [self._prod.endpoint]
            else:
                raise ValueError(f"Prod endpoint not set for {self}")
        else:
            if len(self._cons) < 1:
                raise ValueError(f"Cons endpoint not set for {self}")
            return [con.endpoint for con in self._cons]

    def resolve(
        self,
        loc: ir.Location | None = None,
        ip: ir.InsertionPoint | None = None,
    ) -> None:
        if self._op == None:
            dims_from_stream_per_cons = [
                con.dims_from_stream if con.dims_from_stream else []
                for con in self._cons
            ]
            self._op = object_fifo(
                self.name,
                self._prod_tile_op(),
                self._cons_tiles_ops(),
                self._get_depths(),
                np_ndarray_type_to_memref_type(self._obj_type),
                dimensionsToStream=self._dims_to_stream,
                dimensionsFromStreamPerConsumer=dims_from_stream_per_cons,
                loc=loc,
                ip=ip,
            )

            if isinstance(self._prod.endpoint, ObjectFifoLink):
                self._prod.endpoint.resolve()
            for con in self._cons:
                if isinstance(con.endpoint, ObjectFifoLink):
                    con.endpoint.resolve()

    def _acquire(
        self,
        port: ObjectFifoPort,
        num_elem: int,
    ):
        if num_elem < 1:
            raise ValueError("Must consume at least one element")
        return self.op.acquire(port, num_elem)

    def _release(
        self,
        port: ObjectFifoPort,
        num_elem: int,
    ):
        if num_elem < 1:
            raise ValueError("Must produce at least one element")
        self.op.release(port, num_elem)


class ObjectFifoHandle(Resolvable):
    def __init__(
        self,
        of: ObjectFifo,
        is_prod: bool,
        depth: int | None = None,
        dims_from_stream: list[Sequence[int]] | None = None,
    ):
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
        if is_prod and dims_from_stream:
            raise ValueError("Can only specify dims_from_stream for cons handles")
        elif not is_prod and not dims_from_stream:
            dims_from_stream = of.default_dims_from_stream_per_cons

        self._is_prod = is_prod
        self._object_fifo = of
        self._depth = depth
        self._endpoint = None
        self._dims_from_stream = dims_from_stream

    def acquire(
        self,
        num_elem: int,
    ):
        if self._depth < num_elem:
            raise ValueError(
                f"Number of elements to acquire {num_elem} must be smaller than depth {self._depth}"
            )
        return self._object_fifo._acquire(self._port, num_elem)

    def release(
        self,
        num_elem: int,
    ):
        if self._depth < num_elem:
            raise ValueError(
                f"Number of elements to release {num_elem} must be smaller than depth {self._depth}"
            )
        return self._object_fifo._release(self._port, num_elem)

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
        return self._object_fifo.dtype

    @property
    def handle_type(self) -> str:
        if self._is_prod:
            return "prod"
        return "cons"

    @property
    def depth(self) -> int:
        return self._depth

    @property
    def dims_from_stream(self) -> list[Sequence[int]]:
        if self._is_prod:
            raise ValueError("prod ObjectFifoHandles cannot have dims_from_stream")
        return self._dims_from_stream

    @property
    def endpoint(self) -> ObjectFifoEndpoint | None:
        return self._endpoint

    @endpoint.setter
    def endpoint(self, endpoint: ObjectFifoEndpoint) -> None:
        if self._endpoint and self._endpoint != endpoint:
            raise ValueError(
                f"Endpoint already set for ObjectFifoHandle {self.name}.{self.handle_type}: "
                f"Set to {self._endpoint}, trying to set to {endpoint}"
            )
        self._endpoint = endpoint

    def all_of_endpoints(self) -> list[ObjectFifoEndpoint]:
        return self._object_fifo._get_endpoint(
            is_prod=True
        ) + self._object_fifo._get_endpoint(is_prod=False)

    def join(
        self,
        offsets: list[int],
        placement: PlacementTile = AnyMemTile,
        depths: list[int] | None = None,
        obj_types: list[type[np.ndarray]] = None,
        names: list[str] | None = None,
        dims_to_stream: list[list[Sequence[int] | None]] | None = None,
        dims_from_stream: list[list[Sequence[int] | None]] | None = None,
    ) -> list[ObjectFifo]:
        if not self._is_prod:
            raise ValueError(f"Cannot join() a {self.handle_type} ObjectFifoHandle")
        num_subfifos = len(offsets)
        if depths is None:
            depths = [self.depth] * num_subfifos
        elif len(depths) != num_subfifos:
            raise ValueError("Number of depths does not match number of offsets")

        if obj_types is None:
            obj_types = [self._object_fifo.obj_type] * num_subfifos
        elif len(obj_types) != num_subfifos:
            raise ValueError("Number of obj_types does not match number of offsets")

        if names is None:
            names = [self._object_fifo.name + f"_join{i}" for i in range(num_subfifos)]
        elif len(names) != num_subfifos:
            raise ValueError("Number of names does not match number of offsets")

        if dims_to_stream is None:
            dims_to_stream = [[]] * num_subfifos
        elif len(dims_to_stream) != num_subfifos:
            raise ValueError(
                "Number of dims to stream does not match number of offsets"
            )

        if dims_from_stream is None:
            dims_from_stream = [[]] * num_subfifos
        elif dims_from_stream and len(dims_from_stream) != num_subfifos:
            raise ValueError(
                "Number of dims_from_stream does not match number of offsets"
            )

        # Create subfifos
        subfifos = []
        for i in range(num_subfifos):
            subfifos.append(
                ObjectFifo(
                    obj_types[i],
                    name=names[i],
                    default_depth=depths[i],
                    dims_to_stream=dims_to_stream[i],
                )
            )

        subfifo_cons = [
            s.cons(depth=depths[i], dims_from_stream=dims_from_stream[i])
            for s in subfifos
        ]
        _ = ObjectFifoLink(subfifo_cons, self, placement, offsets, [])
        return subfifos

    def split(
        self,
        offsets: list[int],
        placement: PlacementTile = AnyMemTile,
        depths: list[int] | None = None,
        obj_types: list[type[np.ndarray]] = None,
        names: list[str] | None = None,
        dims_to_stream: list[list[Sequence[int]]] | None = None,
        dims_from_stream: list[list[Sequence[int]]] | None = None,
    ) -> list[ObjectFifo]:
        if self._is_prod:
            raise ValueError(f"Cannot split() a {self.handle_type} ObjectFifoHandle")
        num_subfifos = len(offsets)
        if depths is None:
            depths = [self.depth] * num_subfifos
        elif len(depths) != num_subfifos:
            raise ValueError("Number of depths does not match number of offsets")

        if obj_types is None:
            obj_types = [self._object_fifo.obj_type] * num_subfifos
        elif len(obj_types) != num_subfifos:
            raise ValueError("Number of obj_types does not match number of offsets")

        if names is None:
            names = [self._object_fifo.name + f"_split{i}" for i in range(num_subfifos)]
        elif len(names) != num_subfifos:
            raise ValueError("Number of names does not match number of offsets")

        if dims_to_stream is None:
            dims_to_stream = [[]] * num_subfifos
        elif len(dims_to_stream) != num_subfifos:
            raise ValueError(
                "Number of dims_to_stream arrays does not match number of offsets"
            )

        if dims_from_stream is None:
            dims_from_stream = [[]] * num_subfifos
        elif len(dims_from_stream) != num_subfifos:
            raise ValueError(
                "Number of dims_from_stream arrays does not match number of offsets"
            )

        # Create subfifos
        subfifos = []
        for i in range(num_subfifos):
            subfifos.append(
                ObjectFifo(
                    obj_types[i],
                    name=names[i],
                    default_depth=depths[i],
                    dims_to_stream=dims_to_stream[i],
                    default_dims_from_stream_per_cons=dims_from_stream[i],
                )
            )

        # Create link and set it as endpoints
        subfifo_prods = [s.prod() for s in subfifos]
        _ = ObjectFifoLink(self, subfifo_prods, placement, [], offsets)
        return subfifos

    def forward(
        self,
        placement: PlacementTile = AnyMemTile,
        obj_type: type[np.ndarray] | None = None,
        depth: int | None = None,
        name: str | None = None,
        dims_to_stream: list[Sequence[int]] | None = None,
        dims_from_stream: list[Sequence[int]] | None = None,
    ) -> ObjectFifo:
        if self._is_prod:
            raise ValueError(f"Cannot forward a {self.handle_type} ObjectFifoHandle")
        if obj_type:
            obj_type = [obj_type]
        if depth:
            depth = [depth]
        if name:
            name = [name]
        else:
            name = [self._object_fifo.name + "_fwd"]
        if dims_to_stream:
            dims_to_stream = [dims_to_stream]
        if dims_from_stream:
            dims_from_stream = [dims_from_stream]

        forward_fifo = self.split(
            [0],
            placement=placement,
            obj_types=obj_type,
            depths=depth,
            names=name,
            dims_to_stream=dims_to_stream,
            dims_from_stream=dims_from_stream,
        )
        return forward_fifo[0]

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
        for s in self._srcs:
            s.endpoint = self
        for d in self._dsts:
            d.endpoint = self
        ObjectFifoEndpoint.__init__(self, placement)

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
