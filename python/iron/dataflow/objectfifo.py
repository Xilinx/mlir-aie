# objectfifo.py -*- Python -*-
#
# Copyright (C) 2024 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from ... import ir  # pyright: ignore[reportMissingImports, reportAttributeAccessIssue]
from ...dialects._aie_enum_gen import (  # pyright: ignore[reportMissingImports]
    AIETileType,
    ObjectFifoPort,
)
from ...dialects._aie_ops_gen import (  # pyright: ignore[reportMissingImports]
    ObjectFifoCreateOp,
)
from ...dialects.aie import object_fifo, object_fifo_link
from ...helpers.npdtypes import (
    NpuDType,
    np_ndarray_type_get_dtype,
    np_ndarray_type_get_shape,
    pack_pad_value,
    single_elem_or_list_to_list,
)
from ...helpers.sourceloc import SourceSite
from ...helpers.taplib import TensorAccessPattern
from ...helpers.taplib._symbolic import sprod
from ...helpers.util import np_ndarray_type_to_memref_type
from ..device import AnyComputeTile, AnyMemTile, AnyShimTile, Tile
from ..resolvable import NotResolvedError, Resolvable
from ..scratchpad_parameter import ScratchpadParameter
from .endpoint import ObjectFifoEndpoint


def _object_walk(
    tap: TensorAccessPattern | None, what: str, can_pad: bool = False
) -> TensorAccessPattern | None:
    """Return `tap` after checking that it is a stream walk an ObjectFifo can take."""
    if tap is None:
        return None
    if not isinstance(tap, TensorAccessPattern):
        raise TypeError(
            f"{what} takes a TensorAccessPattern, got {type(tap).__name__}; "
            "TensorAccessPattern(shape, 0, sizes, strides) spells out a walk "
            "by its numbers"
        )
    object_fifo.stream_dims(tap, what, can_pad)
    return tap


def _check_walk_size(
    tap: TensorAccessPattern | None, sizes: list[int], emits: int, what: str
) -> None:
    """Check that `tap` walks a tensor of each of `sizes` elements, padded to `emits`."""
    if tap is None:
        return
    numel = sprod(tap.tensor_dims)
    for size in sizes:
        if numel != size:
            raise ValueError(
                f"{what} {tap!r} walks a tensor of shape {tuple(tap.tensor_dims)}, "
                f"but each transfer moves {size} elements"
            )
    padded = sprod(tap.padded_sizes)
    if tap.padding is not None and padded != emits:
        raise ValueError(
            f"{what} {tap!r} emits {padded} elements, "
            f"but each object it fills has {emits}"
        )


def _same_shim_pin(a: "Tile | None", b: "Tile | None") -> bool:
    """Whether two shim-tile re-pin requests refer to the same placement.

    Tile.__eq__ is identity-based, so compare by (col, row); two unpinned
    (None or col/row None) tiles are considered the same.
    """
    if a is None or b is None:
        return a is b
    return (a.col, a.row) == (b.col, b.row)


class ObjectFifo(Resolvable):
    """A synchronized, explicit dataflow channel between IRON program components such as [`Worker`][iron.Worker]s and the [`Runtime`][iron.Runtime].

    Internally, an ObjectFifo is a circular buffer with a given depth and
    element type. Its users are explicitly either a *producer* or a
    *consumer*, and each user holds an [`ObjectFifoHandle`][iron.dataflow.objectfifo.ObjectFifoHandle]
    carrying its (possibly unplaced) tile.

    Example:
        ```python
        of = ObjectFifo(np.ndarray[(1024,), np.dtype[np.int32]], name="in")
        producer = of.prod()   # one producer handle
        consumer = of.cons()   # one or more consumer handles
        ```
    """

    def __init__(
        self,
        obj_type: type[np.ndarray],
        *,
        depth: int | None = 2,
        name: str | None = None,
        to_stream: TensorAccessPattern | None = None,
        from_stream_per_cons: TensorAccessPattern | None = None,
        plio: bool = False,
        pad_value: int = 0,
        disable_synchronization: bool = False,
        repeat_count: int | None = None,
        delegate_tile: Tile | None = None,
        via_DMA: bool = False,
        init_values: list[np.ndarray] | None = None,
        consumer_obj_type: type[np.ndarray] | None = None,
        aie_stream: tuple[int, int] | None = None,
        packet: bool = False,
        packet_id: int | None = None,
    ):
        """Construct an ObjectFifo.

        Args:
            obj_type (type[np.ndarray]): The type of each buffer in the ObjectFifo
            depth (int | None, optional): The default depth of the ObjectFifo endpoints. Defaults to 2.
            name (str | None, optional): The name of the ObjectFifo. If None is given, the Program names it when it resolves. Defaults to None.
            to_stream (TensorAccessPattern | None, optional): How the producer's DMA
                walks each object onto the AXI stream: a pattern over a tensor the
                size of one object (of each input's object, for a join's output),
                at offset 0. A padded one (from ``.pad()``) also pads the stream,
                on a MemTile. Defaults to None (a linear walk).
            from_stream_per_cons (TensorAccessPattern | None, optional): How each
                consumer's DMA writes the stream into its object (into each
                output's object, for a distribute's input), as for ``to_stream``
                but unpadded. Defaults to None (a linear walk).
            plio (bool, optional): Whether the ObjectFifo uses PLIO connections. Defaults to False.
            pad_value (int, optional): Per-element constant value used to fill the padding
                of a padded ``to_stream``. Packed into the raw 32-bit CONSTANT_PAD_VALUE
                register using this fifo's element width. Defaults to 0.
            disable_synchronization (bool, optional): When True, disables lock-based
                synchronization on the ObjectFifo. Defaults to False.
            repeat_count (int | None, optional): If set, the sending end replicates
                each object this many times before moving on to the next one. The
                receiving end covers a whole batch in one acquire, so its lock
                initializers scale to match. Distinct from ``iter_count``
                (BD-chain iteration count). Defaults to None.
            delegate_tile (Tile | None, optional): Shared-memory delegate tile. When set, the
                ObjectFifo's underlying buffer pool is allocated on this tile's memory module
                instead of the default placement. Lowers to ``aie.objectfifo.allocate``. *Only
                valid when both producer and consumer have shared-memory access to the
                delegate tile* (e.g. self-loop fifos where prod == cons, or fifos between
                adjacent tiles spilling to a neighboring MemTile). The delegate is the storage
                location, not a producer- or consumer-side concept; the underlying op verifier
                rejects this if either endpoint cannot share memory with the delegate.
                Defaults to None.
            via_DMA (bool, optional): When True, force the ObjectFifo to route through DMA
                even when producer and consumer share memory (where a lock-only path would
                otherwise be used). Lowers to the ``via_DMA`` attribute on the underlying
                ``aie.objectfifo`` op. Defaults to False.
            init_values (list[np.ndarray] | None, optional): Per-buffer static initial values
                for the producer endpoint. One ndarray per producer-side buffer; the producer
                tile must be able to hold static data at design startup (e.g. a MemTile).
                Lowers to the ``initValues`` attribute on the underlying ``aie.objectfifo``
                op. Defaults to None.
            consumer_obj_type (type[np.ndarray] | None, optional): Consumer element type for
                asymmetric transfer granularity. When set, the producer sends obj_type-sized
                transfers and the consumer receives consumer_obj_type-sized transfers.
                Producer element count must be an integer multiple of consumer element count.
                Defaults to None.
            aie_stream (tuple[int, int] | None, optional): Mark the fifo as a direct
                AIE-stream connection by stamping the ``aie_stream`` / ``aie_stream_port``
                attributes ``(end, port)`` on the underlying ``aie.objectfifo`` op. Use with
                kernels that emit on the wire via ``put_ms()`` instead of going through an L1
                buffer. Defaults to None.
            packet (bool, optional): Route this ObjectFifo as an ``aie.packet_flow``, sharing
                the stream with other packet flows instead of reserving a circuit for it.
                Decided per fifo, so a design may mix packet- and circuit-switched fifos.
                Defaults to False.
            packet_id (int | None, optional): Pin the 5-bit header the source stamps, for
                designs that route on the id (e.g. a MemTile dispatching to one of several
                cores). Requires ``packet``; when absent, allocation picks an id no other
                flow is using. Defaults to None.

        Raises:
            TypeError: If a stream walk is not a ``TensorAccessPattern``.
            ValueError: If ``depth`` is provided and is less than 1, or a stream
                walk is staged, has a nonzero offset, or pads a consumer.
        """
        self._site = SourceSite.capture()
        self._depth = depth
        if self._depth is not None and self._depth < 1:
            raise ValueError(
                f"Default ObjectFifo depth must be > 0, but got {self._depth}"
            )
        self._obj_type = obj_type
        self._consumer_obj_type: type[np.ndarray] | None = consumer_obj_type
        self._to_stream = _object_walk(to_stream, "to_stream", can_pad=True)
        self._from_stream_per_cons = _object_walk(
            from_stream_per_cons, "from_stream_per_cons"
        )
        self._plio = plio
        self._pad_value = pad_value
        self.name = name
        self._op: ObjectFifoCreateOp | None = None
        self._prod: ObjectFifoHandle | None = None
        self._cons: list[ObjectFifoHandle] = []
        self._resolving = False
        self._iter_count: int | None = None
        self._repeat_count: int | None = repeat_count
        self._disable_synchronization: bool = disable_synchronization
        # Delegate tile for shared-memory buffer placement (lowers to aie.objectfifo.allocate).
        # Must be resolved before resolve() runs — Program.resolve() picks this up via
        # ObjectFifo._delegate_tile when collecting tiles to assign MLIR ops to.
        self._delegate_tile: Tile | None = delegate_tile
        self._via_DMA: bool = via_DMA
        self._init_values: list[np.ndarray] | None = init_values
        self._aie_stream: tuple[int, int] | None = aie_stream
        self._packet: bool = packet
        self._packet_id: int | None = packet_id

    @property
    def depth(self) -> int | None:
        """The default depth of the ObjectFifo's endpoints; ``prod()`` and ``cons()`` may override it."""
        return self._depth

    @property
    def from_stream_per_cons(self) -> TensorAccessPattern | None:
        """How each consumer's DMA writes the stream into its object, unless ``cons()`` overrides it; None for a linear walk."""
        return self._from_stream_per_cons

    @property
    def to_stream(self) -> TensorAccessPattern | None:
        """How the producer's DMA walks each object onto the stream; None for a linear walk."""
        return self._to_stream

    @property
    def op(self) -> ObjectFifoCreateOp:
        if self._op is None:
            raise NotResolvedError()
        return self._op

    @property
    def shape(self) -> Sequence[int]:
        """The shape of each buffer belonging to the ObjectFifo."""
        return np_ndarray_type_get_shape(self._obj_type)

    @property
    def dtype(self) -> type[NpuDType]:
        """The per-element data type of each element in each buffer belonging to the ObjectFifo."""
        return np_ndarray_type_get_dtype(self._obj_type)

    @property
    def obj_type(self) -> type[np.ndarray]:
        """The tensor type of each buffer belonging to the ObjectFifo."""
        return self._obj_type

    @property
    def consumer_obj_type(self) -> type[np.ndarray]:
        """The tensor type of each buffer at a consumer: ``obj_type`` unless the transfer is asymmetric."""
        return self._consumer_obj_type or self._obj_type

    def set_iter_count(self, iter_count: int):
        """Set how many times each end of the ObjectFifo cycles through its buffers.

        Args:
            iter_count (int): Passes each end's BD chain makes before it stops.
                Both ends carry ``depth * repeat_count * iter_count`` objects, so
                a sending end replicating each object ``repeat_count`` times
                makes that many fewer passes than the receiving end opposite it.
                - Must be in range [1, 256]

        Raises:
            ValueError: If iter_count is outside the valid range [1, 256]
        """
        if not iter_count or iter_count < 1 or iter_count > 256:
            raise ValueError("Iter count must be in [1, 256] range.")

        self._iter_count = iter_count

    def __str__(self) -> str:
        prod_endpoint = None
        if self._prod:
            prod_endpoint = self._prod.endpoint
        return (
            f"{self.__class__.__name__}({self._obj_type}, "
            f"depth={self.depth}, name='{self.name}', "
            f"prod={prod_endpoint}, cons={[c.endpoint for c in self._cons]})"
        )

    def prod(
        self,
        depth: int | None = None,
        channel: int | None = None,
        tile: Tile | None = None,
    ) -> ObjectFifoHandle:
        """Return an ObjectFifoHandle of type producer.

        Each ObjectFifo may have only one producer handle, so if one already
        exists, a new reference to this handle will be returned.

        Args:
            depth (int | None, optional): The depth of the buffers at the endpoint corresponding to the producer handle. Defaults to None.
            channel (int | None, optional): Pin the producer endpoint's DMA channel instead of first-free assignment. Defaults to None (auto-assign).
            tile (Tile | None, optional): When this handle drives an ObjectFifo
                from the runtime (passed in ``Runtime`` ``fn_args``), the shim tile
                its host-side DMA binds to. Defaults to None (any available shim tile).

        Raises:
            ValueError: Arguments are validated
            ValueError: If depth was not specified on ObjectFifo construction, depth must be specified here.

        Returns:
            ObjectFifoHandle: The producer handle to this ObjectFifo.
        """
        if self._prod:
            if depth is None:
                if self._depth is None:
                    raise ValueError("If depth is None, then depth must be specified.")
                else:
                    depth = self._depth
            elif depth < 1:
                raise ValueError(f"Depth must be > 1, but got {depth}")
            if channel is not None and self._prod.channel != channel:
                raise ValueError(
                    f"Producer handle for {self.name} already pinned to channel "
                    f"{self._prod.channel}, cannot re-pin to {channel}."
                )
            if tile is not None and not _same_shim_pin(self._prod._shim_tile, tile):
                raise ValueError(
                    f"Producer handle for {self.name} already pinned to shim tile "
                    f"{self._prod._shim_tile}, cannot re-pin to {tile}."
                )
        else:
            self._prod = ObjectFifoHandle(self, True, depth, channel=channel, tile=tile)
        return self._prod

    def cons(
        self,
        depth: int | None = None,
        from_stream: TensorAccessPattern | None = None,
        channel: int | None = None,
        tile: Tile | None = None,
    ) -> ObjectFifoHandle:
        """Return an ObjectFifoHandle of type consumer.

        Each ObjectFifo may have multiple consumers, so this will return a new
        consumer handle every time it is called.

        Args:
            depth (int | None, optional): The depth of the buffers at the endpoint corresponding to this consumer handle. Defaults to None.
            from_stream (TensorAccessPattern | None, optional): How this consumer's DMA writes
                the stream into its object (see ``from_stream_per_cons``). Defaults to None.
            channel (int | None, optional): Pin this consumer endpoint's DMA channel instead of first-free assignment. Defaults to None (auto-assign).
            tile (Tile | None, optional): When this handle drains an ObjectFifo to
                the runtime (passed in ``Runtime`` ``fn_args``), the shim tile its
                host-side DMA binds to. Defaults to None (any available shim tile).

        Raises:
            ValueError: Arguments are validated

        Returns:
            ObjectFifoHandle: A consumer handle to this ObjectFifo.
        """
        if depth is None:
            if self._depth is None:
                raise ValueError("If depth is None, then depth must be specified.")
            else:
                depth = self._depth

        self._cons.append(
            ObjectFifoHandle(
                self,
                is_prod=False,
                depth=depth,
                from_stream=from_stream,
                channel=channel,
                tile=tile,
            )
        )
        return self._cons[-1]

    def tiles(self, cons_only: bool = False) -> list[Tile]:
        """Return the placement tiles corresponding to the endpoints of all handles of this ObjectFifo.

        Raises:
            ValueError: A producer handle must be constructed.
            ValueError: At least one consumer handle must be constructed.

        Returns:
            list[Tile]: A list of tiles of the endpoints of this ObjectFifo.
        """
        tiles = []
        if not cons_only:
            if self._prod is None:
                raise ValueError(
                    "Cannot return prod.tile.op because prod was not created."
                )
            if self._prod.endpoint is None:
                raise ValueError(f"Prod endpoint not set for {self}")
            assert self._prod.endpoint.tile is not None
            tiles += [self._prod.endpoint.tile]
        if self._cons == []:
            raise ValueError("Cannot return cons tiles because cons were not created.")
        for cons in self._cons:
            if cons.endpoint is None:
                raise ValueError(f"Cons endpoint not set for {self}")
            assert cons.endpoint.tile is not None
            tiles.append(cons.endpoint.tile)
        return tiles

    def _prod_tile_op(self) -> Tile:
        if self._prod is None:
            raise ValueError(
                f"Cannot return prod.tile.op for ObjectFifo {self.name} because prod was not created."
            )
        if self._prod.endpoint is None:
            raise ValueError(f"Prod endpoint not set for {self}")
        assert self._prod.endpoint.tile is not None
        return self._prod.endpoint.tile.op

    def _cons_tiles_ops(self) -> list[Tile]:
        if len(self._cons) < 1:
            raise ValueError(
                f"Cannot return cons.tile.op for ObjectFifo {self.name} because no consumers were created."
            )
        ops = []
        for cons in self._cons:
            if cons.endpoint is None:
                raise ValueError(f"Cons endpoint not set for {self}")
            assert cons.endpoint.tile is not None
            ops.append(cons.endpoint.tile.op)
        return ops

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
                if self._prod.endpoint is None:
                    raise ValueError(f"Prod endpoint not set for {self}")
                return [self._prod.endpoint]
            else:
                raise ValueError(f"Prod endpoint not set for {self}")
        else:
            if len(self._cons) < 1:
                raise ValueError(f"Cons endpoint not set for {self}")
            endpoints = []
            for con in self._cons:
                if con.endpoint is None:
                    raise ValueError(f"Cons endpoint not set for {self}")
                endpoints.append(con.endpoint)
            return endpoints

    def _check_walk_sizes(self) -> None:
        assert self._prod is not None
        link = self._prod.endpoint
        sizes = [math.prod(self.shape)]
        if isinstance(link, ObjectFifoLink):
            sizes = link._transfer_sizes(self)
        emits = math.prod(np_ndarray_type_get_shape(self.consumer_obj_type))
        _check_walk_size(self._to_stream, sizes, emits, f"{self.name} to_stream")
        for con in self._cons:
            sizes = [emits]
            if isinstance(con.endpoint, ObjectFifoLink):
                sizes = con.endpoint._transfer_sizes(self)
            _check_walk_size(con.from_stream, sizes, emits, f"{self.name} from_stream")

    def resolve(
        self,
        loc: ir.Location | None = None,
        ip: ir.InsertionPoint | None = None,
    ) -> None:
        if not self._resolving:
            self._resolving = True
            self._check_walk_sizes()
            from_stream_per_cons = [con.from_stream for con in self._cons]

            consumer_datatype = (
                np_ndarray_type_to_memref_type(self._consumer_obj_type)
                if self._consumer_obj_type is not None
                else None
            )
            op = object_fifo(
                self.name,
                self._prod_tile_op(),
                self._cons_tiles_ops(),
                self._get_depths(),
                np_ndarray_type_to_memref_type(self._obj_type),
                dimensionsToStream=self._to_stream,
                dimensionsFromStreamPerConsumer=from_stream_per_cons,
                plio=self._plio,
                padValue=(
                    pack_pad_value(
                        self._pad_value,
                        np.dtype(np_ndarray_type_get_dtype(self._obj_type)).itemsize,
                    )
                    if self._pad_value
                    else None
                ),
                iter_count=self._iter_count,
                disable_synchronization=self._disable_synchronization or None,
                via_DMA=self._via_DMA or None,
                initValues=self._init_values,
                consumer_datatype=consumer_datatype,
                packet=self._packet or None,
                packet_id=self._packet_id,
                loc=ir.Location.name(
                    self.name, childLoc=loc or self._site.location(self.name)
                ),
                ip=ip,
            )
            self._op = op

            if self._repeat_count is not None:
                op.set_repeat_count(self._repeat_count)

            if self._aie_stream is not None:
                op.set_aie_stream(*self._aie_stream)

            # Pin DMA channels requested on the handles. The producer channel
            # and one channel per consumer (-1 = auto-assign that consumer) are
            # stamped onto the op for the stateful-transform pass to honor.
            if self._prod is not None and self._prod.channel is not None:
                op.set_prod_dma_channel(self._prod.channel)
            cons_channels = [con.channel for con in self._cons]
            if any(ch is not None for ch in cons_channels):
                op.set_cons_dma_channels(
                    [-1 if ch is None else ch for ch in cons_channels]
                )

            # Shared-memory delegate: redirect the fifo's buffer pool to a tile
            # whose memory module is shared with both prod and cons. See the
            # delegate_tile docstring on ObjectFifo for the constraint.
            if self._delegate_tile is not None:
                op.allocate(self._delegate_tile.op)

            assert self._prod is not None
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
    """A handle to an [`ObjectFifo`][iron.ObjectFifo], of type *producer* or *consumer*.

    Producer and consumer handles are what [`Worker`][iron.Worker] core
    functions call [`acquire`][iron.dataflow.objectfifo.ObjectFifoHandle.acquire]
    and [`release`][iron.dataflow.objectfifo.ObjectFifoHandle.release] on to
    move data through the fifo. Obtain them via
    [`ObjectFifo.prod()`][iron.dataflow.objectfifo.ObjectFifo.prod] and
    [`ObjectFifo.cons()`][iron.dataflow.objectfifo.ObjectFifo.cons].
    """

    def __init__(
        self,
        of: ObjectFifo,
        is_prod: bool,
        depth: int | None = None,
        from_stream: TensorAccessPattern | None = None,
        channel: int | None = None,
        tile: Tile | None = None,
    ):
        """Construct an ObjectFifoHandle.

        Args:
            of (ObjectFifo): The ObjectFifo to construct the handle for.
            is_prod (bool): Whether the handle should be producer or consumer handle.
            depth (int | None, optional): The depth of the ObjectFifo at this endpoint. Defaults to None.
            from_stream (TensorAccessPattern | None, optional): This consumer's own walk into its object. Only valid for consumer handles. Defaults to None.
            channel (int | None, optional): Pin this endpoint's DMA channel instead of first-free assignment. Defaults to None (auto-assign).
            tile (Tile | None, optional): Shim tile for a runtime-driven endpoint (see prod()/cons()). Defaults to None.

        Raises:
            ValueError: Arguments are validated.
        """
        if depth is None:
            if of.depth:
                depth = of.depth
            else:
                raise ValueError(
                    "Must specify either ObjectFifoHandle depth or ObjectFifo default depth; both are None."
                )
        if depth < 1:
            raise ValueError(f"Depth must be > 0 but is {depth}")
        self._port: ObjectFifoPort = (
            ObjectFifoPort.Produce if is_prod else ObjectFifoPort.Consume
        )
        from_stream = _object_walk(from_stream, "from_stream")
        if is_prod and from_stream is not None:
            raise ValueError("Can only specify from_stream for cons handles")
        elif not is_prod and from_stream is None:
            from_stream = of.from_stream_per_cons

        self._is_prod = is_prod
        self._object_fifo = of
        self._depth = depth
        self._channel = channel
        self._shim_tile = tile
        self._endpoint = None
        self._from_stream = from_stream

    def acquire(
        self,
        num_elem: int,
    ) -> list:
        """Acquire access to some elements of the ObjectFifo, using ObjectFifo synchronization to moderate access.

        Args:
            num_elem (int): Number of elements to acquire. If some elements are already
                acquired, only the additional elements needed to reach a total of
                ``num_elem`` are acquired.

        Raises:
            ValueError: Number of elements cannot exceed ObjectFifo depth.

        Returns:
            An indexable handle to the acquired elements: a single element when ``num_elem == 1``, or an indexable view when ``num_elem > 1``.
        """
        if self._depth < num_elem:
            raise ValueError(
                f"Number of elements to acquire {num_elem} must be smaller than depth {self._depth}"
            )
        return self._object_fifo._acquire(self._port, num_elem)

    def release(
        self,
        num_elem: int,
    ) -> None:
        """Release access to some elements of the ObjectFifo, allowing the other endpoint of the ObjectFifo to acquire them.

        Args:
            num_elem (int): Number of elements to release.

        Raises:
            ValueError: Number of elements cannot exceed ObjectFifo depth.
        """
        if self._depth < num_elem:
            raise ValueError(
                f"Number of elements to release {num_elem} must be smaller than depth {self._depth}"
            )
        self._object_fifo._release(self._port, num_elem)

    @property
    def name(self) -> str | None:
        """The name of the ObjectFifo, None until the Program names an unnamed one."""
        return self._object_fifo.name

    def _derived_name(self, suffix: str) -> str | None:
        name = self._object_fifo.name
        return None if name is None else name + suffix

    @property
    def channel(self) -> int | None:
        """The pinned DMA channel for this handle's endpoint, or None to auto-assign."""
        return self._channel

    @property
    def op(self) -> ObjectFifoCreateOp:
        return self._object_fifo.op

    @property
    def obj_type(self) -> type[np.ndarray]:
        """The per-buffer type of the ObjectFifo."""
        return self._object_fifo.obj_type

    @property
    def shape(self) -> Sequence[int]:
        """The per-buffer shape of the ObjectFifo."""
        return self._object_fifo.shape

    @property
    def dtype(self) -> type[NpuDType]:
        """The per-element datatype of the ObjectFifo."""
        return self._object_fifo.dtype

    @property
    def handle_type(self) -> str:
        """A string referencing the type of this ObjectFifoHandle."""
        if self._is_prod:
            return "prod"
        return "cons"

    @property
    def depth(self) -> int:
        """The depth of this ObjectFifoHandle."""
        return self._depth

    @property
    def from_stream(self) -> TensorAccessPattern | None:
        """How this consumer's DMA writes the stream into its object; None for a linear walk."""
        if self._is_prod:
            raise ValueError("prod ObjectFifoHandles cannot have from_stream")
        return self._from_stream

    @property
    def endpoint(self) -> ObjectFifoEndpoint | None:
        """The endpoint of this ObjectFifoHandle."""
        return self._endpoint

    def __str__(self) -> str:
        my_str = f"ObjectFifoHandle({self.handle_type}, {self.depth}, "
        if not self._is_prod:
            my_str += f"{self.from_stream}, "
        my_str += f"{self._object_fifo})"
        return my_str

    @endpoint.setter
    def endpoint(self, endpoint: ObjectFifoEndpoint) -> None:
        if self._endpoint and self._endpoint != endpoint:
            raise ValueError(
                f"Endpoint already set for ObjectFifoHandle {self.name}.{self.handle_type}: "
                f"Set to {self._endpoint}, trying to set to {endpoint}"
            )
        self._endpoint = endpoint

    def _emit_transfer(
        self,
        rt_data,
        tap,
        wait: bool,
        packet: tuple[int, int] | None,
        offset_parameter: ScratchpadParameter | str | None,
        group,
        managed=True,
        length_parameter: ScratchpadParameter | str | None = None,
        length_unit: int | None = None,
    ):
        """Shared body for fill()/drain().

        Bind the shim endpoint, register the fifo with the active runtime
        sequence, then emit the transfer on the shim allocation this fifo's name
        declares. Returns a [`Task`][iron.runtime.dmataskhandle.Task] handle to
        the transfer (carry it as a ``range_`` iter_arg;
        ``.free()``/``.await_()`` it). See ``emit_shim_transfer`` for the
        arguments.

        Lazy imports break the runtime<->dataflow import cycle.
        """
        from ..runtime._context import active_sequence
        from ..runtime.dmatask import emit_shim_transfer
        from ..runtime.endpoint import RuntimeEndpoint

        # The endpoint is normally bound eagerly when this handle is registered in
        # Runtime fn_args (using its prod()/cons() tile); bind it here too so a
        # handle used only via fill/drain still gets a shim endpoint.
        if self._endpoint is None:
            self.endpoint = RuntimeEndpoint(self._shim_tile)
        active_sequence().note_fifo(self)

        assert self.name is not None
        return emit_shim_transfer(
            self.name,
            rt_data,
            tap=tap,
            wait=wait,
            packet=packet,
            offset_parameter=offset_parameter,
            group=group,
            managed=managed,
            length_parameter=length_parameter,
            length_unit=length_unit,
        )

    def fill(
        self,
        source,
        tap=None,
        wait: bool = False,
        packet: tuple[int, int] | None = None,
        offset_parameter: ScratchpadParameter | str | None = None,
        group=None,
        managed: bool = True,
        length_parameter: ScratchpadParameter | str | None = None,
        length_unit: int | None = None,
    ):
        """Fill this producer ObjectFifo with data from the ``source`` runtime buffer.

        Call from within a [`Runtime`][iron.Runtime] sequence body on a producer
        handle. See ``_emit_transfer`` for the shared arguments; returns a
        [`Task`][iron.runtime.dmataskhandle.Task] handle to the transfer.
        """
        if not self._is_prod:
            raise ValueError("fill() is only valid on a producer ObjectFifoHandle")

        return self._emit_transfer(
            source,
            tap,
            wait,
            packet,
            offset_parameter,
            group,
            managed,
            length_parameter,
            length_unit,
        )

    def drain(
        self,
        dest,
        tap=None,
        wait: bool = False,
        packet: tuple[int, int] | None = None,
        offset_parameter: ScratchpadParameter | str | None = None,
        group=None,
        managed: bool = True,
        length_parameter: ScratchpadParameter | str | None = None,
        length_unit: int | None = None,
    ):
        """Drain this consumer ObjectFifo, writing data to the ``dest`` runtime buffer.

        Call from within a [`Runtime`][iron.Runtime] sequence body on a consumer
        handle. See ``_emit_transfer`` for the shared arguments; returns a
        [`Task`][iron.runtime.dmataskhandle.Task] handle to the transfer.
        """
        if self._is_prod:
            raise ValueError("drain() is only valid on a consumer ObjectFifoHandle")

        return self._emit_transfer(
            dest,
            tap,
            wait,
            packet,
            offset_parameter,
            group,
            managed,
            length_parameter,
            length_unit,
        )

    def all_of_endpoints(self) -> list[ObjectFifoEndpoint]:
        """All endpoints belonging to an ObjectFifo."""
        return self._object_fifo._get_endpoint(
            is_prod=True
        ) + self._object_fifo._get_endpoint(is_prod=False)

    def join(
        self,
        offsets: list[int],
        tile: Tile | None = AnyMemTile,
        depths: list[int] | None = None,
        obj_types: list[type[np.ndarray]] | None = None,
        names: Sequence[str | None] | None = None,
        to_stream: Sequence[TensorAccessPattern | None] | None = None,
        from_stream: Sequence[TensorAccessPattern | None] | None = None,
        plio: bool = False,
        repeat_counts: Sequence[int | None] | None = None,
    ) -> list[ObjectFifo]:
        """Construct multiple ObjectFifos which feed data into a ObjectFifoHandle.

        Note that this function is only valid for producer ObjectFifoHandles.

        Args:
            offsets (list[int]): Offsets into the current producer, each corresponding to a new consumer.
            tile (Tile, optional): The tile where the Join operation occurs. Also accepts None (treated as AnyMemTile). Defaults to AnyMemTile.
            depths (list[int] | None, optional): The depth of each new ObjectFifo. Defaults to None.
            obj_types (list[type[np.ndarray]], optional): The type of the buffers corresponding to each new ObjectFifo. Defaults to None.
            names (Sequence[str | None] | None, optional): The name of each new ObjectFifo. If not given,
                each is named after this one, or by the Program if this one is unnamed.
                Defaults to None.
            to_stream (Sequence[TensorAccessPattern | None] | None, optional): Each new ObjectFifo's
                ``to_stream``. Defaults to None.
            from_stream (Sequence[TensorAccessPattern | None] | None, optional): Each new
                ObjectFifo consumer's ``from_stream``. Defaults to None.
            plio (bool, optional): Set plio on each new ObjectFifo. Defaults to False.
            repeat_counts (Sequence[int | None] | None, optional): Per-sub-fifo MemTile DMA repeat count (see ObjectFifo.repeat_count). Defaults to None.

        Raises:
            ValueError: Arguments are validated

        Returns:
            list[ObjectFifo]: A list of newly constructed ObjectFifos whose consumers are used in this join() operation.
        """
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
            names = [self._derived_name(f"_join{i}") for i in range(num_subfifos)]
        elif len(names) != num_subfifos:
            raise ValueError("Number of names does not match number of offsets")

        if to_stream is None:
            to_stream = [None] * num_subfifos
        elif len(to_stream) != num_subfifos:
            raise ValueError(
                "Number of dims to stream does not match number of offsets"
            )

        if from_stream is None:
            from_stream = [None] * num_subfifos
        elif len(from_stream) != num_subfifos:
            raise ValueError("Number of from_stream does not match number of offsets")

        if repeat_counts is None:
            repeat_counts = [None for _ in range(num_subfifos)]
        elif len(repeat_counts) != num_subfifos:
            raise ValueError("Number of repeat_counts does not match number of offsets")

        # Create subfifos
        subfifos = []
        for i in range(num_subfifos):
            subfifos.append(
                ObjectFifo(
                    obj_types[i],
                    name=names[i],
                    depth=depths[i],
                    to_stream=to_stream[i],
                    plio=plio,
                    repeat_count=repeat_counts[i],
                )
            )

        subfifo_cons = [
            s.cons(depth=depths[i], from_stream=from_stream[i])
            for i, s in enumerate(subfifos)
        ]
        _ = ObjectFifoLink(subfifo_cons, self, tile, offsets, [])
        return subfifos

    def split(
        self,
        offsets: list[int],
        tile: Tile | None = AnyMemTile,
        depths: list[int] | None = None,
        obj_types: list[type[np.ndarray]] | None = None,
        names: Sequence[str | None] | None = None,
        to_stream: Sequence[TensorAccessPattern | None] | None = None,
        from_stream: Sequence[TensorAccessPattern | None] | None = None,
        plio: bool = False,
        repeat_counts: Sequence[int | None] | None = None,
        pad_value: list[int] | None = None,
        channels: Sequence[int | None] | None = None,
    ) -> list[ObjectFifo]:
        """Split the data from an ObjectFifoConsumer handle by sending it to producers in N newly constructed ObjectFifos.

        Note this operation is only valid for ObjectFifoHandles of type consumer.

        Args:
            offsets (list[int]): The offset into the current consumer for each new ObjectFifo producer.
            tile (Tile, optional): The tile where the Split operation takes place. Also accepts None (treated as AnyMemTile). Defaults to AnyMemTile.
            depths (list[int] | None, optional): The depth of each new ObjectFifo. Defaults to None.
            obj_types (list[type[np.ndarray]], optional): The buffer type of each new ObjectFifo. Defaults to None.
            names (Sequence[str | None] | None, optional): The name of each new ObjectFifo. If not given,
                each is named after this one, or by the Program if this one is unnamed.
                Defaults to None.
            to_stream (Sequence[TensorAccessPattern | None] | None, optional): Each new ObjectFifo's
                ``to_stream``; a padded one pads that output. Defaults to None.
            from_stream (Sequence[TensorAccessPattern | None] | None, optional): Each new
                ObjectFifo's ``from_stream_per_cons``. Defaults to None.
            plio (bool, optional): Set plio on each new ObjectFifo. Defaults to False.
            repeat_counts (Sequence[int | None] | None, optional): Per-sub-fifo MemTile DMA repeat count (see ObjectFifo.repeat_count). Defaults to None.
            pad_value (list[int] | None, optional): Per-sub-fifo per-element pad fill value (see ObjectFifo.pad_value). Defaults to None.

            channels (Sequence[int | None] | None, optional): Pin the hardware DMA
                channel each output ObjectFifo produces on, one per output.
                split() builds those producer handles itself, so this is the
                only place to say it. Defaults to None (all compiler-assigned).

        Raises:
            ValueError: Arguments are validated.

        Returns:
            list[ObjectFifo]: A list of newly constructed ObjectFifos whose producers are used in this split() operation.
        """
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
            names = [self._derived_name(f"_split{i}") for i in range(num_subfifos)]
        elif len(names) != num_subfifos:
            raise ValueError("Number of names does not match number of offsets")

        if to_stream is None:
            to_stream = [None] * num_subfifos
        elif len(to_stream) != num_subfifos:
            raise ValueError(
                "Number of to_stream arrays does not match number of offsets"
            )

        if from_stream is None:
            from_stream = [None] * num_subfifos
        elif len(from_stream) != num_subfifos:
            raise ValueError(
                "Number of from_stream arrays does not match number of offsets"
            )

        if repeat_counts is None:
            repeat_counts = [None for _ in range(num_subfifos)]
        elif len(repeat_counts) != num_subfifos:
            raise ValueError("Number of repeat_counts does not match number of offsets")

        if pad_value is None:
            pad_value = [0 for _ in range(num_subfifos)]
        elif len(pad_value) != num_subfifos:
            raise ValueError("Number of pad_value does not match number of offsets")

        # Create subfifos
        subfifos = []
        for i in range(num_subfifos):
            subfifos.append(
                ObjectFifo(
                    obj_types[i],
                    name=names[i],
                    depth=depths[i],
                    to_stream=to_stream[i],
                    from_stream_per_cons=from_stream[i],
                    plio=plio,
                    repeat_count=repeat_counts[i],
                    pad_value=pad_value[i],
                )
            )

        # Create link and set it as endpoints
        pinned: list[int | None] = (
            [None] * len(subfifos) if channels is None else list(channels)
        )
        if len(pinned) != len(subfifos):
            raise ValueError(
                f"split() got {len(pinned)} channels for {len(subfifos)} "
                "outputs; give one per output or none at all."
            )
        # A subfifo's producer handle is built here, so a caller wanting its
        # channel pinned has nowhere else to say it -- prod() refuses to re-pin.
        subfifo_prods = [s.prod(channel=c) for s, c in zip(subfifos, pinned)]
        _ = ObjectFifoLink(self, subfifo_prods, tile, [], offsets)
        return subfifos

    def forward(
        self,
        tile: Tile | None = AnyMemTile,
        obj_type: type[np.ndarray] | None = None,
        depth: int | None = None,
        name: str | None = None,
        to_stream: TensorAccessPattern | None = None,
        from_stream: TensorAccessPattern | None = None,
        plio: bool = False,
        repeat_count: int | None = None,
        pad_value: int = 0,
        channel: int | None = None,
    ) -> ObjectFifo:
        """Forward an ObjectFifoHandle of type consumer to a newly-constructed ObjectFifo.

        This is a special case of the split() operation where the consumer handle
        is forwarded to the producer of a newly-constructed ObjectFifo.

        Args:
            tile (Tile, optional): The tile for the Forward operation. Also accepts None (treated as AnyMemTile). Defaults to AnyMemTile.
            obj_type (type[np.ndarray] | None, optional): The object type of the new ObjectFifo. Defaults to None.
            depth (int | None, optional): The depth of the new ObjectFifo. Defaults to None.
            name (str | None, optional): The name of the new ObjectFifo. If None is given, it
                is named after this one, or by the Program if this one is unnamed. Defaults to None.
            to_stream (TensorAccessPattern | None, optional): The new ObjectFifo's ``to_stream``;
                a padded one pads the forwarded stream. Defaults to None.
            from_stream (TensorAccessPattern | None, optional): The new ObjectFifo's
                ``from_stream_per_cons``. Defaults to None.
            plio (bool, optional): Set plio on each new ObjectFifo. Defaults to False.
            repeat_count (int | None, optional): MemTile DMA repeat count for the new ObjectFifo (see ObjectFifo.repeat_count). Defaults to None.
            pad_value (int, optional): Per-element constant fill value for a padded
                ``to_stream`` (see ObjectFifo.pad_value). Defaults to 0.
            channel (int | None, optional): Pin the hardware DMA channel the
                forwarded ObjectFifo produces on. forward() builds that
                producer handle itself, so this is the only place to say it.
                Defaults to None (assigned by the compiler).

        Raises:
            ValueError: Arguments are Validated

        Returns:
            ObjectFifo: A newly constructed ObjectFifo whose producer used in this forward() operation.
        """
        if self._is_prod:
            raise ValueError(f"Cannot forward a {self.handle_type} ObjectFifoHandle")
        obj_types = [obj_type] if obj_type else None
        depths = [depth] if depth else None
        names = [name or self._derived_name("_fwd")]
        to_stream_arg = [to_stream] if to_stream is not None else None
        from_stream_arg = [from_stream] if from_stream is not None else None

        forward_fifo = self.split(
            [0],
            tile=tile,
            obj_types=obj_types,
            depths=depths,
            names=names,
            to_stream=to_stream_arg,
            from_stream=from_stream_arg,
            plio=plio,
            repeat_counts=[repeat_count] if repeat_count is not None else None,
            pad_value=[pad_value] if pad_value else None,
            channels=[channel] if channel is not None else None,
        )
        return forward_fifo[0]

    def resolve(
        self,
        loc: ir.Location | None = None,
        ip: ir.InsertionPoint | None = None,
    ) -> None:
        self._object_fifo.resolve(loc=loc, ip=ip)


class ObjectFifoLink(ObjectFifoEndpoint, Resolvable):
    """This is an object used internally by split(), join() and forward() operations."""

    def __init__(
        self,
        srcs: list[ObjectFifoHandle] | ObjectFifoHandle,
        dsts: list[ObjectFifoHandle] | ObjectFifoHandle,
        tile: Tile | None = AnyMemTile,
        src_offsets: list[int] | None = None,
        dst_offsets: list[int] | None = None,
    ):
        """Construct an ObjectFifoLink. This is either a many-to-one, one-to-many, or one-to-one operation.

        Args:
            srcs (list[ObjectFifoHandle] | ObjectFifoHandle): A list of consumer ObjectFifoHandles to link.
            dsts (list[ObjectFifoHandle] | ObjectFifoHandle): A list of producer ObjectFifoHandles to link.
            tile (Tile, optional): The tile where the link occurs. Also accepts None (treated as AnyMemTile). Defaults to AnyMemTile.
            src_offsets (list[int] | None, optional): If many sources, one offset per source
                is required to split the destination. Defaults to None (empty list).
            dst_offsets (list[int] | None, optional): If many destinations, one offset per
                destination is required to split the source. Defaults to None (empty list).

        Raises:
            ValueError: Arguments are validated.
        """
        self._site = SourceSite.capture()
        self._srcs = single_elem_or_list_to_list(srcs)
        self._dsts = single_elem_or_list_to_list(dsts)
        self._src_offsets = src_offsets if src_offsets is not None else []
        self._dst_offsets = dst_offsets if dst_offsets is not None else []
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
                "The number of source offsets does not match the number of sources"
            )
        if len(self._dst_offsets) > 0 and len(self._dst_offsets) != len(self._dsts):
            raise ValueError(
                "The number of destination offsets does not match the number of destinations"
            )
        self._op = None
        for s in self._srcs:
            s.endpoint = self
        for d in self._dsts:
            d.endpoint = self
        if tile is None:
            tile = AnyMemTile
        # Isolate singleton defaults, but retain user tiles shared with Workers
        # or other links so they resolve to the same logical tile.
        if any(
            tile is default for default in (AnyMemTile, AnyComputeTile, AnyShimTile)
        ):
            tile = tile.copy()
        # Respect explicit types and let the device infer fully placed tiles.
        placed = tile.col is not None and tile.row is not None
        if tile.tile_type is None and not placed:
            tile.tile_type = AIETileType.MemTile
        ObjectFifoEndpoint.__init__(self, tile)

    def _transfer_sizes(self, of: ObjectFifo) -> list[int]:
        """Return the elements each transfer of `of`'s end of the link moves, per segment it walks.

        The link holds one shared object. A join or distribute splits it at its
        offsets and each participant moves its own segment, while the single
        fifo on the other side walks every segment; a 1:1 link moves the
        larger object, or the arriving one when the departing walk pads.
        """
        srcs = [h._object_fifo for h in self._srcs]
        dsts = [h._object_fifo for h in self._dsts]
        if len(srcs) == len(dsts) == 1:
            in_size, out_size = math.prod(srcs[0].shape), math.prod(dsts[0].shape)
            padded = dsts[0].to_stream is not None and dsts[0].to_stream.padding
            return [out_size if out_size > in_size and not padded else in_size]
        shared, side, offsets = (
            (dsts[0], srcs, self._src_offsets)
            if len(srcs) > 1
            else (srcs[0], dsts, self._dst_offsets)
        )
        ends = [*offsets[1:], math.prod(shared.shape)]
        sizes = [end - start for start, end in zip(offsets, ends)]
        for i, participant in enumerate(side):
            if participant is of:
                return [sizes[i]]
        return sizes

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
                src_ops,
                dst_ops,
                self._src_offsets,
                self._dst_offsets,
                loc=loc or self._site.location(),
                ip=ip,
            )
