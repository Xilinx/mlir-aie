#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Row-streamed IRON lowering of a run of yolo26n operators onto one NPU2 array.

Every activation travels one image row at a time, as `[C/8][W][8]` int8 (the
layout the conv kernels read), through an ObjectFifo from the worker that
produces it to each worker that reads it. A slice of a tensor's channels is
a view into its rows; a concatenation is copied together, part by part, by
the worker that reads it; a nearest 2x upsample is copied pixel by pixel on
the way. Tensors from outside the run are filled from the host and tensors
the run hands on are drained to it, so any run of operators can be checked
against the oracle on its own.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from aie.dialects import memref  # pyright: ignore[reportAttributeAccessIssue]
from aie.dialects.arith import (  # pyright: ignore[reportMissingImports, reportAttributeAccessIssue]
    index_cast,
)
from aie.extras.dialects.arith import (  # pyright: ignore[reportMissingImports]
    constant,
    maxsi,
    minsi,
    select,
)
from aie.extras.types import i32  # pyright: ignore[reportMissingImports]
from aie.helpers.taplib import TensorAccessPattern
from aie.helpers.util import np_ndarray_type_to_memref_type
from aie.iron import (
    Buffer,
    ExternalFunction,
    ObjectFifo,
    TaskGroup,
    Transport,
    Worker,
    kernels,
)
from aie.iron.controlflow import else_, if_, range_
from aie.iron.dataflow.endpoint import ObjectFifoEndpoint
from aie.iron.dataflow.objectfifo import ObjectFifoHandle
from aie.iron.device import AnyComputeTile, AnyMemTile, Tile
from aie.utils.config import aie_kernels_dir
from yolo_params import Params
from yolo_spec import (
    Add,
    Attention,
    Concat,
    Conv,
    DetectPost,
    InputQuant,
    MaxPool,
    Network,
    Op,
    Slice,
    Tensor,
    Upsample,
)

KERNELS_DIR = Path(__file__).resolve().parent / "kernels"
IMAGE_ROW_BYTES = 640 * 3
L1_BYTES = 64 * 1024
# The stack and the core's own sections, and the allocator's alignment slack:
# enough beside two table kernels' tables.
L1_RESERVE = 3 * 1024
# A table kernel's table, in the core's own data; a tile counts those past two.
TABLE_BYTES = 1024
# A table kernel's second copy of its table, which makes its table map about
# 1.5x faster. A tile takes the copies where its plan leaves twice their 1 KB,
# beside two tables, and 2 KB more for each table past them: each copy is
# pinned to a bank of its own, which the allocator then packs around.
TABLE_COPY_BYTES = 2 * 1024
L1_LAG_BYTES = 12 * 1024
# Rows one memtile hold takes: a semaphore lock counts at most 63 objects,
# and they take at most half of a memtile's 512 KB, leaving it room for others.
MEMTILE_HOLD_ROWS = 63
MEMTILE_HOLD_BYTES = 256 * 1024
# Runs of operators, as network op index ranges, that one worker computes
# row by row (see SegmentDesign.fuse). Packed so the whole network takes 30
# of NPU2's 32 tiles in one configuration, with the two attention workers
# and the postprocess, each core's program within its 16 KB.
FUSED = (
    (0, 3),
    (3, 11),
    (11, 12),
    (12, 20),
    (20, 22),
    (24, 34),
    (35, 37),
    (37, 47),
    (47, 54),
    (54, 56),
    (57, 63),
    (66, 73),
    (75, 79),
    (79, 90),
    (92, 93),
    (95, 96),
    (96, 107),
    (107, 113),
    (113, 124),
    (124, 127),
    (129, 133),
    (136, 143),
    (143, 147),
    (147, 151),
    (151, 157),
    (157, 160),
    (160, 167),
)


def row(size: int) -> type[np.ndarray]:
    """The type of `size` bytes of a row."""
    return np.ndarray[(size,), np.dtype[np.int8]]


@dataclass(frozen=True)
class Part:
    """Bytes `[offset, offset + size)` of each row of `source`, twice as wide when `upsample`."""

    source: Tensor
    offset: int
    size: int
    upsample: bool = False

    @property
    def whole(self) -> bool:
        return (
            self.offset == 0
            and self.size == self.source.row_bytes
            and not self.upsample
        )


@dataclass(frozen=True)
class Fill:
    """One host transfer into a fifo: a tensor, in `layout`, at sequence step `step`.

    `rows` sends the tensor's rows as they are. `anchor_classes` sends a
    row `chunk` pixels at a time, each 8-channel group of those pixels in
    turn: the per-anchor records the Detect postprocess reads. `repeat`
    sends the whole tensor `chunk` times, as a conv's weights go once an
    output row. A step's transfers finish before the next step's start.
    """

    fifo: ObjectFifo
    tensor: Tensor
    layout: str = "rows"
    step: int = 0
    chunk: int = 1

    def tap(self, total: int, offset: int) -> TensorAccessPattern:
        """The pattern over a host buffer of `total` int32 words with the tensor at word `offset`."""
        t = self.tensor
        if self.layout == "anchor_classes":
            groups = t.channels8 // 8
            return TensorAccessPattern(
                (total,),
                offset,
                [t.height, t.width // self.chunk, groups, 2 * self.chunk],
                [groups * t.width * 2, 2 * self.chunk, t.width * 2, 1],
            )
        size = (IMAGE_ROW_BYTES if t.name == "images" else t.row_bytes) * t.height // 4
        if self.layout == "repeat":
            return TensorAccessPattern(
                (total,), offset, [1, self.chunk, 1, size], [0, 0, 0, 1]
            )
        return TensorAccessPattern((total,), offset, [1, 1, 1, size], [0, 0, 0, 1])


@dataclass
class Pending:
    """State `transfer_fill` hands `transfer_tail`: fill steps still to run.

    `staging` is the open staged-drains group, or `None` once awaited;
    `staged` names the staged tensors; `steps` the fill steps still to come;
    `drains` the output-drains group the last step's fills join.
    """

    filled: dict[ObjectFifo, ObjectFifoHandle]
    staging: TaskGroup | None
    staged: set[str]
    steps: list[int]
    drains: TaskGroup


def stream_bytes_of(tensor: Tensor) -> int:
    """The bytes of a row of `tensor`'s stream: the image comes as RGB pixels."""
    return IMAGE_ROW_BYTES if tensor.name == "images" else tensor.row_bytes


def pack_weights(op: Conv, weight: np.ndarray) -> np.ndarray:
    """OIYX int8 weights in the layout `op`'s kernel reads, channels padded to groups of 8."""
    if op.groups != 1:
        c = op.dst.channels8
        w = np.zeros((c, 3, 3), np.int8)
        w[: op.dst.channels] = weight[:, 0]
        return w.reshape(c // 8, 8, 3, 3).transpose(0, 2, 3, 1).reshape(-1)
    oc, ic = op.dst.channels8, op.src.channels8
    w = np.zeros((oc, ic, op.kernel, op.kernel), np.int8)
    w[: weight.shape[0], : weight.shape[1]] = weight
    w = w.reshape(oc // 8, 8, ic // 8, 8, op.kernel, op.kernel)
    # [OC/8][IC/8][ky][kx][ic8][oc8]; a 1x1 conv has no ky, kx.
    return np.ascontiguousarray(w.transpose(0, 2, 4, 5, 3, 1)).reshape(-1)


def view(element, offset, size: int, dtype: type = np.int8):
    """`size` elements of `dtype` in an int8 buffer (an acquired element or a Buffer) from byte `offset`, an int or an index value."""
    source = element.op if isinstance(element, Buffer) else element
    kind = np.ndarray[(size,), np.dtype[dtype]]
    at = constant(offset, index=True) if isinstance(offset, int) else offset
    return memref.view(np_ndarray_type_to_memref_type(kind), source, at, [])


class SegmentDesign:
    """Workers and fifos for a contiguous run of spec operators.

    Args:
        network: The whole network, to find each tensor's producer.
        params: Its parameters.
        ops: The run, in network order.
        outputs: Tensors to drain to the host.
        fused: Runs of operators, as network op index ranges `(first, stop)`,
            that one worker each computes row by row; see `fuse`. Runs not
            wholly inside `ops` are ignored; every operator outside them is a
            view, an attention block with its positional conv and add, or the
            Detect postprocess.
        holds: Rows each of a tensor's held readers waits for, from an
            earlier build; see `planned`.
        resident_bytes: The largest step of a fused worker's weight chunks a
            memtile holds.
        reads: Per tensor, its readers sharing its fifo and those taking rows
            over shared memory, from an earlier build; see `planned`.

    Attributes:
        inputs: Tensors filled from the host, in fill order; the image when
            the run starts at the input.
        fills: The host transfers into the run, see `Fill`; `in_fifos` the
            fifos they fill, in order.
        out_fifos: The fifo of each host output tensor.
        workers: Every worker of the run.
    """

    def __init__(
        self,
        network: Network,
        params: Params,
        ops: Sequence[Op],
        outputs: Sequence[Tensor],
        fused: Sequence[tuple[int, int]] = FUSED,
        holds: dict[str, set[int]] | None = None,
        resident_bytes: int = 200 * 1024,
        reads: dict[str, tuple[int, int]] | None = None,
    ) -> None:
        self.params = params
        # A fused worker's step of weight chunks up to this size rests in a
        # memtile, which sends it over and over; larger ones come from DDR.
        self.resident_bytes = resident_bytes
        self.resident: list[int] = []
        self.weight_streams = 0
        self.holds = 0
        # Rows each of a tensor's held readers waits for, from an earlier build
        # of the same run, so the first consumer makes the whole chain; and
        # what this build asks for.
        self.planned_holds = dict(holds or {})
        self.hold_rows: dict[str, set[int]] = {}
        self.read_stages: set[ObjectFifo] = set()
        # The memtile each chain of holds ends in, where a worker's own.
        self.stage_hubs: dict[ObjectFifo, Tile | None] = {}
        # Per tensor, how many readers share its fifo and how many take rows
        # over shared memory, each a fifo of its own, from an earlier build;
        # and what this build asks for. The producer copies each row into
        # the private fifos beyond its own, which serves a private reader
        # when no other reads it.
        self.planned_reads = dict(reads or {})
        self.shared_reads: dict[str, int] = {}
        self.private_reads: dict[str, int] = {}
        self.private_taken: dict[str, int] = {}
        self.private_fifos: dict[str, list[ObjectFifo]] = {}
        self.producer = {op.dst.name: op for op in network.ops}
        self.local = {op.dst.name for op in ops}
        self.fifos: dict[str, ObjectFifo] = {}
        self.functions: dict[str, ExternalFunction] = {}
        self.inputs: list[Tensor] = []
        self.fills: list[Fill] = []
        self.workers: list[Worker] = []
        self.stages: dict[str, list[tuple[ObjectFifo, int]]] = {}
        self.heights: dict[str, int] = {}
        # need[t][u][y]: the row of upstream stream u that row y of stream t waits for.
        self.need: dict[str, dict[str, np.ndarray]] = {}
        # Tensors this run makes and then reads back from DDR in a layout of
        # their own, as the Detect postprocess reads its records: drained, and
        # filled once every such drain is done.
        self.staged: list[tuple[ObjectFifoHandle, Tensor]] = []
        # Weights a run's cores read from DDR, see `weights`: each set's name
        # and size, and its bytes.
        self.loads: list[tuple[Tensor, np.ndarray]] = []
        index = {op.dst.name: i for i, op in enumerate(network.ops)}
        mine = {index[op.dst.name] for op in ops}
        fused = [(a, b) for a, b in fused if set(range(a, b)) <= mine]
        groups = {first: network.ops[first:stop] for first, stop in fused}
        inside = {i for first, stop in fused for i in range(first, stop)}
        # An attention worker holds every value, so it also makes the
        # positional conv over them and adds it in, as PSA does next.
        self.tails: dict[str, tuple[Conv, Add]] = {}
        for i in sorted(mine):
            op = network.ops[i]
            if not isinstance(op, Attention):
                continue
            pe, add = network.ops[i + 1 : i + 3]
            if not (
                isinstance(pe, Conv)
                and pe.src.name == op.values.name
                and isinstance(add, Add)
                and {add.a.name, add.b.name} == {op.dst.name, pe.dst.name}
                and {i + 1, i + 2} <= mine
            ):
                raise NotImplementedError(f"{op.name}: no positional conv and add next")
            if {i + 1, i + 2} & inside:
                raise ValueError(f"{pe.name}: the attention worker makes it")
            self.tails[op.name] = (pe, add)
            inside |= {i + 1, i + 2}
        drained = {t.name for t in outputs}
        for op in ops:
            i = index[op.dst.name]
            if i in groups:
                later = [o for o in ops if index[o.dst.name] >= i + len(groups[i])]
                self.fuse(groups[i], later, drained)
                continue
            if i in inside:
                continue
            match op:
                case Attention():
                    self.attention(op)
                case DetectPost():
                    self.detect(op)
                case Concat() | Slice() | Upsample():
                    pass
                case _:
                    raise ValueError(f"{op.name}: in no fused run")
        self.outputs = list(outputs)
        self.out_fifos = {t.name: self.fifo(t) for t in self.outputs}
        self.in_fifos = list(dict.fromkeys(f.fifo for f in self.fills))
        for name, (shared, private) in self.planned_reads.items():
            if private > (shared == 0) and name not in self.private_fifos:
                raise NotImplementedError(
                    f"{name}: its producer copies no rows to private readers"
                )

    @classmethod
    def planned(
        cls,
        network: Network,
        params: Params,
        ops: Sequence[Op],
        outputs: Sequence[Tensor],
        **options,
    ) -> SegmentDesign:
        """The run built twice: the second time every tensor's chain of holds is
        made whole by its first consumer, a stage ending at each reader's rows,
        and every producer makes the fifos its readers take."""
        design = cls(network, params, ops, outputs, **options)
        reads = {
            name: (design.shared_reads.get(name, 0), design.private_reads.get(name, 0))
            for name in design.shared_reads.keys() | design.private_reads.keys()
        }
        return cls(
            network,
            params,
            ops,
            outputs,
            holds=design.hold_rows,
            reads=reads,
            **options,
        )

    def handles(self) -> list:
        """The runtime sequence's fifo handles: every filled fifo, every drained one, then each staged tensor's drain."""
        return (
            [fifo.prod() for fifo in self.in_fifos]
            + [self.out_fifos[t.name].cons() for t in self.outputs]
            + [drain for drain, _ in self.staged]
        )

    def transfer_fill(
        self, handles: Sequence, locate: Callable[[Tensor], tuple]
    ) -> Pending:
        """This image's drains and its first fill step, the rest left for `transfer_tail`.

        `handles` and `locate` are as `transfer`'s. Issues the staged and the
        output drains (not awaited here) and the first fill step (awaited,
        unless it is also the only one), so its DMA traffic can run while an
        earlier image's `transfer_tail` is still awaited: the two interleave
        in a batch's runtime sequence, one image's fill alongside the one
        before it's tail.
        """
        filled = dict(zip(self.in_fifos, handles))
        drains = TaskGroup()
        staging = TaskGroup() if self.staged else None
        for (_, tensor), drain in zip(
            self.staged, handles[len(handles) - len(self.staged) :]
        ):
            buffer, total, offset = locate(tensor)
            drain.drain(
                buffer,
                TensorAccessPattern(
                    (total,),
                    offset,
                    [1, 1, 1, tensor.row_bytes * tensor.height // 4],
                    [0, 0, 0, 1],
                ),
                wait=True,
                group=staging,
            )
        staged = {tensor.name for _, tensor in self.staged}
        for tensor, handle in zip(self.outputs, handles[len(self.in_fifos) :]):
            buffer, total, offset = locate(tensor)
            size = self.host_bytes(tensor) // 4
            handle.drain(
                buffer,
                TensorAccessPattern((total,), offset, [1, 1, 1, size], [0, 0, 0, 1]),
                wait=True,
                group=drains,
            )
        steps = sorted({f.step for f in self.fills})
        step, rest = steps[0], steps[1:]
        if staging is not None and any(
            f.tensor.name in staged for f in self.fills if f.step == step
        ):
            staging.finish()
            staging = None
        fills = drains if not rest else TaskGroup()
        for fill in (f for f in self.fills if f.step == step):
            buffer, total, offset = locate(fill.tensor)
            filled[fill.fifo].fill(
                buffer, fill.tap(total, offset), wait=bool(rest), group=fills
            )
        if rest:
            fills.finish()
        return Pending(filled, staging, staged, rest, drains)

    def transfer_tail(
        self, pending: Pending, locate: Callable[[Tensor], tuple]
    ) -> TaskGroup:
        """This image's fill steps after the first, from `transfer_fill`'s state.

        Every step but the last finishes before the next starts, and a step
        that reads a staged tensor starts once every staged drain is done.
        The last step's fills join the output drains, left unfinished: the
        caller decides when to finish that group, so this image's tail can
        pipeline against the next image's `transfer_fill`.
        """
        filled, staging, staged, steps, drains = (
            pending.filled,
            pending.staging,
            pending.staged,
            pending.steps,
            pending.drains,
        )
        for step in steps:
            if staging is not None and any(
                f.tensor.name in staged for f in self.fills if f.step == step
            ):
                staging.finish()
                staging = None
            fills = drains if step == steps[-1] else TaskGroup()
            for fill in (f for f in self.fills if f.step == step):
                buffer, total, offset = locate(fill.tensor)
                filled[fill.fifo].fill(
                    buffer, fill.tap(total, offset), wait=step != steps[-1], group=fills
                )
            if step != steps[-1]:
                fills.finish()
        return drains

    def transfer(self, handles: Sequence, locate: Callable[[Tensor], tuple]) -> None:
        """One image's host transfers, inside its runtime sequence, start to finish.

        See `transfer_fill` and `transfer_tail`, run back to back with no
        other image's transfers between them.
        """
        self.transfer_tail(self.transfer_fill(handles, locate), locate).finish()

    def host_bytes(self, tensor: Tensor) -> int:
        """How many bytes `tensor` takes in a host buffer."""
        if tensor.name == "images":
            return IMAGE_ROW_BYTES * tensor.height
        if isinstance(self.producer.get(tensor.name), DetectPost):
            return 4 * 6 * tensor.height
        return tensor.row_bytes * tensor.height

    def fifo(self, tensor: Tensor) -> ObjectFifo:
        if tensor.name not in self.fifos:
            size = IMAGE_ROW_BYTES if tensor.name == "images" else tensor.row_bytes
            shared, private = self.planned_reads.get(tensor.name, (0, 0))
            # A fifo not read over shared memory goes by DMA even between
            # neighbours: each end then holds the objects its plan counts,
            # where one shared pool would sit in either tile.
            self.fifos[tensor.name] = ObjectFifo(
                row(size),
                name=f"t{len(self.fifos)}",
                transport=(
                    Transport.shared_mem()
                    if private and not shared
                    else Transport.dma()
                ),
            )
            if tensor.name not in self.local:
                self.inputs.append(tensor)
                self.fills.append(Fill(self.fifos[tensor.name], tensor))
        return self.fifos[tensor.name]

    def reader(self, tensor: Tensor) -> ObjectFifo:
        """`tensor`'s fifo, for one more reader sharing it."""
        self.shared_reads[tensor.name] = self.shared_reads.get(tensor.name, 0) + 1
        return self.fifo(tensor)

    def private(self, tensor: Tensor) -> ObjectFifo:
        """A fifo of `tensor`'s rows for one reader alone, over shared memory."""
        name = tensor.name
        self.private_reads[name] = self.private_reads.get(name, 0) + 1
        shared, private = self.planned_reads.get(name, (0, 0))
        if not private:
            return self.fifo(tensor)
        taken = self.private_taken.get(name, 0)
        self.private_taken[name] = taken + 1
        if not shared:
            return self.fifo(tensor) if taken == 0 else self.copies(tensor)[taken - 1]
        return self.copies(tensor)[taken]

    def copies(self, tensor: Tensor) -> list[ObjectFifo]:
        """The private fifos `tensor`'s producer copies each of its rows into."""
        name = tensor.name
        if name not in self.private_fifos:
            shared, private = self.planned_reads.get(name, (0, 0))
            self.private_fifos[name] = [
                ObjectFifo(
                    row(tensor.row_bytes),
                    name=f"{self.fifo(tensor).name}_p{i}",
                    transport=Transport.shared_mem(),
                )
                for i in range(private - (shared == 0))
            ]
        return self.private_fifos[name]

    def schedule(
        self,
        out: Tensor,
        windows: list[tuple[Tensor, np.ndarray, np.ndarray]],
        double: bool = True,
    ) -> dict[str, int]:
        """Record which rows of each upstream stream `out` waits for, and size this worker's input fifos.

        Output row y holds rows `lo..hi` of each input stream; `hi` may run
        past the last row, as a 3x3 window does at the bottom border, and
        such virtual rows count as following the last. Where two inputs come
        from one upstream stream, as the direct path of a residual add and
        its pair of 3x3 convs do, the worker holds one input's rows from `lo`
        while the other path pulls the upstream stream on to the row its own
        row y waits for, and the first input's producer keeps up; the fifo
        holds all those rows, or the upstream stream's producer and so both
        paths stall. Returns the depth per input stream: that span
        and a row in flight, or the window (doubled when it is one row and
        `double`).
        """

        def at(rows: np.ndarray, index: np.ndarray) -> np.ndarray:
            last = len(rows) - 1
            return rows[np.minimum(index, last)] + np.maximum(index - last, 0)

        def upstream(tensor: Tensor) -> dict[str, np.ndarray]:
            self.heights[tensor.name] = tensor.height
            return {tensor.name: np.arange(tensor.height)} | self.need.get(
                tensor.name, {}
            )

        need: dict[str, np.ndarray] = {}
        for source, _, hi in windows:
            for name, rows in upstream(source).items():
                need[name] = (
                    np.maximum(need[name], at(rows, hi))
                    if name in need
                    else at(rows, hi)
                )
        self.need[out.name] = need
        self.heights[out.name] = out.height
        depths: dict[str, int] = {}
        for source, lo, hi in windows:
            window = int((np.minimum(hi, source.height - 1) - lo + 1).max())
            span = window
            mine = upstream(source)
            for other, _, other_hi in windows:
                if other is source:
                    continue
                theirs = upstream(other)
                for shared in mine.keys() & theirs.keys():
                    # The rows of this input made from the shared stream's rows
                    # that the other input's row y has pulled through.
                    # A wait past the last row is a wait on the last row:
                    # the stream's producer still has to make it.
                    pulled = np.minimum(
                        at(theirs[shared], other_hi), self.heights[shared] - 1
                    )
                    # This input itself must take the pulled row; a path from
                    # the shared stream need only make the rows before it, to
                    # make room for it.
                    side = "right" if shared == source.name else "left"
                    ahead = np.searchsorted(mine[shared], pulled, side=side) - 1
                    span = max(span, int((ahead - lo + 1).max()) + 1)
            # A one-row window double-buffers; a wider one already overlaps.
            depths[source.name] = max(
                span + (window == 1 and double), depths.get(source.name, 0)
            )
        return depths

    def l1_rows(self, tensor: Tensor, depth: int, window: int) -> int:
        """How many of a `depth`-deep fifo's rows sit in the consuming worker's tile."""
        return (
            depth
            if depth <= window or depth * tensor.row_bytes <= L1_LAG_BYTES
            else window
        )

    def consumer(
        self, tensor: Tensor, depth: int, window: int, hub: Tile | None = None
    ):
        """A consumer handle on `tensor`'s rows, `depth` deep.

        Rows a worker only holds while another of its inputs catches up wait
        past the worker's own window once they would take more than
        `L1_LAG_BYTES` of its tile: in memtile holds, the last of them in
        memtile `hub` where given, so one worker's held inputs can take turns
        on one channel. Such a worker's rows that do not wait come over
        shared memory, so its channels are left to the held rows and its
        weights.
        """
        # An image row lands by phase, word k of every 4 pixels together (see
        # yolo_input.cc); the DMA writing L1 scatters it, as DDR reads that
        # gather it go a word at a time.
        quads = IMAGE_ROW_BYTES // 12
        from_stream = (
            TensorAccessPattern(
                (IMAGE_ROW_BYTES,), 0, [1, quads, 3, 4], [0, 4, 4 * quads, 1]
            )
            if tensor.name == "images"
            else None
        )
        if self.l1_rows(tensor, depth, window) == depth:
            if hub is not None and tensor.name not in self.local:
                # Rows the host fills come through the hub too: a shim shares
                # no memory, and the worker's channels are taken.
                fill = (
                    self.reader(tensor)
                    .cons(depth=2)
                    .forward(tile=hub, depth=2, name=f"fill{self.holds}")
                )
                self.holds += 1
                return fill.cons(depth=depth, from_stream=from_stream)
            if hub is not None:
                return self.private(tensor).cons(depth=depth)
            return self.reader(tensor).cons(depth=depth, from_stream=from_stream)
        stage = self.held(tensor, depth - window, hub)
        # A stage another worker or the chain's next stage reads would
        # broadcast, which takes no turns: a worker with a hub takes its own
        # forward of it there unless the stage is in that hub.
        if hub is not None and (
            stage in self.read_stages or self.stage_hubs.get(stage) != hub
        ):
            stage = stage.cons(depth=2).forward(
                tile=hub, depth=2, name=f"{stage.name}_{len(self.read_stages)}"
            )
        self.read_stages.add(stage)
        return stage.cons(depth=window)

    def held(self, tensor: Tensor, rows: int, hub: Tile | None = None) -> ObjectFifo:
        """A stage of `tensor`'s one chain of memtile holds holding at least `rows`.

        The first stage holding enough, more being only room to spare, or a
        new one extending the chain: a fifo feeds one link at most. The first
        reader makes the chain whole, a stage ending at each planned reader's
        rows: a stage's rows go on to every branch at once, so a reader taking
        them early must leave the chain before the rows a later one waits for.
        """
        self.hold_rows.setdefault(tensor.name, set()).add(rows)
        stages = self.stages.setdefault(tensor.name, [])
        if not stages:
            source, before = self.reader(tensor), 0
            for want in sorted(self.planned_holds.get(tensor.name, set()) | {rows}):
                source = self.hold(
                    source,
                    want - before,
                    stream_bytes_of(tensor),
                    hub=hub if want == rows else None,
                )
                stages.append((source, want))
                before = want
        fit = min(
            (st for st in stages if st[1] >= rows), key=lambda st: st[1], default=None
        )
        if fit is None:
            source, before = (
                max(stages, key=lambda st: st[1])
                if stages
                else (self.reader(tensor), 0)
            )
            fit = (
                self.hold(source, rows - before, stream_bytes_of(tensor), hub=hub),
                rows,
            )
            stages.append(fit)
        return fit[0]

    def hold(
        self, source: ObjectFifo, rows: int, row_bytes: int, hub: Tile | None = None
    ) -> ObjectFifo:
        """`source`'s rows of `row_bytes` held in memtiles, `rows` of them at once.

        A chain of memtile forwards, each holding at most MEMTILE_HOLD_ROWS
        (a lock's count) and MEMTILE_HOLD_BYTES: a pool deeper than its
        memtile's descriptors allow is walked whole by one descriptor a side.
        The last sends from memtile `hub`, where given.
        """
        fifo = source
        most = min(MEMTILE_HOLD_ROWS, max(1, MEMTILE_HOLD_BYTES // row_bytes))
        while rows > 0:
            take = min(rows, most)
            rows -= take
            fifo = fifo.cons(depth=take).forward(
                tile=AnyMemTile if rows or hub is None else hub,
                depth=2,
                name=f"hold{self.holds}",
            )
            self.holds += 1
        self.stage_hubs[fifo] = hub
        return fifo

    def parts(self, tensor: Tensor) -> list[Part]:
        """Where each row of `tensor` comes from, with views, concatenations and upsamples resolved."""
        op = self.producer.get(tensor.name) if tensor.name in self.local else None
        match op:
            case Slice():
                (parent,) = self.parts(op.src)
                if not parent.whole and parent.upsample or op.start % 8:
                    raise NotImplementedError(f"{op.name}: a slice of {parent}")
                return [
                    Part(
                        parent.source,
                        parent.offset + op.start * op.src.width,
                        tensor.row_bytes,
                    )
                ]
            case Concat():
                merged: list[Part] = []
                for part in (p for s in op.srcs for p in self.parts(s)):
                    last = merged[-1] if merged else None
                    if (
                        last is not None
                        and (last.source, last.upsample) == (part.source, part.upsample)
                        and last.offset + last.size == part.offset
                    ):
                        merged[-1] = Part(
                            last.source,
                            last.offset,
                            last.size + part.size,
                            last.upsample,
                        )
                    else:
                        merged.append(part)
                return merged
            case Upsample():
                (inner,) = self.parts(op.src)
                if not inner.whole:
                    raise NotImplementedError(f"{op.name}: an upsample of a view")
                return [Part(inner.source, 0, inner.size, upsample=True)]
            case _:
                return [Part(tensor, 0, tensor.row_bytes)]

    def function(
        self, name: str, source: str, arg_types: list, flags: list[str]
    ) -> ExternalFunction:
        """One ExternalFunction per distinct example kernel build, shared by every worker."""
        if name not in self.functions:
            self.functions[name] = ExternalFunction(
                name,
                source_file=str(KERNELS_DIR / source),
                arg_types=arg_types,
                compile_flags=flags,
                include_dirs=[aie_kernels_dir()],
            )
        return self.functions[name]

    def copier(self, part: Part) -> ExternalFunction:
        """The kernel that writes `part` into a row being assembled."""
        if part.upsample:
            groups = part.size // (8 * part.source.width)
            name = f"yolo_upsample_{groups}x{part.source.width}"
            flags = [
                f"-DYOLO_UPSAMPLE_WIDTH={part.source.width}",
                f"-DYOLO_UPSAMPLE_GROUPS={groups}",
            ]
        else:
            name = f"yolo_copy_{part.size}"
            flags = [f"-DYOLO_COPY_BYTES={part.size}"]
        out = 2 * part.size if part.upsample else part.size
        return self.function(
            name,
            "yolo_rows.cc",
            [row(part.size), row(out)],
            [*flags, f"-DYOLO_ROWS_SYMBOL={name}"],
        )

    def input_quant_kernel(self, op: InputQuant) -> ExternalFunction:
        """The kernel taking a row of uint8 RGB pixels to a row of the network input."""
        table = self.params.of(op.name, "table").astype(np.int64)
        pixels = np.arange(256)
        # The quantizer as a vector multiply and a rounding, saturating shift.
        found = [
            (mul, shift)
            for shift in range(8)
            for mul in range(1, 128)
            if np.array_equal(
                np.minimum((pixels * mul + (1 << shift >> 1)) >> shift, 127), table
            )
        ]
        if not found:
            raise ValueError(f"{op.name}: no multiply and shift gives its table")
        mul, shift = found[0]
        return self.function(
            "yolo_input_quant",
            "yolo_input.cc",
            [row(IMAGE_ROW_BYTES), row(op.dst.row_bytes)],
            [
                f"-DYOLO_INPUT_WIDTH={op.dst.width}",
                f"-DYOLO_INPUT_MUL={mul}",
                f"-DYOLO_INPUT_SHIFT={shift}",
            ],
        )

    def fuse(self, group: Sequence[Op], later: Sequence[Op], drained: set[str]) -> None:
        """One worker computing a run of operators row by row, each at its own rate.

        Every tensor the run reads or makes for itself waits in a ring of
        rows in the worker's tile; a row from another worker is copied into
        its ring as it arrives. The run steps at the rate of its shortest
        tensor, so at each step a tensor of `rate` times that height takes
        `rate` rows: another worker's arrive, and an operator computes its
        own, in network order, `delay` steps after the run starts. Delays are
        the latest that keep every reader's rows in, so rings stay short.
        Every step runs in one loop, so a ring is just the rows it holds at
        once.
        """
        p = self.params
        made = {op.dst.name: op for op in group}
        # Per operator, its inputs: the parts read, how far a row's window
        # reaches, the stride, and the height of the tensor read.
        steps: list[tuple[Op, list[tuple[list[Part], int, int, int]]]] = []
        for op in group:
            match op:
                case Conv():
                    steps.append(
                        (
                            op,
                            [
                                (
                                    self.parts(op.src),
                                    op.kernel // 2,
                                    op.stride,
                                    op.src.height,
                                )
                            ],
                        )
                    )
                case Add():
                    steps.append(
                        (
                            op,
                            [
                                (self.parts(op.a), 0, 1, op.a.height),
                                (self.parts(op.b), 0, 1, op.b.height),
                            ],
                        )
                    )
                case MaxPool():
                    steps.append((op, [(self.parts(op.src), 2, 1, op.src.height)]))
                case InputQuant():
                    images = Tensor("images", 3, op.dst.height, op.dst.width)
                    steps.append(
                        (
                            op,
                            [([Part(images, 0, IMAGE_ROW_BYTES)], 0, 1, op.dst.height)],
                        )
                    )
                case Slice() | Concat() | Upsample():
                    pass
                case _:
                    raise NotImplementedError(f"{op.name}: not fused yet")
        tensors: dict[str, Tensor] = {}
        readers: dict[str, list[tuple[Op, Part, int, int, int]]] = {}
        for op, inputs in steps:
            tensors[op.dst.name] = op.dst
            for parts, reach, stride, height_in in inputs:
                apart = (
                    isinstance(op, Conv)
                    and op.groups > 1
                    and op.stride == 1
                    and len({part.size for part in parts}) == 1
                )
                if reach and len(parts) > 1 and not apart:
                    raise NotImplementedError(f"{op.name}: a window over a concat")
                for part in parts:
                    if reach and part.upsample:
                        raise NotImplementedError(
                            f"{op.name}: a window over an upsample"
                        )
                    tensors[part.source.name] = part.source
                    readers.setdefault(part.source.name, []).append(
                        (op, part, reach, stride, height_in)
                    )

        outs = drained & made.keys()
        for op in later:
            match op:
                case Conv() | MaxPool():
                    read = [op.src]
                case Add():
                    read = [op.a, op.b]
                case Attention():
                    read = [op.qkv]
                case DetectPost():
                    read = [*op.scores, *op.boxes]
                case _:
                    read = []
            outs |= {
                part.source.name
                for t in read
                for part in self.parts(t)
                if part.source.name in made
            }
        if not outs <= {op.dst.name for op, _ in steps}:
            raise NotImplementedError(f"{group[0].name}: a fused run hands on a view")
        # In run order, so the design's fifos come out the same every build.
        outs = [name for name in made if name in outs]

        for op, _ in steps:
            if op.dst.name not in readers and op.dst.name not in outs:
                raise ValueError(f"{op.name}: nothing reads it")
        shortest = min(t.height for t in tensors.values())
        if any(t.height % shortest for t in tensors.values()):
            raise NotImplementedError(f"{group[0].name}: heights not multiples")
        rate = {name: t.height // shortest for name, t in tensors.items()}
        external = [name for name in tensors if name not in made]

        def window(part: Part, reach: int, stride: int, height_in: int, y: int):
            lo = max(stride * y - reach, 0)
            hi = min(stride * y + reach, height_in - 1)
            return (lo // 2, hi // 2) if part.upsample else (lo, hi)

        delay = dict.fromkeys(external, 0)

        def last(name: str, s: int) -> int:
            """The newest row of `name` in by the end of step `s`."""
            m = rate[name]
            return min(m * (s - delay[name]) + m - 1, tensors[name].height - 1)

        def ready(op: Op, inputs, d: int) -> bool:
            m = rate[op.dst.name]
            return all(
                window(part, reach, stride, height_in, y)[1]
                <= last(part.source.name, d + y // m)
                for y in range(op.dst.height)
                for parts, reach, stride, height_in in inputs
                for part in parts
            )

        for op, inputs in steps:
            d = 0
            while not ready(op, inputs, d):
                d += 1
            delay[op.dst.name] = d
        # Then each tensor as late as its readers allow, the last made first.
        for name in [op.dst.name for op, _ in reversed(steps)] + external:
            if name not in readers:
                continue
            m = rate[name]
            delay[name] = min(
                (m * (delay[op.dst.name] + y // rate[op.dst.name]) + m - 1 - hi) // m
                for op, part, reach, stride, height_in in readers[name]
                for y in range(op.dst.height)
                for hi in [window(part, reach, stride, height_in, y)[1]]
            )
        total = max(delay[name] + shortest for name in tensors)

        # Rows each tensor must hold at once: from the oldest a reader still
        # reads to the newest in.
        need: dict[str, int] = {}
        for name, uses in readers.items():
            most = 1
            for s in range(total):
                newest = last(name, s)
                if newest < 0:
                    continue
                oldest = [
                    window(part, reach, stride, height_in, max(y, 0))[0]
                    for op, part, reach, stride, height_in in uses
                    for y in [rate[op.dst.name] * (s - delay[op.dst.name])]
                    if y < op.dst.height
                ]
                if oldest:
                    most = max(most, newest - min(oldest) + 1)
            need[name] = most
        # A row another worker sends that is read only as it arrives is read
        # in place, in the fifo's own element.
        direct = {name for name in external if need[name] == 1 and rate[name] == 1}
        # So is a faster stream that one operator of its rate reads row for
        # row: each row taken as the operator computes from it.
        per_row = {
            name
            for name in external
            if need[name] == rate[name] > 1 and len(readers[name]) == 1
            for op, part, reach, stride, _ in readers[name]
            if reach == 0
            and stride == 1
            and not part.upsample
            and rate[op.dst.name] == rate[name]
        }
        direct |= per_row
        # Operators of one delay next to each other in the run's order: each
        # such stretch computes under one guard, so a row read in place must
        # be read within one stretch of its own delay.
        stretches = [
            list(stretch)
            for _, stretch in itertools.groupby(
                steps, key=lambda step: delay[step[0].dst.name]
            )
        ]
        direct = {
            name
            for name in direct
            if any(
                delay[stretch[0][0].dst.name] == delay[name]
                and {op.name for op, *_ in readers[name]}
                <= {op.name for op, _ in stretch}
                for stretch in stretches
            )
        }
        # A ring holds the bytes of a row its readers read: a tensor the run
        # makes, whole; another worker's, the span of the parts read.
        span: dict[str, tuple[int, int]] = {}
        for op, _ in steps:
            span[op.dst.name] = (0, op.dst.row_bytes)
        for name, uses in readers.items():
            if name not in made:
                span[name] = (
                    min(part.offset for _, part, *_ in uses),
                    max(part.offset + part.size for _, part, *_ in uses),
                )
        kept = {name: hi - lo for name, (lo, hi) in span.items()}
        rings = {name: rows for name, rows in need.items() if name not in direct}

        args: list = []
        at: dict = {}

        def arg(key, value):
            if key not in at:
                at[key] = len(args)
                args.append(value)

        # All inputs in one schedule, so two that share an upstream stream
        # hold the rows one waits for while the other's path catches up.
        depths: dict[str, int] = {}
        for out in outs:
            y = np.arange(tensors[out].height)
            steps_at = delay[out] + y // rate[out]
            windows = []
            for name in external:
                pulled = np.array([max(last(name, int(s)), 0) for s in steps_at])
                windows.append((tensors[name], pulled, pulled))
            for name, depth in self.schedule(tensors[out], windows).items():
                depths[name] = max(depths.get(name, 0), depth)

        # A worker's several held inputs come from one memtile, taking turns
        # on one channel (its weights take the other).
        hub = AnyMemTile.copy() if len(external) > 1 else None
        for name in external:
            source = tensors[name]
            arg(("in", name), self.consumer(source, depths[name], 1, hub))
            if name not in direct:
                lo, _ = span[name]
                arg(
                    ("copy", kept[name], False),
                    self.copier(Part(source, lo, kept[name])),
                )
        convs = [op for op, _ in steps if isinstance(op, Conv)]
        conv_of = {op.name: op for op in convs}
        tables = sum(1 for op in convs if op.act)
        weights_of = {
            op.name: pack_weights(op, p.of(op.name, "weight")) for op in convs
        }
        held = {
            op.name: weights_of[op.name].nbytes + 4 * op.dst.channels8 + 256 * op.act
            for op in convs
        }
        parts_of = {
            op.name: parts
            for op, ((parts, *_),) in ((o, i) for o, i in steps if isinstance(o, Conv))
        }
        concats = {
            name
            for name, parts in parts_of.items()
            if len(parts) > 1 or parts[0].upsample
        }
        # Where its row does not fit assembled, a 1x1 conv over a concat
        # reads each part where it waits, 8 output channels at a time (see
        # conv2dk1_i8_bias_slices_ref), at a call and a table map per chunk.
        concat_slices: dict[str, int] = {}
        for name in concats:
            op = conv_of[name]
            width = math.gcd(
                *(part.size // part.source.width for part in parts_of[name])
            )
            if op.kernel == 1 and width % 8 == 0 and width < op.src.channels8:
                concat_slices[name] = width
        upsampled = {
            name: [part for part in parts_of[name] if part.upsample]
            for name in parts_of
        }

        # Outputs leaving by DMA past a core's two channels take turns on one,
        # in the order the steps make them, so one a run makes before another
        # holds the rows it makes until the other's first.
        sent = [name for name in outs if self.planned_reads.get(name, (0, 0))[0]]
        ahead = {name: 0 for name in outs}
        if len(sent) > 2:
            latest = max(delay[name] for name in sent)
            for name in sent:
                ahead[name] = (latest - delay[name]) * rate[name]

        def plan(
            buffered: int, level: int
        ) -> tuple[dict[str, int], int, dict[str, int]] | None:
            """Weights to stream, what that leaves and the convs sliced, with outputs `buffered` deep.

            Weights stream from DDR a chunk of output channels at a time when
            the run's rows leave too little room: the largest sets first, each
            then taking two chunks of the tile instead of all its channels, and
            the chunks as large as what is left allows. At `level` 2 or 4 a
            streamed conv takes that fraction of its input channels a call,
            8 output channels at a time, so its chunks shrink as much.
            """
            slice_of = dict(sliced)

            def object_bytes(name: str) -> int:
                if name in slice_of:
                    return (conv_of[name].kernel ** 2 * slice_of[name] + 4) * 8
                return per_channel[name] * 8

            spare = (
                L1_BYTES
                - L1_RESERVE
                - sum(rows * kept[name] for name, rows in rings.items())
                - sum(
                    self.l1_rows(tensors[name], depths[name], 1)
                    * stream_bytes_of(tensors[name])
                    for name in external
                )
                - sum(
                    (
                        buffered * (1 + private - (shared == 0) if private else 1)
                        + ahead[name]
                    )
                    * tensors[name].row_bytes
                    for name in outs
                    for shared, private in [self.planned_reads.get(name, (0, 0))]
                )
                - sum(conv_of[name].src.row_bytes for name in assembled)
                - TABLE_BYTES * max(tables - 2, 0)
                - sum(
                    4 * conv_of[name].dst.width * 8
                    + sum(2 * part.size for part in upsampled[name])
                    for name in sliced
                )
                - sum(held.values())
            )
            # The weights stream holds two of its largest chunk.
            streamed: dict[str, int] = {}

            def stream_bytes(chunks: dict[str, int]) -> int:
                return 2 * max(
                    (object_bytes(n) * c // 8 for n, c in chunks.items()), default=0
                )

            for op in sorted(
                (op for op in convs if op.groups == 1), key=lambda op: -held[op.name]
            ):
                if spare - stream_bytes(streamed) >= 0:
                    break
                streamed[op.name] = 8
                spare += held[op.name] - 256 * op.act
                ic = op.src.channels8
                # A slice call reads its lines from where they wait, which an
                # assembled row's parts do not line up with.
                if (
                    level > 1
                    and op.name not in slice_of
                    and op.name not in assembled
                    and (ic // level) % 8 == 0
                ):
                    slice_of[op.name] = ic // level
                    spare -= 4 * op.dst.width * 8
            spare -= stream_bytes(streamed)
            if spare < 0:
                return None
            for name in [n for n in streamed if n not in slice_of]:
                oc = conv_of[name].dst.channels8
                for c in range(oc - 8, streamed[name], -8):
                    grown = dict(streamed, **{name: c})
                    extra = stream_bytes(grown) - stream_bytes(streamed)
                    if oc % c == 0 and extra <= spare:
                        spare -= extra
                        streamed[name] = c
                        break
            return streamed, spare, slice_of

        # Concats' rows assembled where that fits. Then outputs double-buffered
        # where that fits, else one row at a time: a row drains in a few
        # thousand cycles of the tens of thousands it takes. Then thinner
        # slices of the streamed weights, each costing cycles.
        for sliced in ({}, concat_slices):
            assembled = concats - sliced.keys()
            per_channel = {
                op.name: (
                    sliced[op.name] + 4
                    if op.name in sliced
                    else weights_of[op.name].nbytes // op.dst.channels8 + 4
                )
                for op in convs
            }
            fits = {}
            for level in (1, 2, 4):
                fits = {b: got for b in (2, 1) if (got := plan(b, level)) is not None}
                if fits:
                    break
            if fits:
                break
        # Two copies of the tables before double-buffered outputs: a table
        # map is most of a SiLU conv's cycles and reads its two halves from
        # two copies at once, while an output row drains long before the
        # step that makes the next is done.
        buffered = next(
            (
                b
                for b in (2, 1)
                if b in fits and fits[b][1] >= TABLE_COPY_BYTES * max(tables - 1, 1)
            ),
            max(fits, default=1),
        )
        planned = fits.get(buffered)
        if planned is None:
            raise ValueError(
                f"{group[0].name}: no chunks of its weights fit beside its rows"
                f" (KB: rings {sum(r * kept[n] for n, r in rings.items()) / 1024:.1f},"
                f" inputs {sum(self.l1_rows(tensors[n], depths[n], 1) * tensors[n].row_bytes for n in external) / 1024:.1f},"
                f" outputs {sum(tensors[n].row_bytes for n in outs) / 1024:.1f},"
                f" weights {sum(held.values()) / 1024:.1f}, of {(L1_BYTES - L1_RESERVE) / 1024:.0f};"
                f" largest rings: {sorted(((r * kept[n] // 1024, r, n.split('/')[-3:]) for n, r in rings.items()), reverse=True)[:3]})"
            )
        streamed, spare, sliced = planned
        for name in outs:
            arg(
                ("out", name),
                self.fifo(tensors[name]).prod(depth=buffered + ahead[name]),
            )
            privates = self.copies(tensors[name])
            for i, fifo in enumerate(privates):
                arg(("private", name, i), fifo.prod(depth=buffered))
            if name in rings or privates:
                size = tensors[name].row_bytes
                arg(("copy", size, False), self.copier(Part(tensors[name], 0, size)))
        for name, rows in rings.items():
            arg(("ring", name), Buffer(row(rows * kept[name])))
        copies = 2 if spare >= TABLE_COPY_BYTES * max(tables - 1, 1) else 1
        calls = {}
        # Per streamed conv: its chunks, each chunk's weight bytes, its channels.
        streams: dict[str, tuple[int, int, int]] = {}
        # Per streamed conv: its chunks, each its weights and then its bias.
        pieces: dict[str, np.ndarray] = {}
        for op, inputs in steps:
            match op:
                case Conv() if op.name in sliced:
                    width, ic, oc = op.src.width, op.src.channels8, op.dst.channels8
                    width_in = sliced[op.name]
                    bias = np.zeros(oc, np.int32)
                    bias[: op.dst.channels] = p.of(op.name, "bias")
                    table = (
                        [Buffer(initial_value=p.of(op.name, "table").astype(np.int8))]
                        if op.act
                        else []
                    )
                    # [OC/8][slice][slice/8][ky][kx][ic8][oc8]: a call's weights in a row.
                    blocks = weights_of[op.name].reshape(
                        oc // 8, ic // width_in, op.kernel**2 * width_in * 8
                    )
                    if op.name in streamed:
                        pieces[op.name] = np.stack(
                            [
                                np.concatenate(
                                    [block, bias[8 * c : 8 * c + 8].view(np.int8)]
                                )
                                for c in range(oc // 8)
                                for block in blocks[c]
                            ]
                        )
                        consts = table
                    else:
                        consts = [
                            Buffer(initial_value=blocks.reshape(-1)),
                            Buffer(initial_value=bias.view(np.int8)),
                            *table,
                        ]
                    k = (
                        kernels.conv2dk1_i8(
                            width,
                            ic,
                            8,
                            bias=True,
                            table=op.act,
                            rounding="conv_even",
                            table_copies=copies,
                            input_slice=width_in,
                        )
                        if op.kernel == 1
                        else kernels.conv2dk3(
                            width,
                            ic,
                            8,
                            bias=True,
                            table=op.act,
                            stride=op.stride,
                            rounding="conv_even",
                            table_copies=copies,
                            input_slice=width_in,
                        )
                    )
                    arg(
                        ("sums", op.name),
                        Buffer(np.ndarray[(op.dst.width * 8,), np.dtype[np.int32]]),
                    )
                    for part in upsampled[op.name]:
                        arg(
                            ("upsampled", op.name, part.source.name, part.offset),
                            Buffer(row(2 * part.size)),
                        )
                        arg(("copy", part.size, True), self.copier(part))
                    shift = int(p.of(op.name, "shift"))
                    scalars = (
                        [width, ic, 8, shift]
                        if op.kernel == 1
                        else [width, ic, 8, 3, 3, None, shift]
                    )
                case Conv():
                    width, ic, oc = op.src.width, op.src.channels8, op.dst.channels8
                    weights = weights_of[op.name]
                    bias = np.zeros(oc, np.int32)
                    bias[: op.dst.channels] = p.of(op.name, "bias")
                    table = (
                        [Buffer(initial_value=p.of(op.name, "table").astype(np.int8))]
                        if op.act
                        else []
                    )
                    chunk = streamed.get(op.name, oc)
                    if chunk == oc:
                        consts = [
                            Buffer(initial_value=weights),
                            Buffer(initial_value=bias),
                            *table,
                        ]
                    else:
                        per = weights.nbytes // (oc // chunk)
                        packed = np.concatenate(
                            [
                                np.concatenate(
                                    [
                                        weights[c * per : (c + 1) * per],
                                        bias[c * chunk : (c + 1) * chunk].view(np.int8),
                                    ]
                                )
                                for c in range(oc // chunk)
                            ]
                        )
                        pieces[op.name] = packed.reshape(oc // chunk, per + 4 * chunk)
                        consts = table
                        streams[op.name] = (oc // chunk, per, chunk)
                    if op.kernel == 1:
                        k = kernels.conv2dk1_i8(
                            width,
                            ic,
                            chunk,
                            bias=True,
                            table=op.act,
                            rounding="conv_even",
                            table_copies=copies,
                        )
                    elif op.groups == 1:
                        k = kernels.conv2dk3(
                            width,
                            ic,
                            chunk,
                            bias=True,
                            table=op.act,
                            stride=op.stride,
                            rounding="conv_even",
                            table_copies=copies,
                        )
                    elif op.stride == 1:
                        # Over a concat, each part on its own channels.
                        ((parts, *_),) = inputs
                        channels = parts[0].size // width
                        if len(parts) > 1:
                            per = 9 * channels
                            consts = [
                                Buffer(initial_value=c)
                                for i in range(len(parts))
                                for c in (
                                    weights[i * per : (i + 1) * per],
                                    bias[i * channels : (i + 1) * channels],
                                )
                            ] + table
                        k = kernels.bn_conv2dk3_dw(
                            width,
                            channels,
                            channels,
                            bias=True,
                            table=op.act,
                            rounding="conv_even",
                            table_copies=copies,
                        )
                    else:
                        raise NotImplementedError(
                            f"{op.name}: a strided depthwise conv"
                        )
                    ((parts, *_),) = inputs
                    if op.name in assembled:
                        arg(("assembled", op.name), Buffer(row(op.src.row_bytes)))
                        for part in parts:
                            arg(("copy", part.size, part.upsample), self.copier(part))
                    scalars = (
                        [width, ic, chunk, int(p.of(op.name, "shift"))]
                        if op.kernel == 1
                        else [
                            width,
                            ic,
                            chunk,
                            3,
                            3,
                            None,
                            int(p.of(op.name, "shift")),
                            0,
                        ]
                    )
                case Add():
                    sa, sb, so = (
                        p.scale(op.a.name),
                        p.scale(op.b.name),
                        p.scale(op.dst.name),
                    )
                    base = min(sa, sb, so)
                    k = kernels.add(op.dst.row_bytes, dtype=np.int8)
                    consts = []
                    scalars = [sa - base, sb - base, so - base]
                case MaxPool():
                    k = kernels.maxpool2d(op.dst.width, op.dst.channels8)
                    consts = []
                    scalars = []
                case InputQuant():
                    k = self.input_quant_kernel(op)
                    consts = []
                    scalars = []
                case _:
                    raise NotImplementedError(op.name)
            arg(("kernel", op.name), k)
            for i, c in enumerate(consts):
                arg(("const", op.name, i), c)
            calls[op.name] = (len(consts), scalars)

        streams_sliced = {name for name in sliced if name in streamed}
        # Every streamed conv's chunks come down one stream, each padded to the
        # largest, the same ones every step: a step drains the chunks of rows
        # it does not compute. A memtile holds one step's chunks where they
        # fit and sends them over and over; otherwise DDR sends them once a
        # step.
        if pieces:
            size = max(chunks.shape[1] for chunks in pieces.values())
            each = [
                chunk
                for op, _ in steps
                if op.name in pieces
                for _ in range(rate[op.dst.name])
                for chunk in pieces[op.name]
            ]
            sequence = np.zeros((len(each), size), np.int8)
            for i, chunk in enumerate(each):
                sequence[i, : chunk.size] = chunk
            sequence = sequence.reshape(-1)
            name = f"weights{self.weight_streams}"
            self.weight_streams += 1
            if sequence.size <= self.resident_bytes:
                wts = ObjectFifo(
                    row(sequence.size),
                    depth=1,
                    name=name,
                    consumer_obj_type=row(size),
                    init_values=[sequence],
                )
                wts.prod().endpoint = ObjectFifoEndpoint(AnyMemTile.copy())
                self.resident.append(sequence.size)
            else:
                wts = ObjectFifo(row(size), name=name)
                tensor = Tensor(f"{group[0].name}/weights", 8, 1, sequence.size // 8)
                self.loads.append((tensor, sequence))
                self.fills.append(Fill(wts, tensor, "repeat", chunk=total))
            arg(("weights",), wts.cons(depth=2))

        def body(*values):
            arrived = {}

            def slot(name, r, part=None):
                size = tensors[name].row_bytes
                offset, length = (0, size) if part is None else (part.offset, part.size)
                if name in arrived:
                    element = arrived[name]
                    return (
                        element
                        if part is None or part.whole
                        else view(element, offset, length)
                    )
                if part is None:
                    offset, length = span[name][0], kept[name]
                return view(
                    values[at[("ring", name)]],
                    (r % rings[name]) * kept[name] + offset - span[name][0],
                    length,
                )

            def borders(center, height_in, y, height, stride):
                """The rows above and below `center`, clamped at the map's edges, and a 3x3 kernel's border check for output row `y` of `height`."""
                above = maxsi(center - 1, constant(0, index=True))
                below = minsi(center + 1, constant(height_in - 1, index=True))
                inner = (
                    select(
                        y == height - 1,
                        constant(2, type=i32()),
                        constant(1, type=i32()),
                    )
                    if stride == 1
                    else constant(1, type=i32())
                )
                return above, below, select(y == 0, constant(0, type=i32()), inner)

            def convolve(op, k, rows, consts, target, scalars):
                if op.name not in streams:
                    k(*rows, *consts, target, *scalars)
                    return
                chunks, per, chunk = streams[op.name]
                size = op.dst.width * chunk
                wts = values[at[("weights",)]]
                for c in range_(chunks):
                    wc = wts.acquire(1)
                    k(
                        *rows,
                        view(wc, 0, per),
                        view(wc, per, chunk, np.int32),
                        *consts,
                        view(target, c * size, size),
                        *scalars,
                    )
                    wts.release(1)

            def compute(op, inputs, y, target):
                count, scalars = calls[op.name]
                k = values[at[("kernel", op.name)]]
                consts = [values[at[("const", op.name, i)]] for i in range(count)]
                match op:
                    case Conv() if op.name in sliced:
                        ((parts, _, stride, height_in),) = inputs
                        width_in = sliced[op.name]
                        size = width_in * op.src.width
                        # Per slice, the rows a call reads: a 1x1 its row, a
                        # 3x3 its three lines.
                        lines = []
                        tail = list(scalars)
                        if op.kernel == 3:
                            ((part,),) = [parts]
                            center = stride * y
                            above, below, check = borders(
                                center, height_in, y, op.dst.height, op.stride
                            )
                            tail = [*scalars[:5], check, scalars[6]]
                            rows = [
                                slot(part.source.name, r, part)
                                for r in (above, center, below)
                            ]
                            lines = [
                                [view(r, i * size, size) for r in rows]
                                for i in range(part.size // size)
                            ]
                        for part in parts if op.kernel == 1 else []:
                            r = stride * y // 2 if part.upsample else stride * y
                            source = slot(part.source.name, r, part)
                            if part.upsample:
                                up = values[
                                    at[
                                        (
                                            "upsampled",
                                            op.name,
                                            part.source.name,
                                            part.offset,
                                        )
                                    ]
                                ]
                                values[at[("copy", part.size, True)]](source, up)
                                source = up
                            count_in = (2 if part.upsample else 1) * part.size
                            lines += [
                                [view(source, i * size, size)]
                                for i in range(count_in // size)
                            ]
                        sums = values[at[("sums", op.name)]]
                        wts = (
                            values[at[("weights",)]]
                            if op.name in streams_sliced
                            else None
                        )
                        size_out = op.dst.width * 8
                        block = op.kernel**2 * width_in * 8
                        # Rolled over the output channels: a call per chunk of
                        # 8 and slice is too much program to unroll.
                        for c in range_(op.dst.channels8 // 8):
                            for i, x in enumerate(lines):
                                if wts is None:
                                    w = view(
                                        consts[0],
                                        c * (len(lines) * block) + i * block,
                                        block,
                                    )
                                    b = view(consts[1], c * 32, 8, np.int32)
                                    rest = consts[2:]
                                else:
                                    wc = wts.acquire(1)
                                    w = view(wc, 0, block)
                                    b = view(wc, block, 8, np.int32)
                                    rest = consts
                                k(
                                    *x,
                                    w,
                                    b,
                                    *rest,
                                    sums,
                                    view(target, c * size_out, size_out),
                                    *tail,
                                    i,
                                )
                                if wts is not None:
                                    wts.release(1)
                    case Conv() if op.kernel == 3 and len(inputs[0][0]) > 1:
                        ((parts, _, stride, height_in),) = inputs
                        above, below, check = borders(y, height_in, y, op.dst.height, 1)
                        size = parts[0].size
                        channels = size // op.src.width
                        weights, rest = (
                            consts[: 2 * len(parts)],
                            consts[2 * len(parts) :],
                        )
                        for i, part in enumerate(parts):
                            k(
                                *[
                                    slot(part.source.name, r, part)
                                    for r in (above, y, below)
                                ],
                                weights[2 * i],
                                weights[2 * i + 1],
                                *rest,
                                view(target, i * size, size),
                                op.src.width,
                                channels,
                                channels,
                                3,
                                3,
                                check,
                                *scalars[6:],
                            )
                    case Conv() if op.kernel == 3:
                        (((part,), _, stride, height_in),) = inputs
                        center = stride * y
                        above, below, check = borders(
                            center, height_in, y, op.dst.height, op.stride
                        )
                        rows = [
                            slot(part.source.name, r, part)
                            for r in (above, center, below)
                        ]
                        convolve(
                            op,
                            k,
                            rows,
                            consts,
                            target,
                            [*scalars[:5], check, *scalars[6:]],
                        )
                    case Conv():
                        ((parts, _, stride, _),) = inputs
                        if op.name not in assembled:
                            x = slot(parts[0].source.name, stride * y, parts[0])
                        else:
                            x = values[at[("assembled", op.name)]]
                            offset = 0
                            for part in parts:
                                r = stride * y // 2 if part.upsample else stride * y
                                width = 2 * part.size if part.upsample else part.size
                                values[at[("copy", part.size, part.upsample)]](
                                    slot(part.source.name, r, part),
                                    view(x, offset, width),
                                )
                                offset += width
                        convolve(op, k, [x], consts, target, scalars)
                    case Add():
                        ((pa,), *_), ((pb,), *_) = inputs
                        k(
                            slot(pa.source.name, y, pa),
                            slot(pb.source.name, y, pb),
                            target,
                            *scalars,
                        )
                    case InputQuant():
                        (((part,), *_),) = inputs
                        k(slot(part.source.name, y, part), target)
                    case MaxPool():
                        (((part,), *_),) = inputs
                        top = op.src.height - 1
                        lines = [
                            slot(
                                part.source.name,
                                minsi(
                                    maxsi(r, constant(0, index=True)),
                                    constant(top, index=True),
                                ),
                                part,
                            )
                            for r in [y + d for d in range(-2, 3)]
                        ]
                        k(*lines, target)

            def stepped(s):
                """Step `s`, an index value: each row it takes, and each stretch of operators, under a guard that its rows are in their maps."""
                for name in external:
                    if name in direct:
                        continue
                    m = rate[name]
                    for j in range(m):
                        r = s * m + (j - m * delay[name])
                        with if_((r >= 0) & (r < tensors[name].height)):
                            handle = values[at[("in", name)]]
                            x = handle.acquire(1)
                            if kept[name] != tensors[name].row_bytes:
                                x = view(x, span[name][0], kept[name])
                            values[at[("copy", kept[name], False)]](x, slot(name, r))
                            handle.release(1)
                for stretch in stretches:
                    d = delay[stretch[0][0].dst.name]
                    names = {op.name for op, _ in stretch}
                    reads = [
                        name
                        for name in external
                        if name in direct
                        and name not in per_row
                        and delay[name] == d
                        and readers[name][0][0].name in names
                    ]
                    # Every tensor's height is its rate times the shortest's.
                    with if_((s >= d) & (s < d + shortest)) as branch:
                        for name in reads:
                            arrived[name] = values[at[("in", name)]].acquire(1)
                        for op, inputs in stretch:
                            name = op.dst.name
                            m = rate[name]
                            rows = [n for n in per_row if readers[n][0][0] is op]
                            for j in range(m):
                                y = s * m + (j - m * d)
                                handle = (
                                    values[at[("out", name)]] if name in outs else None
                                )
                                o = handle.acquire(1) if handle is not None else None
                                target = slot(name, y) if name in rings else o
                                for n in rows:
                                    arrived[n] = values[at[("in", n)]].acquire(1)
                                compute(op, inputs, y, target)
                                for n in rows:
                                    values[at[("in", n)]].release(1)
                                    del arrived[n]
                                if handle is None:
                                    continue
                                copy = ("copy", tensors[name].row_bytes, False)
                                if name in rings:
                                    values[at[copy]](slot(name, y), o)
                                for i in range(len(self.private_fifos[name])):
                                    private = values[at[("private", name, i)]]
                                    values[at[copy]](o, private.acquire(1))
                                    private.release(1)
                                handle.release(1)
                        for name in reads:
                            values[at[("in", name)]].release(1)
                            del arrived[name]
                    drained = sum(
                        rate[op.dst.name] * len(pieces[op.name])
                        for op, _ in stretch
                        if op.name in pieces
                    )
                    if drained:
                        with else_(branch):
                            wts = values[at[("weights",)]]
                            for _ in range_(drained):
                                wts.acquire(1)
                                wts.release(1)

            # The steps run as one loop: unrolled, the steps at either end of
            # a map take more program than a tile has.
            for s in range_(total):
                stepped(s)

        self.workers.append(Worker(body, args))

    def attention(self, op: Attention) -> None:
        """One worker for every head, with PSA's positional conv and add after it.

        Whole qkv rows come in as they are made, to store each head's keys
        and values; the map's queries wait in a memtile, only their columns
        sent on, and come once every key is in. The worker then makes the
        positional conv over the values it stores and adds it in: its row is
        the add's.
        """
        p = self.params
        (part,) = self.parts(op.qkv)
        if not part.whole:
            raise NotImplementedError(f"{op.name}: attention over a view")
        qkv = part.source
        height, width = qkv.height, qkv.width
        kd, hd = op.key_dim, op.head_dim
        per_head = (2 * kd + hd) * width
        s_qkv, s_q = p.scale(qkv.name), int(p.of(op.name, "q_scale"))
        s_s, s_p = int(p.of(op.name, "score_scale")), int(p.of(op.name, "prob_scale"))
        q_mul = int(p.of(op.name, "q_mul"))
        if not -128 <= q_mul <= 127:
            raise ValueError(f"{op.name}: the query multiplier {q_mul} is not an int8")
        shifts = [
            s_q - s_qkv - int(p.of(op.name, "q_mul_scale")),
            s_s - s_q - s_qkv,
            -s_p,
            p.scale(op.dst.name) - s_qkv - s_p,
        ]
        k = kernels.attention_i8(height * width, kd, hd, width)
        upstream = {qkv.name: np.full(height, height - 1)}
        for name, rows in self.need.get(qkv.name, {}).items():
            upstream[name] = np.full(height, rows[height - 1])
        size = hd * width
        pe, add = self.tails[op.name]
        if pe.kernel != 3 or pe.stride != 1 or pe.groups != pe.src.channels:
            raise NotImplementedError(f"{pe.name}: not a 3x3 depthwise conv")
        if pe.act:
            raise NotImplementedError(f"{pe.name}: an activation")
        self.need[add.dst.name] = upstream
        self.heights[add.dst.name] = height
        self.heights[qkv.name] = height
        self.fifo(add.dst)
        # A walk over the map's stream of rows: its memtile holds them all.
        queries = (
            self.reader(qkv)
            .cons(depth=height)
            .forward(
                depth=2,
                name=f"queries{self.holds}",
                obj_type=row(op.heads * kd * width),
                to_stream=TensorAccessPattern(
                    (height * qkv.row_bytes,),
                    0,
                    [height, op.heads, kd * width],
                    [qkv.row_bytes, per_head, 1],
                ),
            )
        )
        pe_weights = pack_weights(pe, p.of(pe.name, "weight"))
        pe_bias = np.zeros(pe.dst.channels8, np.int32)
        pe_bias[: pe.dst.channels] = p.of(pe.name, "bias")
        pe_shift = int(p.of(pe.name, "shift"))
        k_pe = kernels.bn_conv2dk3_dw(width, 8, 8, bias=True, rounding="conv_even")
        sa, sb, so = (
            p.scale(add.a.name),
            p.scale(add.b.name),
            p.scale(add.dst.name),
        )
        base = min(sa, sb, so)
        k_add = kernels.add(add.dst.row_bytes, dtype=np.int8)
        attended_first = add.a.name == op.dst.name
        # The map's stores outgrow one tile: the keys stay in the worker's
        # own, the values go to a neighbour that runs no core. Placement
        # cannot see a core's pinned tables before it is compiled, so a
        # store beside another worker could overfill that tile.
        home, store = AnyComputeTile.copy(), AnyComputeTile.copy()
        stores = [
            ObjectFifo(
                np.ndarray[(height * width * n,), np.dtype[np.int8]],
                depth=1,
                name=f"{name}{h}_{self.holds}",
                disable_synchronization=True,
                delegate_tile=tile,
            )
            for name, n, tile in (("keys", kd, home), ("values", hd, store))
            for h in range(op.heads)
        ]

        copies = self.copies(add.dst)

        def together(rows, queries, out, k, load, k_pe, k_add, *rest):
            table, w, b, attended, encoded = rest[:5]
            filled = [s.acquire(1) for s in rest[5 : 5 + 2 * op.heads]]
            for r in range_(height):
                x = rows.acquire(1)
                for h in range(op.heads):
                    load(
                        view(x, h * per_head + kd * width, kd * width),
                        view(x, h * per_head + 2 * kd * width, hd * width),
                        filled[h],
                        filled[op.heads + h],
                        index_cast(i32(), r),
                    )
                rows.release(1)
            stored = [s.acquire(1) for s in rest[5 + 2 * op.heads : 5 + 4 * op.heads]]
            copier = rest[5 + 4 * op.heads : 6 + 4 * op.heads]
            privates = rest[6 + 4 * op.heads :]
            kts, vs = stored[: op.heads], stored[op.heads :]

            def answer(lines, check):
                q = queries.acquire(1)
                o = out.acquire(1)
                for h in range(op.heads):
                    k(
                        view(q, h * kd * width, kd * width),
                        kts[h],
                        vs[h],
                        table,
                        view(attended, h * size, size),
                        q_mul,
                        *shifts,
                    )
                    # A store keeps each 8 values of a row together.
                    for g in range_(hd // 8):
                        k_pe(
                            *[
                                view(vs[h], g * (height * width * 8) + y, width * 8)
                                for y in lines
                            ],
                            view(w, g * 72 + h * hd * 9, 72),
                            view(b, g * 32 + h * hd * 4, 8, np.int32),
                            view(encoded, g * (width * 8) + h * size, width * 8),
                            width,
                            8,
                            8,
                            3,
                            3,
                            check,
                            pe_shift,
                            0,
                        )
                pair = (attended, encoded) if attended_first else (encoded, attended)
                k_add(*pair, o, sa - base, sb - base, so - base)
                for private in privates:
                    copier[0](o, private.acquire(1))
                    private.release(1)
                queries.release(1)
                out.release(1)

            line = width * 8
            answer([0, 0, line], 0)
            for r in range_(1, height - 1):
                answer([(r - 1) * line, r * line, (r + 1) * line], 1)
            answer([(height - 2) * line, (height - 1) * line, (height - 1) * line], 2)

        self.workers.append(
            Worker(
                together,
                [
                    # One of each a time: the stores fill the tile.
                    self.reader(qkv).cons(depth=1),
                    queries.cons(depth=1),
                    self.fifo(add.dst).prod(depth=2),
                    k,
                    k.load_kv,
                    k_pe,
                    k_add,
                    Buffer(initial_value=p.of(op.name, "exp_table").astype(np.int32)),
                    Buffer(initial_value=pe_weights),
                    Buffer(initial_value=pe_bias.view(np.int8)),
                    Buffer(row(op.dst.row_bytes)),
                    Buffer(row(pe.dst.row_bytes)),
                    *[s.prod() for s in stores],
                    *[s.cons() for s in stores],
                    *(
                        [self.copier(Part(add.dst, 0, add.dst.row_bytes))]
                        + [fifo.prod(depth=2) for fifo in copies]
                        if copies
                        else []
                    ),
                ],
                # aiecc measures a 6.4 KB stack and 6.1 KB of the kernel's
                # own tables; stated, placement counts them beside the keys.
                stack_size=7 * 1024,
                data_size=13 * 512,
                tile=home,
            )
        )

    def detect(self, op: DetectPost) -> None:
        """The end-to-end postprocess on one worker, from per-anchor records the host sends twice.

        The first pass finds each anchor's best class score and picks the
        300 best anchors; the second gathers those anchors' class logits and
        box distances; the worker then picks the 300 best (anchor, class)
        scores and writes their boxes, scores and classes. See
        kernels/yolo_detect.cc.
        """
        p = self.params
        tables = p.of(op.name, "sigmoid").astype(np.float32)
        values = np.unique(tables)
        rank = np.searchsorted(values, tables).astype(np.int16)
        if values.size > 768:
            raise ValueError(
                f"{op.name}: {values.size} distinct scores, the kernel ranks 768"
            )
        chunk = 20
        anchors = sum(t.height * t.width for t in op.scores)
        records = ObjectFifo(row(chunk * 80), name="detect_classes")
        boxes = ObjectFifo(row(chunk * 8), name="detect_boxes")
        # Made here, the Detect outputs go to DDR first, and the records come
        # back once they all have, after the run's own inputs (step 0).
        made = [t for t in (*op.scores, *op.boxes) if t.name in self.local]
        if made and len(made) < len(op.scores) + len(op.boxes):
            raise NotImplementedError(f"{op.name}: some Detect outputs made here")
        first = 1 if made else 0
        for tensor in made:
            self.staged.append((self.reader(tensor).cons(depth=2), tensor))
        for step in (first, first + 1):
            for tensor in op.scores:
                self.fills.append(
                    Fill(records, tensor, "anchor_classes", step, chunk=chunk)
                )
        for tensor in op.boxes:
            self.fills.append(Fill(boxes, tensor, step=first + 1))
        if not made:
            self.inputs += [*op.scores, *op.boxes]
        self.fifos[op.dst.name] = ObjectFifo(
            np.ndarray[(6 * op.dst.height,), np.dtype[np.int32]], name="detections"
        )
        # How far below an anchor's highest logit a higher score can lie: max
        # looks only that far when the highest's rank is not its peak.
        reach = 0
        for level in rank:
            for m in np.nonzero(level != np.maximum.accumulate(level))[0]:
                reach = max(reach, int(m - np.nonzero(level[:m] > level[m])[0].min()))
        flags = [f"-DYOLO_DETECT_CHUNK={chunk}", f"-DYOLO_DETECT_REACH={reach}"]
        best_t = np.ndarray[(anchors,), np.dtype[np.int16]]
        ranks_t = np.ndarray[(2 * 768,), np.dtype[np.int16]]
        words_t = np.ndarray[(768,), np.dtype[np.int32]]
        k_max = self.function(
            "yolo_detect_max",
            "yolo_detect.cc",
            [row(chunk * 80), ranks_t, best_t, np.int32],
            flags,
        )
        bind = k_max.object_file.bind
        keep_t = np.ndarray[(300,), np.dtype[np.int16]]
        count_t = np.ndarray[(4,), np.dtype[np.int32]]
        # A chunk's picked bits are read as a word, past the last byte.
        picked_t = np.ndarray[(anchors // 8 + 4,), np.dtype[np.uint8]]
        k_select = bind(
            "yolo_detect_select",
            [best_t, words_t, keep_t, picked_t, keep_t, count_t],
        )
        k_gather = bind(
            "yolo_detect_gather",
            [
                row(chunk * 80),
                row(chunk * 8),
                picked_t,
                keep_t,
                row(300 * 80),
                row(300 * 4),
                count_t,
                np.int32,
            ],
        )
        k_finish = bind(
            "yolo_detect_finish",
            [row(300 * 80), row(300 * 4), keep_t, ranks_t, words_t, words_t, keep_t]
            + [
                np.ndarray[(6 * op.dst.height,), np.dtype[np.int32]],
                np.int32,
                np.int32,
                np.int32,
            ],
        )
        shifts = [p.scale(t.name) for t in op.boxes]
        bits = np.zeros(768, np.int32)
        bits[: tables.size] = tables.view(np.int32).reshape(-1)
        # Each level's ranks, then its peaks: the highest rank of any logit up to each.
        rank_table = np.zeros(2 * 768, np.int16)
        rank_table[: rank.size] = rank.reshape(-1)
        rank_table[768 : 768 + rank.size] = np.maximum.accumulate(
            rank, axis=-1
        ).reshape(-1)
        n = anchors // chunk

        def body(cls, box, out, k_max, k_select, k_gather, k_finish, *b):
            ranks, hist, first, picked, rows, count, cand, dist, sigmoid, best = b
            # Each anchor's best rank waits in the candidates' memory, which
            # the gather fills only after the select has read it.
            top = view(cand, 0, anchors, np.int16)
            for i in range_(n):
                r = cls.acquire(1)
                k_max(r, ranks, top, index_cast(i32(), i))
                cls.release(1)
            k_select(top, hist, first, picked, rows, count)
            for i in range_(n):
                r = cls.acquire(1)
                d = box.acquire(1)
                k_gather(r, d, picked, rows, cand, dist, count, index_cast(i32(), i))
                cls.release(1)
                box.release(1)
            o = out.acquire(1)
            k_finish(cand, dist, first, ranks, sigmoid, hist, best, o, *shifts)
            out.release(1)

        buffers = [
            Buffer(initial_value=rank_table),
            Buffer(words_t),
            Buffer(keep_t),
            Buffer(picked_t),
            Buffer(keep_t),
            Buffer(count_t),
            Buffer(row(300 * 80)),
            Buffer(row(300 * 4)),
            Buffer(initial_value=bits),
            Buffer(keep_t),
        ]
        self.workers.append(
            Worker(
                body,
                [
                    records.cons(depth=2),
                    boxes.cons(depth=2),
                    self.fifos[op.dst.name].prod(depth=1),
                    k_max,
                    k_select,
                    k_gather,
                    k_finish,
                    *buffers,
                ],
            )
        )
