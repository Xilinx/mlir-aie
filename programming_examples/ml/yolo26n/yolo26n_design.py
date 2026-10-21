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
# A memtile channel has 24 buffer descriptors, one per staged row.
MEMTILE_LAG_ROWS = 8
# Rows one memtile hold takes: a semaphore lock counts at most 63 objects,
# and they take at most half of a memtile's 512 KB, leaving it room for others.
MEMTILE_HOLD_ROWS = 63
MEMTILE_HOLD_BYTES = 256 * 1024
# Ring bytes a fused run spends to roll a shorter period: less code per step.
RING_SLACK_BYTES = 4 * 1024
# What one buffering worker's ring takes of its tile.
BUFFER_TILE_BYTES = 48 * 1024
# Runs of operators, as network op index ranges, that one worker computes
# row by row (see SegmentDesign.fuse): the bottlenecks of model.2, 4 and
# 8, the C3k blocks of model.6, 13, 16 and 19, the PSA blocks' FFNs, and
# the Detect head's P3 box branch and P4 and P5 branches. P3's class
# branch stays apart: fused, its run is its segment's slowest worker.
FUSED = (
    (6, 9),
    (15, 18),
    (24, 28),
    (28, 34),
    (78, 82),
    (82, 88),
    (95, 99),
    (99, 105),
    (41, 44),
    (44, 47),
    (65, 67),
    (68, 71),
    (112, 116),
    (116, 122),
    (135, 137),
    (138, 141),
    (143, 146),
    (151, 154),
    (154, 156),
    (156, 159),
    (160, 162),
    (162, 164),
    (164, 167),
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


class RowInput:
    """How a row-wise worker gets one input tensor's row at each step.

    The worker steps through `period` output rows per iteration: 2 when a
    part is upsampled, whose source row then serves both.
    """

    def __init__(
        self, design: SegmentDesign, tensor: Tensor, into_output: bool = False
    ) -> None:
        self.parts = design.parts(tensor)
        self.sources = list(dict.fromkeys(p.source for p in self.parts))
        self.handles: list = []
        self.period = 2 if any(p.upsample for p in self.parts) else 1
        self.assembled = into_output or len(self.parts) > 1 or self.parts[0].upsample
        self.copiers = [design.copier(p) for p in self.parts] if self.assembled else []
        self.buffer = (
            [Buffer(row(tensor.row_bytes))]
            if self.assembled and not into_output
            else []
        )

    def windows(self, height: int) -> list[tuple[Tensor, np.ndarray, np.ndarray]]:
        """Each source stream's rows held for output rows `0..height`, as (source, lo, hi)."""
        y = np.arange(height)
        out = []
        for source in self.sources:
            upsampled = any(p.upsample for p in self.parts if p.source == source)
            rows = y // 2 if upsampled else y
            out.append((source, rows, rows))
        return out

    def connect(self, design: SegmentDesign, depths: dict[str, int]) -> None:
        self.handles = [design.consumer(s, depths[s.name], 2) for s in self.sources]

    @property
    def fn_args(self) -> list:
        return [*self.handles, *self.copiers, *self.buffer]

    def bind(self, args: list) -> BoundRowInput:
        n = len(self.handles)
        handles = dict(zip(self.sources, args[:n]))
        copiers = args[n : n + len(self.copiers)]
        buffer = args[n + len(self.copiers)] if self.buffer else None
        del args[: len(self.fn_args)]
        return BoundRowInput(self, handles, copiers, buffer)


class BoundRowInput:
    """A RowInput inside its worker's body, on the worker's own handles."""

    def __init__(self, spec: RowInput, handles: dict, copiers: list, buffer) -> None:
        self.spec = spec
        self.handles = handles
        self.copiers = copiers
        self.buffer = buffer
        self.held: dict = {}

    def acquire(self, phase: int, into=None):
        """Take this step's rows and return the assembled one, built in `into` if given."""
        rows = {}
        for source, handle in self.handles.items():
            upsampled = any(p.upsample for p in self.spec.parts if p.source == source)
            if phase == 0 or not upsampled:
                rows[source] = handle.acquire(1)
            else:
                rows[source] = self.held[source]
        self.held = rows
        if not self.spec.assembled:
            part = self.spec.parts[0]
            element = rows[part.source]
            return element if part.whole else view(element, part.offset, part.size)
        target = self.buffer if into is None else into
        at = 0
        for part, copier in zip(self.spec.parts, self.copiers):
            element = rows[part.source]
            whole = part.offset == 0 and part.size == part.source.row_bytes
            size = 2 * part.size if part.upsample else part.size
            copier(
                element if whole else view(element, part.offset, part.size),
                view(target, at, size),
            )
            at += size
        return target

    def release(self, phase: int) -> None:
        for source, handle in self.handles.items():
            upsampled = any(p.upsample for p in self.spec.parts if p.source == source)
            if phase == self.spec.period - 1 or not upsampled:
                handle.release(1)


class SegmentDesign:
    """Workers and fifos for a contiguous run of spec operators.

    Args:
        network: The whole network, to find each tensor's producer.
        params: Its parameters.
        ops: The run, in network order.
        outputs: Tensors to drain to the host.
        fused: Runs of operators, as network op index ranges `(first, stop)`,
            that one worker each computes row by row; see `fuse`. Runs not
            wholly inside `ops` are ignored.
        attention_split: Workers per attention head, off chip.
        on_chip: Whether rows that wait stay in memtiles rather than DDR.
        holds: Rows each of a tensor's held readers waits for, from an
            earlier build; see `planned`.
        resident_bytes: The largest step of a fused worker's weight chunks a
            memtile holds on chip.
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
        attention_split: int = 2,
        on_chip: bool = False,
        holds: dict[str, set[int]] | None = None,
        resident_bytes: int = 0,
        reads: dict[str, tuple[int, int]] | None = None,
    ) -> None:
        self.params = params
        # On chip, a fused worker's step of weight chunks up to this size rests
        # in a memtile, which sends it over and over; larger ones come from DDR.
        self.resident_bytes = resident_bytes
        self.resident: list[int] = []
        self.weight_streams = 0
        self.attention_split = attention_split
        # Rows a consumer waits for past its own window, a whole map or a lag,
        # held in memtiles rather than in DDR or in buffering workers.
        self.on_chip = on_chip
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
        # Tensors that travel inside another stream's rows, as SPPF's pools do,
        # or across several, as attention's heads do.
        self.alias: dict[str, Part] = {}
        self.virtual: dict[str, list[Part]] = {}
        self.heights: dict[str, int] = {}
        # need[t][u][y]: the row of upstream stream u that row y of stream t waits for.
        self.need: dict[str, dict[str, np.ndarray]] = {}
        # Tensors whose rows a memtile joins from several workers' channel
        # slices: already the output of a link, so a lagging consumer's rows
        # wait in buffering workers instead of a memtile stage.
        self.joined: set[str] = set()
        # Streams held through DDR: the shim's drain of the source, the fifo
        # filled back, and the rows' shape for the host buffer.
        self.spills: list[tuple[ObjectFifoHandle, ObjectFifo, Tensor]] = []
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
        # On chip an attention worker holds every value, so it also makes the
        # positional conv over them and adds it in, as PSA does next.
        self.tails: dict[str, tuple[Conv, Add]] = {}
        for i in sorted(mine):
            op = network.ops[i]
            if not on_chip or not isinstance(op, Attention):
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
                case InputQuant():
                    self.input_quant(op)
                case Conv():
                    self.conv(op)
                case Add():
                    self.add(op)
                case MaxPool():
                    self.maxpool(op)
                case Attention():
                    self.attention(op)
                case DetectPost():
                    self.detect(op)
                case Concat() | Slice() | Upsample():
                    pass
                case _:
                    raise NotImplementedError(f"{type(op).__name__} is not lowered yet")
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
        """The run built twice when on chip: the second time every tensor's chain
        of holds is made whole by its first consumer, a stage ending at each
        reader's rows, and every producer makes the fifos its readers take."""
        design = cls(network, params, ops, outputs, **options)
        if not options.get("on_chip"):
            return design
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
        """The runtime sequence's fifo handles: every filled fifo, every drained one, then each spill's drain and fill."""
        return (
            [fifo.prod() for fifo in self.in_fifos]
            + [self.out_fifos[t.name].cons() for t in self.outputs]
            + [h for drain, back, _ in self.spills for h in (drain, back.prod(depth=2))]
            + [drain for drain, _ in self.staged]
        )

    def transfer(self, handles: Sequence, locate: Callable[[Tensor], tuple]) -> None:
        """The run's host transfers, inside its runtime sequence.

        `handles` are `handles()`' as the sequence received them; `locate`
        gives each tensor's host buffer, its size in int32 words and the
        tensor's word offset in it; a spill's rows are the tensor it carries.
        Drains are issued first, then each spill's and each staged tensor's;
        then each fill step's transfers, every step but the last finishing
        before the next starts, and a step that reads a staged tensor
        starting once every staged drain is done; then each spill's fill,
        once its drain is done.
        """
        filled = dict(zip(self.in_fifos, handles))
        drains = TaskGroup()
        spilled = handles[
            len(self.in_fifos) + len(self.outputs) : len(handles) - len(self.staged)
        ]
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
        refills = []
        for (_, _, rows), drain, back in zip(self.spills, spilled[0::2], spilled[1::2]):
            buffer, total, offset = locate(rows)
            tap = TensorAccessPattern(
                (total,),
                offset,
                [1, 1, 1, rows.row_bytes * rows.height // 4],
                [0, 0, 0, 1],
            )
            group = TaskGroup()
            drain.drain(buffer, tap, wait=True, group=group)
            refills.append((group, back, buffer, tap))
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
        fills = None
        for step in steps:
            if staging is not None and any(
                f.tensor.name in staged for f in self.fills if f.step == step
            ):
                staging.finish()
                staging = None
            fills = TaskGroup()
            for fill in (f for f in self.fills if f.step == step):
                buffer, total, offset = locate(fill.tensor)
                filled[fill.fifo].fill(
                    buffer, fill.tap(total, offset), wait=step != steps[-1], group=fills
                )
            if step != steps[-1]:
                fills.finish()
        back_in = TaskGroup() if refills else None
        for group, back, buffer, tap in refills:
            group.finish()
            back.fill(buffer, tap, wait=True, group=back_in)
        drains.finish()
        if fills is not None:
            fills.finish()
        if back_in is not None:
            back_in.finish()

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
            # On chip, a fifo not read over shared memory goes by DMA even
            # between neighbours: each end then holds the objects its plan
            # counts, where one shared pool would sit in either tile.
            self.fifos[tensor.name] = ObjectFifo(
                row(size),
                name=f"t{len(self.fifos)}",
                transport=(
                    Transport.shared_mem()
                    if private and not shared
                    else Transport.dma() if self.on_chip else None
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
        `L1_LAG_BYTES` of its tile: in a memtile while they are few, since
        each row takes a buffer descriptor of the memtile's channel, and
        otherwise in the tile of a worker that only buffers them. On chip,
        they wait in memtile holds, the last of them in memtile `hub` where
        given, so one worker's held inputs can take turns on one channel. Such
        a worker's rows that do not wait come over shared memory, so its
        channels are left to the held rows and its weights.
        """
        if self.l1_rows(tensor, depth, window) == depth:
            if self.on_chip and hub is not None:
                return self.private(tensor).cons(depth=depth)
            return self.reader(tensor).cons(depth=depth)
        # A fifo joins one memtile link, so the consumers that stage share
        # the stage, and one needing more lag extends it.
        lag = depth - window
        stages = self.stages.setdefault(tensor.name, [])
        if self.on_chip and tensor.name not in self.joined:
            stage = self.held(tensor, lag, hub)
            # A stage another worker or the chain's next stage reads would
            # broadcast, which takes no turns: a worker with a hub takes its
            # own forward of it there unless the stage is in that hub.
            if hub is not None and (
                stage in self.read_stages or self.stage_hubs.get(stage) != hub
            ):
                stage = stage.cons(depth=2).forward(
                    tile=hub, depth=2, name=f"{stage.name}_{len(self.read_stages)}"
                )
            self.read_stages.add(stage)
            return stage.cons(depth=window)
        base = max(
            (st for st in stages if st[1] <= lag), key=lambda st: st[1], default=None
        )
        if base is None or base[1] < lag:
            source, held = (self.reader(tensor), 0) if base is None else base
            if depth >= tensor.height - 1:
                # The consumer waits for the whole map (its last row clamped,
                # all but one): hold it in DDR.
                staged = self.spill(
                    source if base is None else self.reader(tensor),
                    Tensor(f"{tensor.name}/spill", *tensor.shape),
                )
            elif (
                not stages
                and lag <= MEMTILE_LAG_ROWS
                and tensor.name not in self.joined
            ):
                staged = source.cons(depth=lag).forward(
                    depth=2, name=f"lag{sum(map(len, self.stages.values()))}"
                )
            else:
                staged = self.buffer(
                    source, tensor.row_bytes, tensor.height, lag - held
                )
            stages.append((staged, lag))
            base = stages[-1]
        staged = base[0]
        return staged.cons(depth=window)

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
        self,
        source: ObjectFifo,
        rows: int,
        row_bytes: int,
        obj_type: type[np.ndarray] | None = None,
        to_stream: TensorAccessPattern | None = None,
        hub: Tile | None = None,
    ) -> ObjectFifo:
        """`source`'s rows of `row_bytes` held in memtiles, `rows` of them at once.

        A chain of memtile forwards, each holding at most MEMTILE_HOLD_ROWS
        (a lock's count) and MEMTILE_HOLD_BYTES: a pool deeper than its
        memtile's descriptors allow is walked whole by one descriptor a side.
        The last sends `to_stream` of each row as an `obj_type`, where given,
        from memtile `hub` where given.
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
                obj_type=None if rows else obj_type,
                to_stream=None if rows else to_stream,
            )
            self.holds += 1
        self.stage_hubs[fifo] = hub
        return fifo

    def weights(self, op: Conv, packed: np.ndarray, chunk_bytes: int) -> ObjectFifo:
        """`op`'s packed weight chunks, each with its bias, sent from DDR once an output row.

        Read from the host's weights buffer rather than set up with the
        configuration: configuring the array costs what it loads.
        """
        fifo = ObjectFifo(row(chunk_bytes), name=f"weights{self.weight_streams}")
        self.weight_streams += 1
        tensor = Tensor(f"{op.name}/weights", 8, 1, packed.size // 8)
        self.loads.append((tensor, packed))
        self.fills.append(Fill(fifo, tensor, "repeat", chunk=op.dst.height))
        return fifo

    def spill(self, source: ObjectFifo, rows: Tensor) -> ObjectFifo:
        """`source`'s rows held in DDR: drained as they come, filled back once all have.

        For a consumer that waits for the whole map anyway, as one across an
        attention barrier does: no tile holds them, and the runtime sequence
        starts the fill once the drain is done. `rows` names the stream and
        gives its row size and height for the host buffer.
        """
        back = ObjectFifo(row(rows.row_bytes), name=f"spill{len(self.spills)}")
        self.spills.append((source.cons(depth=2), back, rows))
        return back

    def buffer(
        self, source: ObjectFifo, size: int, height: int, lag: int
    ) -> ObjectFifo:
        """Workers that hold `lag` rows of `source` in their tiles and pass each on that much later.

        A compute tile's DMA has 16 buffer descriptors, so the rows wait in a
        ring buffer of the worker's own rather than in a deep fifo: it reads
        rows until it holds `lag`, then passes the oldest on as each new one
        arrives. A lag beyond one tile is spread over a chain of them.
        """
        workers = -(-lag * size // BUFFER_TILE_BYTES)
        for i in range(workers):
            held = lag // workers + (i < lag % workers)
            out = ObjectFifo(row(size), name=f"held{len(self.workers)}")
            copy = self.copier(Part(Tensor("rows", 0, 0, 0), 0, size))

            def body(src, dst, copy, ring, held=held):
                def emit(r):
                    o = dst.acquire(1)
                    copy(view(ring, (r % held) * size, size), o)
                    dst.release(1)

                def take(r):
                    i = src.acquire(1)
                    copy(i, view(ring, (r % held) * size, size))
                    src.release(1)
                    if r >= held - 1:
                        emit(r - held + 1)

                # Row r's slot repeats every `held` rows, so the steady state
                # rolls a group of `held` rows with every slot known.
                steady = (height - held + 1) // held
                for r in range(held - 1):
                    take(r)
                for _ in range_(steady):
                    for r in range(held - 1, 2 * held - 1):
                        take(r)
                for r in range(held - 1 + steady * held, height):
                    take(r)
                for r in range(height - held + 1, height):
                    emit(r)

            self.workers.append(
                Worker(
                    body,
                    [
                        source.cons(depth=1),
                        out.prod(depth=1),
                        copy,
                        Buffer(row(held * size)),
                    ],
                )
            )
            source = out
        return source

    def parts(self, tensor: Tensor) -> list[Part]:
        """Where each row of `tensor` comes from, with views, concatenations and upsamples resolved."""
        if tensor.name in self.alias:
            return [self.alias[tensor.name]]
        if tensor.name in self.virtual:
            return self.virtual[tensor.name]
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

    def input_quant(self, op: InputQuant) -> None:
        k = self.input_quant_kernel(op)
        images = Tensor("images", 3, op.dst.height, op.dst.width)
        rows = np.arange(op.dst.height)
        depths = self.schedule(op.dst, [(images, rows, rows)])
        image = self.consumer(images, depths["images"], 2)

        def body(src, out, k):
            for _ in range_(op.dst.height):
                pixels = src.acquire(1)
                o = out.acquire(1)
                k(pixels, o)
                src.release(1)
                out.release(1)

        self.workers.append(Worker(body, [image, self.fifo(op.dst).prod(depth=2), k]))

    def assemble(self, tensor: Tensor) -> Tensor:
        """A worker that copies `tensor`'s parts into rows of a stream of their own."""
        assembled = Tensor(f"{tensor.name}/assembled", *tensor.shape)
        self.local.add(assembled.name)
        src = RowInput(self, tensor, into_output=True)
        # Double-buffered if that fits the tile, else one row at a time.
        for double in (True, False):
            depths = self.schedule(assembled, src.windows(tensor.height), double)
            held = sum(
                self.l1_rows(t, depths[t.name], 1) * t.row_bytes for t in src.sources
            )
            if held + (1 + double) * tensor.row_bytes <= L1_BYTES - L1_RESERVE:
                break
        src.connect(self, depths)

        def body(*args):
            args = list(args)
            inp = src.bind(args)
            (out,) = args
            for _ in range_(tensor.height // src.period):
                for phase in range(src.period):
                    o = out.acquire(1)
                    inp.acquire(phase, into=o)
                    inp.release(phase)
                    out.release(1)

        self.workers.append(
            Worker(body, [*src.fn_args, self.fifo(assembled).prod(depth=1 + double)])
        )
        return assembled

    def conv(self, op: Conv) -> None:
        p = self.params
        shift = int(p.of(op.name, "shift"))
        ic, oc = op.src.channels8, op.dst.channels8
        weights = pack_weights(op, p.of(op.name, "weight"))
        bias = np.zeros(oc, np.int32)
        bias[: op.dst.channels] = p.of(op.name, "bias")
        table = p.of(op.name, "table").astype(np.int8) if op.act else None
        src_tensor = op.src
        pieces: list[Part] = []
        while True:
            reader: RowInput | Part
            if op.kernel == 1:
                reader = RowInput(self, src_tensor)
                if len(reader.sources) > 2:
                    src_tensor = self.assemble(src_tensor)
                    continue
                windows = reader.windows(op.dst.height)
                local_bytes = src_tensor.row_bytes if reader.assembled else 0
            else:
                parts = self.parts(src_tensor)
                # A depthwise conv computes each part of one stream apart, into
                # its channels of the output row; others read one whole part.
                apart = (
                    op.groups > 1
                    and len({part.source for part in parts}) == 1
                    and not any(part.upsample for part in parts)
                )
                if not apart and (len(parts) > 1 or parts[0].upsample):
                    src_tensor = self.assemble(src_tensor)
                    continue
                pieces = parts
                reader = parts[0]
                y = np.arange(op.dst.height)
                lo = np.maximum(op.stride * y - 1, 0)
                windows = [(reader.source, lo, op.stride * y + 1)]
                local_bytes = 0
            # Double-buffered if the weights then fit, else one row at a time.
            for double in (True, False):
                depths = dict(self.schedule(op.dst, windows, double))
                window = 2 if op.kernel == 1 else 3
                if double and op.stride > 1:
                    # The worker holds its whole window while it computes, so
                    # the producer runs ahead only into rows past it.
                    depths[windows[0][0].name] += op.stride
                held = sum(
                    self.l1_rows(s, depths[s.name], window) * s.row_bytes
                    for s, _, _ in windows
                )
                free = (
                    L1_BYTES
                    - L1_RESERVE
                    - held
                    - local_bytes
                    - (1 + double) * op.dst.row_bytes
                    - 256 * op.act
                )
                chunk = oc
                while (
                    chunk
                    and (weights.nbytes + bias.nbytes)
                    * chunk
                    // oc
                    * (1 if chunk == oc else 2)
                    > free
                ):
                    chunk = next((c for c in range(chunk - 8, 0, -8) if oc % c == 0), 0)
                if chunk:
                    break
            if chunk == oc or len(windows) == 1:
                break
            # A streamed conv needs its second input channel for the weights.
            src_tensor = self.assemble(src_tensor)
        if chunk == 0:
            raise ValueError(
                f"{op.name}: no output-channel chunk of its weights fits beside {held} B of rows"
            )
        used = (weights.nbytes + bias.nbytes) * chunk // oc * (1 if chunk == oc else 2)
        copies = 2 if op.act and free - used >= TABLE_COPY_BYTES else 1

        if chunk == oc:
            per = 0
            constants = [
                Buffer(initial_value=weights),
                Buffer(initial_value=bias),
                *([Buffer(initial_value=table)] if op.act else []),
            ]
        else:
            n = oc // chunk
            per = weights.nbytes // n
            # Every chunk, each followed by its bias, sent a chunk at a time.
            packed = np.concatenate(
                [
                    np.concatenate(
                        [
                            weights[c * per : (c + 1) * per],
                            bias[c * chunk : (c + 1) * chunk].view(np.int8),
                        ]
                    )
                    for c in range(n)
                ]
            )
            wts = self.weights(op, packed, per + 4 * chunk)
            constants = [
                wts.cons(depth=2),
                *([Buffer(initial_value=table)] if op.act else []),
            ]
        out_part = op.dst.width * chunk

        def call(k, x, consts, o, *scalars):
            """One output row: the whole conv, or chunk by chunk with each chunk's weights acquired."""
            if chunk == oc:
                k(*x, *consts, o, *scalars)
                return
            wts, *rest = consts
            for c in range(oc // chunk):
                wc = wts.acquire(1)
                k(
                    *x,
                    view(wc, 0, per),
                    view(wc, per, chunk, np.int32),
                    *rest,
                    view(o, c * out_part, out_part),
                    *scalars,
                )
                wts.release(1)

        if isinstance(reader, RowInput):
            src = reader
            k = kernels.conv2dk1_i8(
                src_tensor.width,
                ic,
                chunk,
                bias=True,
                table=op.act,
                rounding="conv_even",
                table_copies=copies,
            )
            src.connect(self, depths)
            # After the inputs' handles: the order fifos are made in steers
            # placement, and this one measured faster.
            out = self.fifo(op.dst).prod(depth=1 + double)

            def body1(*args):
                args = list(args)
                inp = src.bind(args)
                out, k, *consts = args
                for _ in range_(op.dst.height // src.period):
                    for phase in range(src.period):
                        x = inp.acquire(phase)
                        o = out.acquire(1)
                        call(k, [x], consts, o, src_tensor.width, ic, chunk, shift)
                        inp.release(phase)
                        out.release(1)

            self.workers.append(Worker(body1, [*src.fn_args, out, k, *constants]))
            return

        part = reader
        if len(pieces) > 1:
            self.depthwise_apart(op, pieces, depths, double, shift, table, copies)
            return
        if op.groups == 1:
            k = kernels.conv2dk3(
                src_tensor.width,
                ic,
                chunk,
                bias=True,
                table=op.act,
                stride=op.stride,
                rounding="conv_even",
                table_copies=copies,
            )
        elif chunk == oc:
            k = kernels.bn_conv2dk3_dw(
                src_tensor.width,
                ic,
                oc,
                bias=True,
                table=op.act,
                rounding="conv_even",
                table_copies=copies,
            )
        else:
            raise NotImplementedError(f"{op.name}: a streamed depthwise conv")
        source = self.consumer(part.source, depths[part.source.name], 3)
        out = self.fifo(op.dst).prod(depth=1 + double)

        def body3(src, out, k, *consts):
            def step(rows, check):
                if not part.whole:
                    rows = [view(r, part.offset, part.size) for r in rows]
                o = out.acquire(1)
                call(
                    k,
                    rows,
                    consts,
                    o,
                    src_tensor.width,
                    ic,
                    chunk,
                    3,
                    3,
                    check,
                    shift,
                    0,
                )
                out.release(1)

            rows = src.acquire(2)
            step([rows[0], rows[0], rows[1]], 0)
            if op.stride == 1:
                for _ in range_(op.dst.height - 2):
                    rows = src.acquire(3)
                    step([rows[0], rows[1], rows[2]], 1)
                    src.release(1)
                rows = src.acquire(2)
                step([rows[0], rows[1], rows[1]], 2)
                src.release(2)
            else:
                src.release(1)
                for _ in range_(op.dst.height - 1):
                    rows = src.acquire(3)
                    step([rows[0], rows[1], rows[2]], 1)
                    src.release(2)
                src.release(1)

        self.workers.append(Worker(body3, [source, out, k, *constants]))

    def depthwise_apart(
        self,
        op: Conv,
        pieces: list[Part],
        depths: dict[str, int],
        double: bool,
        shift: int,
        table: np.ndarray | None,
        copies: int,
    ) -> None:
        """A depthwise 3x3 conv over parts of one stream, each part computed on its own channels."""
        p = self.params
        width = op.src.width
        sizes = {part.size for part in pieces}
        if len(sizes) > 1 or op.stride != 1:
            raise NotImplementedError(
                f"{op.name}: depthwise parts of unequal size, or strided"
            )
        channels = sizes.pop() // width
        weights = pack_weights(op, p.of(op.name, "weight"))
        bias = np.zeros(op.dst.channels8, np.int32)
        bias[: op.dst.channels] = p.of(op.name, "bias")
        per = 9 * channels
        consts = []
        for i in range(len(pieces)):
            consts += [
                Buffer(initial_value=weights[i * per : (i + 1) * per]),
                Buffer(initial_value=bias[i * channels : (i + 1) * channels]),
            ]
        if op.act:
            consts.append(Buffer(initial_value=table))
        k = kernels.bn_conv2dk3_dw(
            width,
            channels,
            channels,
            bias=True,
            table=op.act,
            rounding="conv_even",
            table_copies=copies,
        )
        source = self.consumer(pieces[0].source, depths[pieces[0].source.name], 3)
        out = self.fifo(op.dst).prod(depth=1 + double)
        size = channels * width

        def body(src, out, k, *consts):
            weights, rest = consts[: 2 * len(pieces)], consts[2 * len(pieces) :]

            def step(rows, check):
                o = out.acquire(1)
                for i, part in enumerate(pieces):
                    k(
                        *[view(r, part.offset, part.size) for r in rows],
                        weights[2 * i],
                        weights[2 * i + 1],
                        *rest,
                        view(o, i * size, size),
                        width,
                        channels,
                        channels,
                        3,
                        3,
                        check,
                        shift,
                        0,
                    )
                out.release(1)

            rows = src.acquire(2)
            step([rows[0], rows[0], rows[1]], 0)
            for _ in range_(op.dst.height - 2):
                rows = src.acquire(3)
                step([rows[0], rows[1], rows[2]], 1)
                src.release(1)
            rows = src.acquire(2)
            step([rows[0], rows[1], rows[1]], 2)
            src.release(2)

        self.workers.append(Worker(body, [source, out, k, *consts]))

    def fuse(self, group: Sequence[Op], later: Sequence[Op], drained: set[str]) -> None:
        """One worker computing a run of operators row by row, each at its own rate.

        Every tensor the run reads or makes for itself waits in a ring of
        rows in the worker's tile; a row from another worker is copied into
        its ring as it arrives. The run steps at the rate of its shortest
        tensor, so at each step a tensor of `rate` times that height takes
        `rate` rows: another worker's arrive, and an operator computes its
        own, in network order, `delay` steps after the run starts. Delays are
        the latest that keep every reader's rows in, so rings stay short.
        Each ring is a divisor of its rows in a period of steps, so the
        steady state rolls with every slot known; on chip, where every step
        runs in one loop, a ring is just the rows it holds at once.
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
        # Operators of one delay next to each other in the run's order: on
        # chip each such stretch computes under one guard, so a row read in
        # place must be read within one stretch of its own delay.
        stretches = [
            list(stretch)
            for _, stretch in itertools.groupby(
                steps, key=lambda step: delay[step[0].dst.name]
            )
        ]
        if self.on_chip:
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
        wanted = {name: rows for name, rows in need.items() if name not in direct}

        def sized(period: int) -> dict[str, int] | None:
            rings = {}
            for name, rows in wanted.items():
                slots = rate[name] * period
                ring = next((d for d in range(rows, slots + 1) if slots % d == 0), None)
                if ring is None:
                    return None
                rings[name] = ring
            return rings

        # Each step of a period is code of its own: the shortest period whose
        # rings take at most RING_SLACK_BYTES more than the least. On chip the
        # steps run as one loop that finds each slot at run time, so a ring
        # holds just the rows it must.
        options = {n: found for n in range(1, 17) if (found := sized(n)) is not None}
        bytes_at = {
            n: sum(rows * kept[name] for name, rows in rings.items())
            for n, rings in options.items()
        }
        least = min(bytes_at.values())
        period = min(
            n for n, used in bytes_at.items() if used <= least + RING_SLACK_BYTES
        )
        rings = dict(wanted) if self.on_chip else options[period]

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

        def oldest_read(name: str, s: int) -> int:
            """The oldest row of `name` a reader reads at step `s` or later."""
            lows = [
                window(part, reach, stride, height_in, max(y, 0))[0]
                for op, part, reach, stride, height_in in readers[name]
                for y in [rate[op.dst.name] * (s - delay[op.dst.name])]
                if y < op.dst.height
            ]
            return min(lows, default=tensors[name].height)

        def first_held(name: str, s: int) -> int:
            """The oldest row of `name` still held as step `s` starts."""
            return 0 if s <= 0 else min(oldest_read(name, s), last(name, s - 1) + 1)

        # On chip, a worker's several held inputs come from one memtile,
        # taking turns on one channel (its weights take the other).
        hub = AnyMemTile.copy() if self.on_chip and len(external) > 1 else None
        # Another worker's rows a run reads over several steps wait in the
        # fifo's own objects, a window sliding as the steps go, where that
        # takes less of the tile than a ring of the part read and the fifo.
        inplace: dict[str, tuple[int, int]] = {}
        for name in external:
            if name in direct or self.on_chip:
                continue
            row_bytes = stream_bytes_of(tensors[name])
            hold = need[name] + rate[name]
            depth = depths[name] + need[name]
            copied = (
                rings[name] * kept[name]
                + self.l1_rows(tensors[name], depths[name], 1) * row_bytes
            )
            if (
                self.l1_rows(tensors[name], depth, hold) == depth
                and depth * row_bytes < copied
            ):
                inplace[name] = (depth, hold)
                del rings[name]
        for name in external:
            source = tensors[name]
            if name in inplace:
                arg(("in", name), self.consumer(source, *inplace[name], hub))
                continue
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
        assembled = {
            name
            for name, parts in parts_of.items()
            if len(parts) > 1 or parts[0].upsample
        }
        # A 1x1 conv over a concat reads each part where it waits, 8 output
        # channels at a time (see conv2dk1_i8_bias_slices_ref): no row
        # assembled from the parts.
        sliced: dict[str, int] = {}
        for name in assembled:
            op = conv_of[name]
            width = math.gcd(
                *(part.size // part.source.width for part in parts_of[name])
            )
            if op.kernel == 1 and width % 8 == 0 and width < op.src.channels8:
                sliced[name] = width
        assembled -= sliced.keys()
        upsampled = {
            name: [part for part in parts_of[name] if part.upsample]
            for name in parts_of
        }
        per_channel = {
            op.name: (
                sliced[op.name] + 4
                if op.name in sliced
                else weights_of[op.name].nbytes // op.dst.channels8 + 4
            )
            for op in convs
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

        # The input kernel quantizes a whole row on its stack.
        stack = (
            IMAGE_ROW_BYTES if any(isinstance(op, InputQuant) for op in group) else 0
        )

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
                - stack
                - sum(rows * kept[name] for name, rows in rings.items())
                - sum(
                    self.l1_rows(tensors[name], *inplace.get(name, (depths[name], 1)))
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
                if level > 1 and op.name not in slice_of and (ic // level) % 8 == 0:
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

        # Outputs double-buffered where that fits, else one row at a time: a
        # row drains in a few thousand cycles of the tens of thousands it takes.
        # Then thinner slices of the streamed weights, each costing cycles.
        buffered, planned = next(
            (
                (b, got)
                for level in (1, 2, 4)
                for b in (2, 1)
                if (got := plan(b, level)) is not None
            ),
            (1, None),
        )
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

        # A step rolls when every row it takes and computes is inside its
        # tensor and away from every border.
        def interior(s: int) -> bool:
            for name in external:
                r = rate[name] * (s - delay[name])
                if r < 0 or r + rate[name] > tensors[name].height:
                    return False
            for op, inputs in steps:
                m = rate[op.dst.name]
                for j in range(m):
                    y = m * (s - delay[op.dst.name]) + j
                    if not 0 <= y < op.dst.height:
                        return False
                    for _, reach, stride, height_in in inputs:
                        if stride * y - reach < 0 or stride * y + reach >= height_in:
                            return False
            return True

        rolled = [s for s in range(total) if interior(s)]
        first = rolled[0] if rolled else total
        groups = 0
        while all(interior(first + groups * period + i) for i in range(period)):
            groups += 1

        streams_sliced = {name for name in sliced if name in streamed}
        # Every streamed conv's chunks come down one stream, each padded to the
        # largest, the same ones every step: a step drains the chunks of rows
        # it does not compute. On chip a memtile holds one step's chunks and
        # sends them over and over; otherwise DDR sends them once a step.
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
            if self.on_chip and sequence.size <= self.resident_bytes:
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
            # Per in-place input: the oldest row held, and the held objects.
            held = {}

            def slot(name, r, part=None):
                size = tensors[name].row_bytes
                offset, length = (0, size) if part is None else (part.offset, part.size)
                if name in held:
                    base, objects = held[name]
                    element = objects[r - base]
                    return (
                        element
                        if part is None or part.whole
                        else view(element, offset, length)
                    )
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

            def borders(center, height_in, y, height, stride, steady):
                """The rows above and below `center`, clamped at the map's edges, and a 3x3 kernel's border check for output row `y` of `height`.

                A step run as a loop knows its row only at run time, and works
                them out then.
                """
                if isinstance(y, int):
                    above = center - 1 if steady or center > 0 else center
                    below = center + 1 if steady or center + 1 < height_in else center
                    check = 1 if steady or 0 < y < height - 1 else (0 if y == 0 else 2)
                    if stride > 1 and not steady:
                        check = 0 if y == 0 else 1
                    return above, below, check
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

            def compute(op, inputs, y, steady, target):
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
                                center, height_in, y, op.dst.height, op.stride, steady
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
                        above, below, check = borders(
                            y, height_in, y, op.dst.height, 1, steady
                        )
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
                            center, height_in, y, op.dst.height, op.stride, steady
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
                                (
                                    r
                                    if steady
                                    else (
                                        min(max(r, 0), top)
                                        if isinstance(r, int)
                                        else minsi(
                                            maxsi(r, constant(0, index=True)),
                                            constant(top, index=True),
                                        )
                                    )
                                ),
                                part,
                            )
                            for r in [y + d for d in range(-2, 3)]
                        ]
                        k(*lines, target)

            def step(s, steady):
                arrived.clear()
                for name in inplace:
                    base = first_held(name, s)
                    count = last(name, s) - base + 1
                    if count > 0:
                        got = values[at[("in", name)]].acquire(count)
                        held[name] = (base, [got] if count == 1 else got)
                for name in external:
                    if name in inplace:
                        continue
                    m = rate[name]
                    for j in range(m):
                        r = m * (s - delay[name]) + j
                        if not (steady or 0 <= r < tensors[name].height):
                            continue
                        handle = values[at[("in", name)]]
                        x = handle.acquire(1)
                        if name in direct:
                            arrived[name] = x
                            continue
                        lo = span[name][0]
                        if kept[name] != tensors[name].row_bytes:
                            x = view(x, lo, kept[name])
                        values[at[("copy", kept[name], False)]](x, slot(name, r))
                        handle.release(1)
                for op, inputs in steps:
                    name = op.dst.name
                    m = rate[name]
                    for j in range(m):
                        y = m * (s - delay[name]) + j
                        if not (steady or 0 <= y < op.dst.height):
                            if op.name in pieces:
                                wts = values[at[("weights",)]]
                                for _ in range_(len(pieces[op.name])):
                                    wts.acquire(1)
                                    wts.release(1)
                            continue
                        handle = values[at[("out", name)]] if name in outs else None
                        o = handle.acquire(1) if handle is not None else None
                        target = slot(name, y) if name in rings else o
                        compute(op, inputs, y, steady, target)
                        if handle is not None:
                            copy = ("copy", tensors[name].row_bytes, False)
                            if name in rings:
                                values[at[copy]](slot(name, y), o)
                            for i in range(len(self.private_fifos[name])):
                                private = values[at[("private", name, i)]]
                                values[at[copy]](o, private.acquire(1))
                                private.release(1)
                            handle.release(1)
                for name in arrived:
                    values[at[("in", name)]].release(1)
                for name in inplace:
                    done = first_held(name, s + 1) - first_held(name, s)
                    if done > 0:
                        values[at[("in", name)]].release(done)

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
                            for j in range(m):
                                y = s * m + (j - m * d)
                                handle = (
                                    values[at[("out", name)]] if name in outs else None
                                )
                                o = handle.acquire(1) if handle is not None else None
                                target = slot(name, y) if name in rings else o
                                compute(op, inputs, y, False, target)
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

            # On chip the steps run as one loop: unrolled, the steps at either
            # end of a map take more program than a tile has.
            if self.on_chip:
                for s in range_(total):
                    stepped(s)
                return
            for s in range(first):
                step(s, False)
            if groups:
                for _ in range_(groups):
                    for i in range(period):
                        step(first + i, True)
            for s in range(first + groups * period, total):
                step(s, False)

        self.workers.append(Worker(body, args))

    def maxpool(self, op: MaxPool) -> None:
        """A 5x5 pool; a pool of a pool carries its input's rows along in front of its own.

        SPPF concatenates the input of a chain of three pools with all three
        outputs. Carrying each pool's input along in its output stream makes
        the three outputs one stream, so the concat reads two streams.
        """
        (part,) = self.parts(op.src)
        if part.upsample:
            raise NotImplementedError(f"{op.name}: a pool of an upsample")
        carry = isinstance(self.producer.get(op.src.name), MaxPool)
        if carry:
            carried = part.source
            stream = Tensor(
                f"{op.name}/carry",
                carried.channels + op.dst.channels,
                op.dst.height,
                op.dst.width,
            )
            for name, alias in list(self.alias.items()):
                if alias.source == carried:
                    self.alias[name] = Part(stream, alias.offset, alias.size)
            if op.src.name not in self.alias:
                self.alias[op.src.name] = Part(stream, 0, carried.row_bytes)
            self.alias[op.dst.name] = Part(stream, carried.row_bytes, op.dst.row_bytes)
        else:
            stream = op.dst
        self.local.add(stream.name)
        height = op.dst.height
        y = np.arange(height)
        lo, hi = np.maximum(y - 2, 0), np.minimum(y + 2, height - 1)
        depths = self.schedule(stream, [(part.source, lo, y + 2)])
        k = kernels.maxpool2d(op.dst.width, op.dst.channels8)
        copy = (
            [self.copier(Part(part.source, 0, part.source.row_bytes))] if carry else []
        )
        source = self.consumer(part.source, depths[part.source.name], 5)
        pooled_at = stream.row_bytes - op.dst.row_bytes

        def body(src, out, k, *copy):
            # The rows are few (20), so the window walks unrolled.
            held = 0
            for row in range(height):
                if row and lo[row] > lo[row - 1]:
                    src.release(int(lo[row] - lo[row - 1]))
                rows = src.acquire(int(hi[row] - lo[row] + 1))
                rows = rows if isinstance(rows, list) else [rows]
                lines = [
                    rows[min(max(r, 0), height - 1) - int(lo[row])]
                    for r in range(row - 2, row + 3)
                ]
                if not part.whole:
                    lines = [view(line, part.offset, part.size) for line in lines]
                o = out.acquire(1)
                k(*lines, view(o, pooled_at, op.dst.row_bytes) if carry else o)
                if carry:
                    copy[0](rows[row - int(lo[row])], view(o, 0, pooled_at))
                out.release(1)
                held = int(hi[row] - lo[row] + 1)
            src.release(held)

        self.workers.append(
            Worker(body, [source, self.fifo(stream).prod(depth=2), k, *copy])
        )

    def attention(self, op: Attention) -> None:
        """`attention_split` workers per head: each stores the map's keys and values, then answers its share of the rows of queries.

        A memtile splits each qkv row into every head's queries and its keys
        and values. A head answers its first query only once it has every
        key, so its queries wait in a tile of their own. Worker j of a head
        answers rows j, j + n, ... of its n; a memtile joins the workers' rows
        n at a time, and the head's consumers take them a row at a time.
        """
        p = self.params
        (part,) = self.parts(op.qkv)
        if not part.whole:
            raise NotImplementedError(f"{op.name}: attention over a view")
        qkv = part.source
        height, width = qkv.height, qkv.width
        kd, hd = op.key_dim, op.head_dim
        per_head = (2 * kd + hd) * width
        offsets, types = [], []
        for h in range(op.heads):
            offsets += [h * per_head, h * per_head + kd * width]
            types += [row(kd * width), row((kd + hd) * width)]
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
        if self.on_chip:
            # One worker answers every head. Whole qkv rows come in as they are
            # made, to store each head's keys and values; the map's queries
            # wait in memtiles, only their columns sent on, and come once
            # every key is in. The worker then makes the positional conv over
            # the values it stores and adds it in: its row is the add's.
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
                stored = [
                    s.acquire(1) for s in rest[5 + 2 * op.heads : 5 + 4 * op.heads]
                ]
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
                    pair = (
                        (attended, encoded) if attended_first else (encoded, attended)
                    )
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
                answer(
                    [(height - 2) * line, (height - 1) * line, (height - 1) * line], 2
                )

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
                        Buffer(
                            initial_value=p.of(op.name, "exp_table").astype(np.int32)
                        ),
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
            return
        pieces = (
            self.reader(qkv)
            .cons(depth=2)
            .split(
                offsets,
                obj_types=types,
                depths=[2] * len(types),
                names=[
                    f"{self.fifo(qkv).name}_head{i // 2}{'kv' if i % 2 else 'q'}"
                    for i in range(len(types))
                ],
            )
        )
        split = self.attention_split
        if height % split:
            raise NotImplementedError(f"{op.name}: rows not a multiple of {split}")
        # A memtile joins every worker's rows into `split` whole output rows,
        # the heads side by side; consumers take a row at a time.
        self.need[op.dst.name] = upstream
        self.heights[op.dst.name] = height
        self.heights[qkv.name] = height
        whole = op.dst.row_bytes
        self.fifos[op.dst.name] = ObjectFifo(
            row(split * whole),
            name=f"t{len(self.fifos)}",
            consumer_obj_type=row(whole),
        )
        self.joined.add(op.dst.name)
        name = self.fifo(op.dst).name
        # In offset order: a join's inputs fill its object in turn.
        rows = (
            self.fifo(op.dst)
            .prod(depth=2)
            .join(
                [j * whole + h * size for j in range(split) for h in range(op.heads)],
                obj_types=[row(size)] * (split * op.heads),
                depths=[2] * (split * op.heads),
                names=[
                    f"{name}_head{h}_rows{j}"
                    for j in range(split)
                    for h in range(op.heads)
                ],
            )
        )
        for h in range(op.heads):
            halves = rows[h :: op.heads]
            # Queries wait for every key, in DDR.
            queries = self.spill(
                pieces[2 * h], Tensor(f"{op.name}/head{h}/queries", kd, height, width)
            )
            for j, half in enumerate(halves):

                def body(queries, keys, out, k, load, table, kt, v, j=j):
                    # The rows are few (20), so the store walks unrolled.
                    for r in range(height):
                        kv = keys.acquire(1)
                        load(
                            view(kv, 0, kd * width),
                            view(kv, kd * width, hd * width),
                            kt,
                            v,
                            r,
                        )
                        keys.release(1)
                    for _ in range_(height // split):
                        for i in range(split):
                            q = queries.acquire(1)
                            if i == j:
                                o = out.acquire(1)
                                k(q, kt, v, table, o, q_mul, *shifts)
                                out.release(1)
                            queries.release(1)

                self.workers.append(
                    Worker(
                        body,
                        [
                            queries.cons(depth=2),
                            pieces[2 * h + 1].cons(depth=2),
                            half.prod(),
                            k,
                            k.load_kv,
                            Buffer(
                                initial_value=p.of(op.name, "exp_table").astype(
                                    np.int32
                                )
                            ),
                            Buffer(
                                np.ndarray[(height * width * kd,), np.dtype[np.int8]]
                            ),
                            Buffer(
                                np.ndarray[(height * width * hd,), np.dtype[np.int8]]
                            ),
                        ],
                    )
                )
        self.virtual[op.values.name] = [
            Part(qkv, h * per_head + 2 * kd * width, hd * width)
            for h in range(op.heads)
        ]

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

    def add(self, op: Add) -> None:
        p = self.params
        sa, sb, so = p.scale(op.a.name), p.scale(op.b.name), p.scale(op.dst.name)
        base = min(sa, sb, so)
        a_shift, b_shift, out_shift = sa - base, sb - base, so - base
        k = kernels.add(op.dst.row_bytes, dtype=np.int8)
        a, b = RowInput(self, op.a), RowInput(self, op.b)
        # A core has two input DMA channels.
        if len(a.sources) + len(b.sources) > 2:
            if len(a.sources) > 1:
                a = RowInput(self, self.assemble(op.a))
            if len(b.sources) > 1:
                b = RowInput(self, self.assemble(op.b))
        if a.period != 1 or b.period != 1:
            raise NotImplementedError(f"{op.name}: an add of an upsample")
        if set(a.sources) & set(b.sources):
            raise NotImplementedError(f"{op.name}: both operands read one stream")
        depths = self.schedule(
            op.dst, a.windows(op.dst.height) + b.windows(op.dst.height)
        )
        a.connect(self, depths)
        b.connect(self, depths)

        def body(*args):
            args = list(args)
            x, y = a.bind(args), b.bind(args)
            out, k = args
            for _ in range_(op.dst.height):
                u, v = x.acquire(0), y.acquire(0)
                o = out.acquire(1)
                k(u, v, o, a_shift, b_shift, out_shift)
                x.release(0)
                y.release(0)
                out.release(1)

        self.workers.append(
            Worker(body, [*a.fn_args, *b.fn_args, self.fifo(op.dst).prod(depth=2), k])
        )
