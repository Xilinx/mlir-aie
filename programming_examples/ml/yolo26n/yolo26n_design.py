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

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from aie.dialects import memref  # pyright: ignore[reportAttributeAccessIssue]
from aie.dialects.arith import index_cast  # pyright: ignore[reportAttributeAccessIssue]
from aie.extras.dialects.arith import constant  # pyright: ignore[reportMissingImports]
from aie.extras.types import i32  # pyright: ignore[reportMissingImports]
from aie.helpers.taplib import TensorAccessPattern
from aie.helpers.util import np_ndarray_type_to_memref_type
from aie.iron import Buffer, ExternalFunction, ObjectFifo, TaskGroup, Worker, kernels
from aie.iron.controlflow import range_
from aie.iron.dataflow.endpoint import ObjectFifoEndpoint
from aie.iron.device import AnyMemTile
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
# The stack and the core's own sections, and the allocator's alignment slack.
L1_RESERVE = 3 * 1024
L1_LAG_BYTES = 12 * 1024
# A memtile channel has 24 buffer descriptors, one per staged row.
MEMTILE_LAG_ROWS = 8
# What one buffering worker's ring takes of its tile.
BUFFER_TILE_BYTES = 48 * 1024


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

    `rows` sends the tensor's rows as they are. `anchor_classes` sends each
    pixel's channels together (every 8-channel group of it in turn), and
    `anchor_boxes` each pixel's first 4 channels: the per-anchor records
    the Detect postprocess reads. A step's transfers finish before the next
    step's start.
    """

    fifo: ObjectFifo
    tensor: Tensor
    layout: str = "rows"
    step: int = 0

    def tap(self, total: int, offset: int) -> TensorAccessPattern:
        """The pattern over a host buffer of `total` int32 words with the tensor at word `offset`."""
        t = self.tensor
        if self.layout == "anchor_classes":
            groups = t.channels8 // 8
            return TensorAccessPattern(
                (total,),
                offset,
                [t.height, t.width, groups, 2],
                [groups * t.width * 2, 2, t.width * 2, 1],
            )
        if self.layout == "anchor_boxes":
            return TensorAccessPattern(
                (total,), offset, [1, 1, t.height * t.width, 1], [0, 0, 2, 1]
            )
        size = (IMAGE_ROW_BYTES if t.name == "images" else t.row_bytes) * t.height // 4
        return TensorAccessPattern((total,), offset, [1, 1, 1, size], [0, 0, 0, 1])


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


def view(element, offset: int, size: int, dtype: type = np.int8):
    """`size` elements of `dtype` in an int8 buffer (an acquired element or a Buffer) from byte `offset`."""
    source = element.op if isinstance(element, Buffer) else element
    kind = np.ndarray[(size,), np.dtype[dtype]]
    return memref.view(
        np_ndarray_type_to_memref_type(kind), source, constant(offset, index=True), []
    )


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
    ) -> None:
        self.params = params
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
        for op in ops:
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

    def handles(self) -> list:
        """The runtime sequence's fifo handles: every filled fifo, then every drained one."""
        return [fifo.prod() for fifo in self.in_fifos] + [
            self.out_fifos[t.name].cons() for t in self.outputs
        ]

    def transfer(self, handles: Sequence, locate: Callable[[Tensor], tuple]) -> None:
        """The run's host transfers, inside its runtime sequence.

        `handles` are `handles()`' as the sequence received them; `locate`
        gives each tensor's host buffer, its size in int32 words and the
        tensor's word offset in it. Drains are issued first; then each fill
        step's transfers, every step but the last finishing before the next
        starts.
        """
        filled = dict(zip(self.in_fifos, handles))
        drains = TaskGroup()
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
            fills = TaskGroup()
            for fill in (f for f in self.fills if f.step == step):
                buffer, total, offset = locate(fill.tensor)
                filled[fill.fifo].fill(
                    buffer, fill.tap(total, offset), wait=step != steps[-1], group=fills
                )
            if step != steps[-1]:
                fills.finish()
        drains.finish()
        if fills is not None:
            fills.finish()

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
            self.fifos[tensor.name] = ObjectFifo(row(size), name=f"t{len(self.fifos)}")
            if tensor.name not in self.local:
                self.inputs.append(tensor)
                self.fills.append(Fill(self.fifos[tensor.name], tensor))
        return self.fifos[tensor.name]

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

    def consumer(self, tensor: Tensor, depth: int, window: int):
        """A consumer handle on `tensor`'s rows, `depth` deep.

        Rows a worker only holds while another of its inputs catches up wait
        past the worker's own window once they would take more than
        `L1_LAG_BYTES` of its tile: in a memtile while they are few, since
        each row takes a buffer descriptor of the memtile's channel, and
        otherwise in the tile of a worker that only buffers them.
        """
        if self.l1_rows(tensor, depth, window) == depth:
            return self.fifo(tensor).cons(depth=depth)
        # A fifo joins one memtile link, so the consumers that stage share
        # the stage, and one needing more lag extends it.
        lag = depth - window
        stages = self.stages.setdefault(tensor.name, [])
        base = max(
            (st for st in stages if st[1] <= lag), key=lambda st: st[1], default=None
        )
        if base is None or base[1] < lag:
            source, held = (self.fifo(tensor), 0) if base is None else base
            if not stages and lag <= MEMTILE_LAG_ROWS:
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

                # The schedule is static and the rows are few, so it unrolls.
                for r in range(height):
                    i = src.acquire(1)
                    copy(i, view(ring, (r % held) * size, size))
                    src.release(1)
                    if r >= held - 1:
                        emit(r - held + 1)
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

    def input_quant(self, op: InputQuant) -> None:
        table = Buffer(initial_value=self.params.of(op.name, "table").astype(np.int8))
        k = self.function(
            "yolo_input_quant",
            "yolo_input.cc",
            [row(IMAGE_ROW_BYTES), row(256), row(op.dst.row_bytes)],
            [f"-DYOLO_INPUT_WIDTH={op.dst.width}"],
        )
        images = Tensor("images", 3, op.dst.height, op.dst.width)
        rows = np.arange(op.dst.height)
        depths = self.schedule(op.dst, [(images, rows, rows)])
        image = self.consumer(images, depths["images"], 2)

        def body(src, out, table, k):
            for _ in range_(op.dst.height):
                pixels = src.acquire(1)
                o = out.acquire(1)
                k(pixels, table, o)
                src.release(1)
                out.release(1)

        self.workers.append(
            Worker(body, [image, self.fifo(op.dst).prod(depth=2), table, k])
        )

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
        table = (
            [Buffer(initial_value=p.of(op.name, "table").astype(np.int8))]
            if op.act
            else []
        )
        src_tensor = op.src
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
                if len(parts) > 1 or parts[0].upsample:
                    src_tensor = self.assemble(src_tensor)
                    continue
                (reader,) = parts
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

        if chunk == oc:
            constants = [
                Buffer(initial_value=weights),
                Buffer(initial_value=bias),
                *table,
            ]
            per = 0
        else:
            n = oc // chunk
            per = weights.nbytes // n
            # One memtile buffer of every chunk, each followed by its bias, sent
            # a chunk at a time: one descriptor however many chunks there are.
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
            wts = ObjectFifo(
                np.ndarray[(packed.size,), np.dtype[np.int8]],
                depth=1,
                consumer_obj_type=np.ndarray[(per + 4 * chunk,), np.dtype[np.int8]],
                init_values=[packed],
            )
            wts.prod().endpoint = ObjectFifoEndpoint(AnyMemTile.copy())
            constants = [wts.cons(depth=2), *table]
        out = self.fifo(op.dst).prod(depth=1 + double)
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
            )
            src.connect(self, depths)

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
        if op.groups == 1:
            k = kernels.conv2dk3(
                src_tensor.width,
                ic,
                chunk,
                bias=True,
                table=op.act,
                stride=op.stride,
                rounding="conv_even",
            )
        elif chunk == oc:
            k = kernels.bn_conv2dk3_dw(
                src_tensor.width, ic, oc, bias=True, table=op.act, rounding="conv_even"
            )
        else:
            raise NotImplementedError(f"{op.name}: a streamed depthwise conv")
        source = self.consumer(part.source, depths[part.source.name], 3)

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
        """One worker per head: it stores the map's keys and values, then answers its queries.

        A memtile splits each qkv row into every head's queries and its keys
        and values. A head answers its first query only once it has every
        key, so its queries wait in a tile of their own.
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
        pieces = (
            self.fifo(qkv)
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
        heads = []
        for h in range(op.heads):
            out = Tensor(f"{op.name}/head{h}", hd, height, width)
            self.local.add(out.name)
            self.need[out.name] = upstream
            self.heights[out.name] = height
            self.heights[qkv.name] = height
            heads.append(out)

            def body(queries, keys, out, k, load, table, kt, v):
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
                for _ in range_(height):
                    q = queries.acquire(1)
                    o = out.acquire(1)
                    k(q, kt, v, table, o, q_mul, *shifts)
                    queries.release(1)
                    out.release(1)

            self.workers.append(
                Worker(
                    body,
                    [
                        # Queries wait for every key in a tile of their own.
                        self.buffer(pieces[2 * h], kd * width, height, height).cons(
                            depth=2
                        ),
                        pieces[2 * h + 1].cons(depth=2),
                        self.fifo(out).prod(depth=2),
                        k,
                        k.load_kv,  # pyright: ignore[reportAttributeAccessIssue]
                        Buffer(
                            initial_value=p.of(op.name, "exp_table").astype(np.int32)
                        ),
                        Buffer(np.ndarray[(height * width * kd,), np.dtype[np.int8]]),
                        Buffer(np.ndarray[(height * width * hd,), np.dtype[np.int8]]),
                    ],
                )
            )
        self.virtual[op.dst.name] = [Part(t, 0, t.row_bytes) for t in heads]
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
        boxes = ObjectFifo(row(chunk * 4), name="detect_boxes")
        for step in (0, 1):
            for tensor in op.scores:
                self.fills.append(Fill(records, tensor, "anchor_classes", step))
        for tensor in op.boxes:
            self.fills.append(Fill(boxes, tensor, "anchor_boxes", 1))
        self.inputs += [*op.scores, *op.boxes]
        self.fifos[op.dst.name] = ObjectFifo(
            np.ndarray[(6 * op.dst.height,), np.dtype[np.int32]], name="detections"
        )
        flags = [f"-DYOLO_DETECT_CHUNK={chunk}"]
        k_max = self.function(
            "yolo_detect_max",
            "yolo_detect.cc",
            [
                row(chunk * 80),
                np.ndarray[(768,), np.dtype[np.int16]],
                row(anchors),
                np.int32,
            ],
            flags,
        )
        bind = k_max.object_file.bind
        ranks_t = np.ndarray[(768,), np.dtype[np.int16]]
        words_t = np.ndarray[(768,), np.dtype[np.int32]]
        keep_t = np.ndarray[(300,), np.dtype[np.int16]]
        count_t = np.ndarray[(4,), np.dtype[np.int32]]
        picked_t = np.ndarray[(anchors // 8,), np.dtype[np.uint8]]
        k_select = bind(
            "yolo_detect_select",
            [row(anchors), ranks_t, words_t, keep_t, picked_t, keep_t, count_t],
        )
        k_gather = bind(
            "yolo_detect_gather",
            [
                row(chunk * 80),
                row(chunk * 4),
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
        rank_table = np.zeros(768, np.int16)
        rank_table[: rank.size] = rank.reshape(-1)
        n = anchors // chunk

        def body(cls, box, out, k_max, k_select, k_gather, k_finish, *b):
            top, ranks, hist, first, picked, rows, count, cand, dist, sigmoid, best = b
            for i in range_(n):
                r = cls.acquire(1)
                k_max(r, ranks, top, index_cast(i32(), i))
                cls.release(1)
            k_select(top, ranks, hist, first, picked, rows, count)
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
            Buffer(row(anchors)),
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
