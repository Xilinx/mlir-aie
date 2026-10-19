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

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from aie.dialects import memref  # pyright: ignore[reportAttributeAccessIssue]
from aie.extras.dialects.arith import constant  # pyright: ignore[reportMissingImports]
from aie.helpers.util import np_ndarray_type_to_memref_type
from aie.iron import Buffer, ExternalFunction, ObjectFifo, Worker, kernels
from aie.iron.controlflow import range_
from aie.iron.dataflow.endpoint import ObjectFifoEndpoint
from aie.iron.device import AnyMemTile
from aie.utils.config import aie_kernels_dir

from gen_yolo_data import Params
from yolo_spec import (
    Add,
    Attention,
    Concat,
    Conv,
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
        in_fifos / out_fifos: The fifo of each host input and output tensor.
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
        self.workers: list[Worker] = []
        self.stages: dict[str, tuple[ObjectFifo, int]] = {}
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
                case Concat() | Slice() | Upsample():
                    pass
                case _:
                    raise NotImplementedError(f"{type(op).__name__} is not lowered yet")
        self.outputs = list(outputs)
        self.out_fifos = {t.name: self.fifo(t) for t in self.outputs}
        self.in_fifos = {t.name: self.fifos[t.name] for t in self.inputs}

    def fifo(self, tensor: Tensor) -> ObjectFifo:
        if tensor.name not in self.fifos:
            if tensor.name not in self.local:
                self.inputs.append(tensor)
            size = IMAGE_ROW_BYTES if tensor.name == "images" else tensor.row_bytes
            self.fifos[tensor.name] = ObjectFifo(row(size), name=f"t{len(self.fifos)}")
        return self.fifos[tensor.name]

    def schedule(
        self, out: Tensor, windows: list[tuple[Tensor, np.ndarray, np.ndarray]]
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
        paths stall. A wait on a virtual row is a wait on a finished stream,
        which nothing stalls. Returns the depth per input stream: that span
        and a row in flight, or the window (doubled when it is one row).
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
                    pulled = at(theirs[shared], other_hi)
                    live = pulled < self.heights[shared]
                    if live.any():
                        # This input itself must take the pulled row; a path
                        # from the shared stream need only make the rows
                        # before it, to make room for it.
                        side = "right" if shared == source.name else "left"
                        ahead = (
                            np.searchsorted(mine[shared], pulled[live], side=side) - 1
                        )
                        span = max(span, int((ahead - lo[live] + 1).max()) + 1)
            # A one-row window double-buffers; a wider one already overlaps.
            depths[source.name] = max(span + (window == 1), depths.get(source.name, 0))
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
        # A fifo joins one memtile link, so the consumers that stage share it.
        if tensor.name not in self.stages:
            source = self.fifo(tensor)
            if depth - window <= MEMTILE_LAG_ROWS:
                staged = source.cons(depth=depth - window).forward(
                    depth=2, name=f"lag{len(self.stages)}"
                )
            else:
                staged = self.buffer(
                    source, tensor.row_bytes, tensor.height, depth - window
                )
            self.stages[tensor.name] = (staged, depth - window)
        staged, held = self.stages[tensor.name]
        if depth - window > held:
            raise NotImplementedError(
                f"{tensor.name}: a later reader needs {depth - window} staged rows, not {held}"
            )
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
        src.connect(self, self.schedule(assembled, src.windows(tensor.height)))

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
            Worker(body, [*src.fn_args, self.fifo(assembled).prod(depth=2)])
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
            if op.kernel == 1:
                src = RowInput(self, src_tensor)
                if len(src.sources) > 2:
                    src_tensor = self.assemble(src_tensor)
                    continue
                windows = src.windows(op.dst.height)
                local_bytes = src_tensor.row_bytes if src.assembled else 0
            else:
                parts = self.parts(src_tensor)
                if len(parts) > 1 or parts[0].upsample:
                    src_tensor = self.assemble(src_tensor)
                    continue
                (part,) = parts
                y = np.arange(op.dst.height)
                lo = np.maximum(op.stride * y - 1, 0)
                windows = [(part.source, lo, op.stride * y + 1)]
                local_bytes = 0
            depths = dict(self.schedule(op.dst, windows))
            window = 2 if op.kernel == 1 else 3
            held = sum(
                self.l1_rows(s, depths[s.name], window) * s.row_bytes
                for s, _, _ in windows
            )
            free = (
                L1_BYTES
                - L1_RESERVE
                - held
                - local_bytes
                - 2 * op.dst.row_bytes
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
        out = self.fifo(op.dst).prod(depth=2)
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

        if op.kernel == 1:
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
                        k.load_kv,
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
