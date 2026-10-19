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
    Concat,
    Conv,
    InputQuant,
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
        self.handles = [design.fifo(s).cons(depth=depths[s.name]) for s in self.sources]

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

        Output row y holds rows `lo..hi` of each input stream. Where two
        inputs come from one upstream stream, as the direct path of a residual
        add and its pair of 3x3 convs do, the worker holds one input's rows
        from `lo` while the other path pulls the upstream stream on to the
        row its own row y waits for, and the first input's producer keeps up;
        the fifo holds all those rows, or the upstream stream's producer and
        so both paths stall. Returns the depth per input stream: that span
        and a row in flight, or the window (doubled when it is one row).
        """
        need: dict[str, np.ndarray] = {}
        for source, _, hi in windows:
            upstream = {source.name: np.arange(source.height)} | self.need.get(
                source.name, {}
            )
            for name, rows in upstream.items():
                need[name] = (
                    np.maximum(need[name], rows[hi]) if name in need else rows[hi]
                )
        self.need[out.name] = need
        depths: dict[str, int] = {}
        for source, lo, hi in windows:
            window = int((hi - lo + 1).max())
            span = window
            mine = {source.name: np.arange(source.height)} | self.need.get(
                source.name, {}
            )
            for other, _, other_hi in windows:
                if other is source:
                    continue
                theirs = {other.name: np.arange(other.height)} | self.need.get(
                    other.name, {}
                )
                for shared in mine.keys() & theirs.keys():
                    # The rows of this input made from the shared stream's rows
                    # that the other input's row y has pulled through.
                    pulled = theirs[shared][other_hi]
                    ahead = np.searchsorted(mine[shared], pulled, side="right") - 1
                    # Once the shared stream has made its last row nothing
                    # can block its producer, so only earlier rows count.
                    live = pulled < theirs[shared].max()
                    if live.any():
                        span = max(span, int((ahead - lo + 1)[live].max()) + 1)
            # A one-row window double-buffers; a wider one already overlaps.
            depths[source.name] = max(span + (window == 1), depths.get(source.name, 0))
        return depths

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
        image = self.fifo(images).cons(depth=depths["images"])

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
                windows = src.windows(op.dst.height)
                local_bytes = src_tensor.row_bytes if src.assembled else 0
            else:
                (part,) = self.parts(src_tensor)
                if part.upsample:
                    raise NotImplementedError(f"{op.name}: a 3x3 conv of an upsample")
                y = np.arange(op.dst.height)
                lo = np.maximum(op.stride * y - 1, 0)
                hi = np.minimum(op.stride * y + 1, src_tensor.height - 1)
                windows = [(part.source, lo, hi)]
                local_bytes = 0
            depths = dict(self.schedule(op.dst, windows))
            held = sum(depths[s.name] * s.row_bytes for s, _, _ in windows)
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
        source = self.fifo(part.source).cons(depth=depths[part.source.name])

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

    def add(self, op: Add) -> None:
        p = self.params
        sa, sb, so = p.scale(op.a.name), p.scale(op.b.name), p.scale(op.dst.name)
        base = min(sa, sb, so)
        a_shift, b_shift, out_shift = sa - base, sb - base, so - base
        k = kernels.add(op.dst.row_bytes, dtype=np.int8)
        a, b = RowInput(self, op.a), RowInput(self, op.b)
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
