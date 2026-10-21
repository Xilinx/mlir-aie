# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Numpy oracle: simulate a specialized design's data movement with taplib gather/scatter.

Independent of the Lean semantics and of the symbolic exporter: it interprets the
runtime sequence of a design specialized to one shape, and follows the conventions
of test/python/taplib/stream_simulation.py (a DMA gathers an object onto the
stream, the next one scatters the stream into consecutive objects).
"""

import math

import numpy as np
from aie import ir
from aie.helpers.npdtypes import np_ndarray_type_get_shape
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import Program, Worker
from aie.iron.dataflow.objectfifo import ObjectFifo, ObjectFifoLink
from aie.iron.runtime.endpoint import RuntimeEndpoint

HOST, CORE = 1 << 61, 1 << 62

PYTHON_OPS = {
    "arith.addi": lambda a, b: a + b,
    "arith.subi": lambda a, b: a - b,
    "arith.muli": lambda a, b: a * b,
    "arith.floordivsi": lambda a, b: a // b,
    "arith.divsi": lambda a, b: (abs(a) // abs(b)) * (1 if (a < 0) == (b < 0) else -1),
    "arith.remsi": lambda a, b: a
    - b * ((abs(a) // abs(b)) * (1 if (a < 0) == (b < 0) else -1)),
    "arith.maxsi": max,
    "arith.minsi": min,
    "arith.andi": lambda a, b: a & b,
    "arith.ori": lambda a, b: a | b,
}
CMPI = [
    lambda a, b: a == b,
    lambda a, b: a != b,
    lambda a, b: a < b,
    lambda a, b: a <= b,
    lambda a, b: a > b,
    lambda a, b: a >= b,
]
CASTS = ("arith.extsi", "arith.extui", "arith.trunci", "arith.index_cast")


class Interpreter:
    """Runs a runtime sequence with Python integers, recording the shim transfers it issues."""

    def __init__(self, runtime):
        self.runtime = runtime
        self.ivs: list[tuple[ir.Value, int]] = []
        self.issued: list[tuple[str, int, TensorAccessPattern]] = []
        self.block(runtime.sequence_op.body.blocks[0])

    def value(self, x) -> int:
        if isinstance(x, (int, np.integer)):
            return int(x)
        v = ir.Value(x)
        for iv, i in reversed(self.ivs):
            if iv == v:
                return i
        op = v.owner.operation
        if op.name == "arith.constant":
            return ir.IntegerAttr(op.attributes["value"]).value
        if op.name in CASTS:
            return self.value(op.operands[0])
        if op.name == "arith.cmpi":
            predicate = ir.IntegerAttr(op.attributes["predicate"]).value
            return int(
                CMPI[predicate if predicate < 6 else predicate - 4](
                    *map(self.value, op.operands)
                )
            )
        return PYTHON_OPS[op.name](*map(self.value, op.operands))

    def tap(self, tap: TensorAccessPattern) -> TensorAccessPattern:
        return TensorAccessPattern(
            [self.value(d) for d in tap.tensor_dims],
            self.value(tap.offset),
            [self.value(d) for d in tap.sizes],
            [self.value(d) for d in tap.strides],
        )

    def block(self, block: ir.Block) -> None:
        for op in block.operations:
            op = op.operation
            if op.name == "aiex.dma_configure_task_for":
                t = next(t for t in self.runtime.transfers if t.task.operation == op)
                buf = ir.BlockArgument(ir.Value(t.rt_data.op)).arg_number
                self.issued.append((t.alloc, buf, self.tap(t.tap)))
            elif op.name == "scf.if":
                taken = 0 if self.value(op.operands[0]) else 1
                if len(op.regions[taken].blocks):
                    self.block(op.regions[taken].blocks[0])
            elif op.name == "scf.for":
                lo, hi, step = map(self.value, op.operands[:3])
                body = op.regions[0].blocks[0]
                for i in range(lo, hi, step):
                    self.ivs.append((ir.Value(body.arguments[0]), i))
                    self.block(body)
                    self.ivs.pop()
            elif op.name == "cf.assert":
                assert self.value(op.operands[0]), f"runtime guard fails: {op}"


class Simulation:
    """Element provenance for every stream of a specialized Program.

    Host element `idx` of runtime buffer `b` is `HOST | b << 48 | idx`; slot `s` of the
    `j`-th object worker `w` produces on fifo `f` is `CORE | w << 52 | f << 44 | j << 20 | s`.
    """

    def __init__(self, program: Program):
        self.runtime = program.entry
        self.workers = [w for c in program.configurations for w in c.workers]
        self.fifos: list[ObjectFifo] = []
        pending = [h.object_fifo for h in self.runtime.fifos]
        pending += [h.object_fifo for w in self.workers for h in w.fifos]
        while pending:
            of = pending.pop(0)
            if any(of is f for f in self.fifos):
                continue
            self.fifos.append(of)
            for h in [of.producer, *of.consumers]:
                if isinstance(h.endpoint, ObjectFifoLink):
                    pending += [
                        x.object_fifo for x in h.endpoint.srcs + h.endpoint.dsts
                    ]
        self.issued = Interpreter(self.runtime).issued

    def transfers(self, name: str):
        for alloc, buf, tap in self.issued:
            if alloc == name:
                yield buf, tap

    def host_stream(self, name: str) -> np.ndarray:
        parts = []
        for buf, tap in self.transfers(name):
            ids = (
                HOST
                | (buf << 48)
                | np.arange(math.prod(tap.tensor_dims), dtype=np.int64)
            )
            parts.append(tap.gather(ids).reshape(-1))
        return np.concatenate(parts) if parts else np.zeros(0, np.int64)

    def segments(self, link: ObjectFifoLink, shared: int) -> list[tuple[int, int]]:
        """The `(start, size)` pieces a join's or distribute's offsets cut a shared object into."""
        offsets = link.src_offsets if len(link.srcs) > 1 else link.dst_offsets
        if len(offsets) <= 1:
            return [(0, shared)]
        ends = [*offsets[1:], shared]
        return [(a, b - a) for a, b in zip(offsets, ends)]

    def consumer_objects(
        self, of: ObjectFifo, endpoint, count: int | None
    ) -> np.ndarray:
        """The objects `endpoint` receives on `of` after its scatter; all of them for None.

        A distribute's input scatters each destination's piece with the same walk.
        """
        cons = next(c for c in of.consumers if c.endpoint is endpoint)
        size = math.prod(np_ndarray_type_get_shape(of.consumer_obj_type))
        stream = self.stream(of, None if count is None else count * size)
        count = len(stream) // size if count is None else count
        assert len(stream) >= count * size, f"{of.name}: stream too short"
        objs = stream[: count * size].reshape(count, size)
        walk = cons.from_stream
        if walk is None:
            return objs
        pieces = [(0, size)]
        if isinstance(endpoint, ObjectFifoLink) and len(endpoint.dsts) > 1:
            pieces = self.segments(endpoint, size)
        out = np.full((count, size), -1, np.int64)
        for j in range(count):
            for d, (start, n) in enumerate(pieces):
                chunk = objs[j, d * walk.numel : (d + 1) * walk.numel]
                piece = out[j, start : start + n].reshape(walk.tensor_dims)
                out[j, start : start + n] = walk.scatter(chunk, out=piece).reshape(-1)
        return out

    def stream(self, of: ObjectFifo, need: int | None) -> np.ndarray:
        """The stream `of`'s producer emits: all of it for None, else at least `need` elements.

        A join's output walks each source's piece of the shared object with `to_stream`.
        """
        prod = of.producer.endpoint
        if isinstance(prod, RuntimeEndpoint):
            return self.host_stream(of.name)
        size = math.prod(of.shape)
        walk = of.to_stream
        pieces = [(0, size)]
        if isinstance(prod, ObjectFifoLink) and len(prod.srcs) > 1:
            pieces = self.segments(prod, size)
        per_object = size if walk is None else len(pieces) * walk.numel
        reps = of.repeat_count or 1
        count = None if need is None else -(-need // (per_object * reps))
        if isinstance(prod, Worker):
            assert count is not None, f"{of.name}: a core's stream is unbounded"
            w = next(i for i, x in enumerate(self.workers) if x is prod)
            f = next(i for i, x in enumerate(self.fifos) if x is of)
            j = np.arange(count, dtype=np.int64)[:, None]
            s = np.arange(size, dtype=np.int64)[None, :]
            objs = CORE | (w << 52) | (f << 44) | (j << 20) | s
        else:
            objs = self.link_objects(prod, of, count)
        out = []
        for obj in objs:
            if walk is None:
                emitted = obj
            else:
                emitted = np.concatenate(
                    [walk.gather(obj[a : a + n]).reshape(-1) for a, n in pieces]
                )
            out += [emitted] * reps
        return np.concatenate(out) if out else np.zeros(0, np.int64)

    def link_objects(
        self, link: ObjectFifoLink, dst: ObjectFifo, count: int | None
    ) -> np.ndarray:
        """`dst`'s share of the link's shared objects: the first `count`, or all for None."""
        if len(link.srcs) == 1:
            shared = self.consumer_objects(link.srcs[0].object_fifo, link, count)
            if len(link.dsts) == 1:
                return shared
            d = next(i for i, h in enumerate(link.dsts) if h.object_fifo is dst)
            start = link.dst_offsets[d]
            return shared[:, start : start + math.prod(dst.shape)]
        assert count is not None, "a join of core streams needs a demand"
        size = math.prod(dst.shape)
        shared = np.full((count, size), -1, np.int64)
        for h, start in zip(link.srcs, link.src_offsets):
            part = self.consumer_objects(h.object_fifo, link, count)
            shared[:, start : start + part.shape[1]] = part
        return shared

    def lines(self) -> list[str]:
        out = []
        for of in self.fifos:
            for c in of.consumers:
                if isinstance(c.endpoint, Worker):
                    w = next(i for i, x in enumerate(self.workers) if x is c.endpoint)
                    objs = self.consumer_objects(of, c.endpoint, None)
                    for j, obj in enumerate(objs):
                        out.append(
                            f"deliver {of.name} {w} {j} "
                            + " ".join(self.show(x) for x in obj)
                        )
                elif isinstance(c.endpoint, RuntimeEndpoint):
                    for buf, idx, x in self.drain_writes(of):
                        out.append(f"write {buf} {idx} {self.show(x)}")
        return out

    def drain_writes(self, of: ObjectFifo) -> list[tuple[int, int, int]]:
        """`(buf, idx, element)` for every host element the runtime's drains of `of` write, in order."""
        writes = []
        for buf, tap in self.transfers(of.name):
            idx = tap.gather(np.arange(math.prod(tap.tensor_dims))).reshape(-1)
            writes += [(buf, int(i)) for i in idx]
        stream = self.stream(of, len(writes))
        return [(buf, idx, int(x)) for (buf, idx), x in zip(writes, stream)]

    def decode(self, x: int) -> tuple:
        """`("host", buf, idx)`, `("core", worker, fifo name, obj, slot)` or `("none",)`."""
        x = int(x)
        if x & CORE:
            w, f = (x >> 52) & 0xFF, (x >> 44) & 0xFF
            return ("core", w, self.fifos[f].name, (x >> 20) & 0xFFFFFF, x & 0xFFFFF)
        if x & HOST:
            return ("host", (x >> 48) & 0xFF, x & ((1 << 48) - 1))
        return ("none",)

    def show(self, x: int) -> str:
        match self.decode(x):
            case ("core", w, f, j, s):
                return f"c{w}:{f}:{j}:{s}"
            case ("host", b, i):
                return f"h{b}:{i}"
        return "none"
