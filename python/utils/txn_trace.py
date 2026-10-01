# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Decode and semantically compare NPU TXN instruction streams.

A runtime sequence compiled once with `DispatchTime` scalars and a fully
static specialization of the same sequence program the same DMA transfers, but
their word streams are not byte-identical: the dynamic path draws buffer
descriptors from a free-list pool (different BD ids, a "BD free" poll before
each reuse), assembles BD words at build time, and leaves the buffer-address
word to the address patch. This module replays a stream into the register
state it programs and reduces it to a list of *events*, the things the DMA
hardware actually acts on:

* ``push``: a queue write on a channel, resolved to the BD it starts (transfer
  length, address as (host argument, byte offset) or an absolute address,
  the addressing dimensions in a normalized form, iteration, repeat count,
  whether a task-completion token is issued, and the channel control bits).
* ``wait``: a task-completion-token wait.
* ``pdi`` / ``preempt``: load-PDI and preemption markers, kept verbatim.

Two streams are equivalent when their event lists are equal. ``compare``
reports the first divergence; ``explain`` prints the events for a human.

The decoding covers the AIE2 / AIE2p shim, memtile and core DMA register
layouts used by npu1 and npu2 (register fields from aie-rt's
``xaiemlgbl_params.h``).
"""

from __future__ import annotations

import argparse
import dataclasses
from dataclasses import dataclass
from typing import Iterable, NamedTuple, Sequence

import numpy as np

# Opcodes from include/aie/Runtime/TxnEncoding.h.
OPC_WRITE = 0
OPC_BLOCKWRITE = 1
OPC_MASKWRITE = 3
OPC_MASKPOLL = 4
OPC_PREEMPT = 6
OPC_LOADPDI = 8
OPC_TCT = 1 << 7
OPC_DDR_PATCH = (1 << 7) + 1

# AIE2 tile addressing.
_COL_SHIFT = 25
_ROW_SHIFT = 20
_REG_MASK = (1 << _ROW_SHIFT) - 1

_BD_STRIDE = 0x20


class _DmaLayout(NamedTuple):
    bd_base: int
    bd_words: int
    ctrl_base: int
    mm2s_delta: int
    channels: int
    bd_id_mask: int


# Rows: 0 shim, 1 memtile, >=2 core. Identical on AIE2 and AIE2p.
_LAYOUT = {
    "shim": _DmaLayout(0x1D000, 8, 0x1D200, 0x10, 2, 0xF),
    "mem": _DmaLayout(0xA0000, 8, 0xA0600, 0x30, 6, 0x3F),
    "core": _DmaLayout(0x1D000, 6, 0x1DE00, 0x10, 2, 0xF),
}


def _bits(word: int, lsb: int, width: int) -> int:
    return (word >> lsb) & ((1 << width) - 1)


def _contiguous(
    dims: Sequence[tuple[int, int]], outer_stride: int, length: int
) -> bool:
    """Whether dims (innermost first) scan `length` words contiguously.

    Each stride must equal the product of the inner wraps; the outer stride
    only matters when the transfer runs past one pass over the dims.
    """
    extent = 1
    for wrap, stride in dims:
        if stride != extent:
            return False
        extent *= wrap
    return length <= extent or outer_stride == extent


def _tile_kind(row: int) -> str:
    return "shim" if row == 0 else ("mem" if row == 1 else "core")


@dataclass(frozen=True)
class Op:
    """One decoded TXN instruction."""

    kind: str
    pos: int
    words: tuple[int, ...]
    # Convenience fields (0 when not applicable).
    addr: int = 0
    value: int = 0
    mask: int = 0
    data: tuple[int, ...] = ()

    def __str__(self) -> str:
        if self.kind == "write32":
            return f"write32   {self.addr:#010x} <- {self.value:#010x}"
        if self.kind == "maskwrite32":
            return (
                f"maskwrite {self.addr:#010x} <- {self.value:#010x} & {self.mask:#010x}"
            )
        if self.kind == "maskpoll32":
            return (
                f"maskpoll  {self.addr:#010x} == {self.value:#010x} & {self.mask:#010x}"
            )
        if self.kind == "blockwrite":
            return (
                f"blockwrite {self.addr:#010x} <- ["
                + " ".join(f"{d:#x}" for d in self.data)
                + "]"
            )
        if self.kind == "patch":
            return (
                f"patch     {self.addr:#010x} <- arg{self.data[0]} + {self.data[1]:#x}"
            )
        if self.kind == "tct":
            return f"tct       {self.words[2]:#010x} {self.words[3]:#010x}"
        return f"{self.kind} {self.words}"


def decode(words: Sequence[int] | np.ndarray) -> list[Op]:
    """Split a TXN stream (header included) into instructions."""
    w = [int(x) & 0xFFFFFFFF for x in words]
    if len(w) < 4:
        raise ValueError("stream shorter than the 4-word TXN header")
    ops: list[Op] = [Op("header", 0, tuple(w[:4]))]
    pos = 4
    n = len(w)
    while pos < n:
        opc = w[pos]
        base = opc & 0xFF
        if base == OPC_WRITE:
            ops.append(
                Op(
                    "write32",
                    pos,
                    tuple(w[pos : pos + 6]),
                    addr=w[pos + 2],
                    value=w[pos + 4],
                )
            )
            pos += 6
        elif base == OPC_MASKWRITE:
            ops.append(
                Op(
                    "maskwrite32",
                    pos,
                    tuple(w[pos : pos + 7]),
                    addr=w[pos + 2],
                    value=w[pos + 4],
                    mask=w[pos + 5],
                )
            )
            pos += 7
        elif base == OPC_MASKPOLL:
            ops.append(
                Op(
                    "maskpoll32",
                    pos,
                    tuple(w[pos : pos + 7]),
                    addr=w[pos + 2],
                    value=w[pos + 4],
                    mask=w[pos + 5],
                )
            )
            pos += 7
        elif base == OPC_BLOCKWRITE:
            total = w[pos + 3] // 4
            if total < 4:
                raise ValueError(f"blockwrite at word {pos} shorter than its header")
            ops.append(
                Op(
                    "blockwrite",
                    pos,
                    tuple(w[pos : pos + total]),
                    addr=w[pos + 2],
                    data=tuple(w[pos + 4 : pos + total]),
                )
            )
            pos += total
        elif base == OPC_TCT:
            total = w[pos + 1] // 4
            ops.append(Op("tct", pos, tuple(w[pos : pos + total])))
            pos += total
        elif base == OPC_DDR_PATCH:
            total = w[pos + 1] // 4
            arg_plus = w[pos + 10] | (w[pos + 11] << 32)
            ops.append(
                Op(
                    "patch",
                    pos,
                    tuple(w[pos : pos + total]),
                    addr=w[pos + 6],
                    data=(w[pos + 8], arg_plus),
                )
            )
            pos += total
        elif base == OPC_LOADPDI:
            ops.append(Op("loadpdi", pos, tuple(w[pos : pos + 4])))
            pos += 4
        elif base == OPC_PREEMPT:
            ops.append(Op("preempt", pos, (opc,), value=opc >> 8))
            pos += 1
        else:
            raise ValueError(f"unknown TXN opcode {opc:#x} at word {pos}")
    return ops


@dataclass(frozen=True)
class Transfer:
    """A buffer descriptor as the DMA sees it.

    The form is independent of how the stream encoded it. Fields that pack
    several register bits (`packet`, `flags`) keep the tile's own encoding,
    so they compare between BDs of one tile kind.
    """

    length: int  # buffer_length register value, in 32-bit words
    address: tuple  # ("arg", idx, byte_offset) or ("abs", address)
    dims: tuple[
        tuple[int, int], ...
    ]  # (wrap, stride) innermost first; unit wraps dropped unless padded
    outer_stride: int  # stride applied when every dim wraps
    iteration: tuple[int, int]  # (wrap, stride)
    packet: int  # packet enable, type and id, and out-of-order BD id
    flags: int  # locks, valid, TLAST suppress, compression; no chaining
    burst_axcache: tuple[int, int]
    padding: tuple[tuple[int, int], ...] = ()  # memtile (before, after) per dim

    @staticmethod
    def from_words(kind: str, w: Sequence[int], patched: tuple | None) -> "Transfer":
        n_words = _LAYOUT[kind].bd_words
        if len(w) < n_words:
            raise ValueError(f"a {kind} BD has {n_words} words")
        padding = ()
        burst_axcache = (0, 0)
        if kind == "shim":
            length = w[0]
            address = w[1] | ((w[2] & 0xFFFF) << 32)
            dims = (
                (_bits(w[3], 20, 10), _bits(w[3], 0, 20) + 1),
                (_bits(w[4], 20, 10), _bits(w[4], 0, 20) + 1),
            )
            outer_stride = _bits(w[5], 0, 20) + 1
            iteration = (_bits(w[6], 20, 6), _bits(w[6], 0, 20) + 1)
            packet = _bits(w[2], 16, 15)
            flags = w[7] & ~(0xF << 27) & ~(1 << 26)
            burst_axcache = (w[4] >> 30, _bits(w[5], 24, 4))
        elif kind == "mem":
            length = _bits(w[0], 0, 17)
            address = _bits(w[1], 0, 19)
            dims = tuple(
                (_bits(x, 17, 10), _bits(x, 0, 17) + 1) for x in (w[2], w[3], w[4])
            )
            if dims[2][0]:
                outer_stride = _bits(w[5], 0, 17) + 1
            else:
                # A zero d2 wrap leaves d2 unbounded, as the shim's d2 is:
                # buffer_length ends the transfer and d3 never steps.
                dims, outer_stride = dims[:2], dims[2][1]
            iteration = (_bits(w[6], 17, 6), _bits(w[6], 0, 17) + 1)
            packet = _bits(w[0], 17, 15)
            flags = w[7] | (w[2] >> 31) << 32 | (w[4] >> 31) << 33
            pads = (
                (_bits(w[1], 26, 6), _bits(w[5], 17, 6)),
                (_bits(w[3], 27, 5), _bits(w[5], 23, 5)),
                (_bits(w[4], 27, 4), _bits(w[5], 28, 4)),
            )
            if any(b or a for b, a in pads):
                padding = pads
        else:
            length = _bits(w[0], 0, 14)
            address = _bits(w[0], 14, 14)
            dims = (
                (_bits(w[3], 13, 8), _bits(w[2], 0, 13) + 1),
                (_bits(w[3], 21, 8), _bits(w[2], 13, 13) + 1),
            )
            outer_stride = _bits(w[3], 0, 13) + 1
            iteration = (_bits(w[4], 13, 6), _bits(w[4], 0, 13) + 1)
            packet = _bits(w[1], 16, 15)
            flags = w[5] & ~(0xF << 27) & ~(1 << 26) | (w[1] >> 31) << 32
        if not padding:
            # A dimension with wrap 1 never applies its own stride (the
            # counter wraps after every element and the next dimension
            # steps), and wrap 0 means unused; both are absent from the
            # effective pattern. Linear mode (no dims) and [1, 1] wraps with
            # an outer stride of 1 move the same bytes.
            dims = tuple(d for d in dims if d[0] > 1)
            # A contiguous ND scan (innermost stride 1, every outer stride the
            # product of the inner wraps, the next block following on) moves
            # the same words as linear mode; the static emitter folds it.
            if dims and _contiguous(dims, outer_stride, length):
                dims, outer_stride = (), 1
        return Transfer(
            length=length,
            address=patched or ("abs", address),
            dims=dims,
            outer_stride=outer_stride,
            iteration=iteration,
            packet=packet,
            flags=flags,
            burst_axcache=burst_axcache,
            padding=padding,
        )

    def __str__(self) -> str:
        if self.address[0] == "arg":
            addr = f"arg{self.address[1]}+{self.address[2]:#x}"
        else:
            addr = f"{self.address[1]:#x}"
        dims = " ".join(f"{wrap}x{stride}" for wrap, stride in self.dims) or "linear"
        it = (
            f" iter {self.iteration[0]}x{self.iteration[1]}"
            if self.iteration[0]
            else ""
        )
        pad = (
            " pad=[" + " ".join(f"{b}/{a}" for b, a in self.padding) + "]"
            if self.padding
            else ""
        )
        return (
            f"len={self.length} @{addr} dims=[{dims}] outer={self.outer_stride}"
            f"{pad}{it}"
        )


@dataclass(frozen=True)
class Event:
    kind: str  # push | wait | loadpdi | preempt
    col: int = 0
    row: int = 0
    direction: str = ""  # S2MM | MM2S
    channel: int = 0
    bd: Transfer | None = None
    repeat: int = 0
    issue_token: bool = False
    ctrl: int = 0
    raw: tuple[int, ...] = ()

    def __str__(self) -> str:
        where = f"({self.col},{self.row}) {self.direction} ch{self.channel}"
        if self.kind == "push":
            tok = " token" if self.issue_token else ""
            rep = f" repeat={self.repeat}" if self.repeat else ""
            return f"push {where}{rep}{tok} ctrl={self.ctrl:#x}: {self.bd}"
        if self.kind == "wait":
            return f"wait {where} cols={self.raw[0]} rows={self.raw[1]}"
        return f"{self.kind} {self.raw}"


@dataclass
class _ChannelState:
    ctrl: int = 0


def trace(
    words: Sequence[int] | np.ndarray, *, ops: Iterable[Op] | None = None
) -> list[Event]:
    """Replay a stream and return the DMA events it triggers, in order."""
    ops = list(ops) if ops is not None else decode(words)
    regs: dict[int, int] = {}
    patches: dict[int, tuple] = {}  # BD address-word register -> ("arg", idx, plus)
    events: list[Event] = []

    def bd_words(
        col: int, row: int, kind: str, bd_id: int
    ) -> tuple[list[int], tuple | None]:
        layout = _LAYOUT[kind]
        tile = (col << _COL_SHIFT) | (row << _ROW_SHIFT)
        bd_base = tile | (layout.bd_base + bd_id * _BD_STRIDE)
        w = [regs.get(bd_base + 4 * i, 0) for i in range(layout.bd_words)]
        # Only shim BDs address host memory, through the word-1 patch.
        patched = patches.get(bd_base + 4) if kind == "shim" else None
        return w, patched

    def classify(addr: int) -> tuple[int, int, str, str, int] | None:
        """Classify a register address.

        Returns (col, row, kind, direction, channel) when addr is a DMA queue
        register, else None.
        """
        col = (addr >> _COL_SHIFT) & 0x7F
        row = (addr >> _ROW_SHIFT) & 0x1F
        reg = addr & _REG_MASK
        kind = _tile_kind(row)
        layout = _LAYOUT[kind]
        for direction, delta in (("S2MM", 0), ("MM2S", layout.mm2s_delta)):
            for ch in range(layout.channels):
                if reg == layout.ctrl_base + delta + ch * 8 + 4:
                    return col, row, kind, direction, ch
        return None

    for op in ops:
        if op.kind == "write32":
            regs[op.addr] = op.value
        elif op.kind == "maskwrite32":
            regs[op.addr] = (regs.get(op.addr, 0) & ~op.mask) | (op.value & op.mask)
        elif op.kind == "blockwrite":
            for i, d in enumerate(op.data):
                regs[op.addr + 4 * i] = d
            # A fresh BD image supersedes an earlier patch of its address word.
            for i in range(len(op.data)):
                patches.pop(op.addr + 4 * i, None)
        elif op.kind == "patch":
            patches[op.addr] = ("arg", op.data[0], op.data[1])
        elif op.kind == "tct":
            w2, w3 = op.words[2], op.words[3]
            direction = "S2MM" if (w2 & 0xFF) == 0 else "MM2S"
            # word 3: nrow [15:8], ncol [23:16], channel [31:24].
            events.append(
                Event(
                    "wait",
                    col=(w2 >> 16) & 0xFF,
                    row=(w2 >> 8) & 0xFF,
                    direction=direction,
                    channel=(w3 >> 24) & 0xFF,
                    raw=((w3 >> 16) & 0xFF, (w3 >> 8) & 0xFF),
                )
            )
        elif op.kind in ("loadpdi", "preempt"):
            events.append(Event(op.kind, raw=op.words))
        if op.kind != "write32":
            continue
        hit = classify(op.addr)
        if hit is None:
            continue
        col, row, kind, direction, ch = hit
        bd_id = op.value & _LAYOUT[kind].bd_id_mask
        repeat = (op.value >> 16) & 0xFF
        issue = bool(op.value >> 31)
        w, patched = bd_words(col, row, kind, bd_id)
        bd = Transfer.from_words(kind, w, patched)
        # A linear BD re-run repeat+1 times with an iteration dimension that
        # advances by its own length is one linear transfer that long; the
        # static emitter folds a contiguous repeat dimension into the length.
        iter_wrap, iter_stride = bd.iteration[0] + 1, bd.iteration[1]
        if (
            kind == "shim"
            and not bd.dims
            and bd.outer_stride == 1
            and repeat
            and iter_wrap == repeat + 1
            and iter_stride == bd.length
        ):
            bd = dataclasses.replace(bd, length=bd.length * iter_wrap, iteration=(0, 1))
            repeat = 0
        events.append(
            Event(
                "push",
                col=col,
                row=row,
                direction=direction,
                channel=ch,
                bd=bd,
                repeat=repeat,
                issue_token=issue,
                ctrl=regs.get(op.addr - 4, 0),
            )
        )
    return events


def compare(
    a: Sequence[int] | np.ndarray, b: Sequence[int] | np.ndarray, *, names=("a", "b")
) -> list[str]:
    """Return the differences between two streams' DMA events.

    Empty when the streams are equivalent.
    """
    ea, eb = trace(a), trace(b)
    out: list[str] = []
    for i, (x, y) in enumerate(zip(ea, eb)):
        if x != y:
            out.append(f"event {i} differs:\n  {names[0]}: {x}\n  {names[1]}: {y}")
            break
    if len(ea) != len(eb):
        out.append(f"{names[0]} has {len(ea)} events, {names[1]} has {len(eb)}")
    return out


def explain(words: Sequence[int] | np.ndarray, *, raw: bool = False) -> str:
    """Human-readable listing of a stream's DMA events (or raw ops)."""
    if raw:
        return "\n".join(str(op) for op in decode(words))
    return "\n".join(f"{i:4d} {e}" for i, e in enumerate(trace(words)))


def load(path) -> np.ndarray:
    """Read an insts.bin / .txn file as a uint32 array."""
    return np.fromfile(path, dtype=np.uint32)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Decode or compare NPU TXN instruction streams."
    )
    p.add_argument(
        "stream", nargs="+", help="insts.bin file(s); two files are compared"
    )
    p.add_argument(
        "--raw", action="store_true", help="list instructions instead of DMA events"
    )
    args = p.parse_args(argv)
    if len(args.stream) == 1:
        print(explain(load(args.stream[0]), raw=args.raw))
        return 0
    if len(args.stream) != 2:
        p.error("give one stream to decode or two to compare")
    diffs = compare(
        load(args.stream[0]), load(args.stream[1]), names=tuple(args.stream)
    )
    if not diffs:
        print("equivalent")
        return 0
    print("\n".join(diffs))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
