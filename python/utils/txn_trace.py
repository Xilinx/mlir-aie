# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Decode and semantically compare NPU TXN instruction streams.

A runtime sequence compiled once with `DispatchTime` scalars and a fully
static specialization of the same sequence program the same DMA transfers, but
their word streams are not byte-identical: the dynamic path draws buffer
descriptors from a free-list pool (different BD ids), polls for room in a
channel's task queue, assembles BD words at build time, and leaves the
buffer-address word to the address patch. This module replays a stream into
the register state it programs and reduces it to a list of *events*, the
things the hardware actually acts on:

* ``push``: a queue write on a channel, resolved to the chain of BDs it runs
  (for each, transfer length, address as (host argument, byte offset) or an
  absolute address, the addressing dimensions in a normalized form, and
  iteration), the repeat count, whether a task-completion token is issued,
  and the channel control bits.
* ``wait``: a task-completion-token wait.
* ``write``: a register write outside the DMA BD and channel registers
  (runtime parameters, locks, anything else the sequence programs).
* ``loadpdi`` / ``preempt`` / ``scratchpad`` / ``update_reg``: kept verbatim.

Polls only wait for state another op produces, so they are not events. A push
of a BD the stream never wrote is an error rather than a transfer of zeros:
its contents come from somewhere the stream does not show.

Two streams are equivalent when their headers name the same device and their
event lists are equal. ``compare`` reports the first divergence; ``explain``
prints the events for a human.

The decoding covers the AIE2 / AIE2p shim, memtile and core DMA register
layouts used by npu1 and npu2 (register fields from aie-rt's
``xaiemlgbl_params.h``).
"""

from __future__ import annotations

import argparse
import ctypes
import dataclasses
from dataclasses import dataclass
from enum import IntEnum
from typing import NamedTuple, Sequence

import numpy as np


class Opcode(IntEnum):
    """TXN opcodes, from include/aie/Runtime/TxnEncoding.h."""

    WRITE = 0
    BLOCKWRITE = 1
    MASKWRITE = 3
    MASKPOLL = 4
    PREEMPT = 6
    LOADPDI = 8
    CREATE_SCRATCHPAD = 10
    UPDATE_REG = 12
    TCT = 1 << 7
    DDR_PATCH = (1 << 7) + 1


@dataclass(frozen=True)
class Header:
    """The 4-word TXN header."""

    major: int
    minor: int
    dev_gen: int  # 3: npu1 (Phoenix, Hawk Point), 4: npu2 (Strix, Krackan)
    num_rows: int
    num_cols: int
    num_mem_tile_rows: int
    num_ops: int
    size: int  # bytes, header included

    @staticmethod
    def of(words: Sequence[int]) -> "Header":
        w0, w1, num_ops, size = words[:4]
        return Header(
            major=w0 & 0xFF,
            minor=(w0 >> 8) & 0xFF,
            dev_gen=(w0 >> 16) & 0xFF,
            num_rows=w0 >> 24,
            num_cols=w1 & 0xFF,
            num_mem_tile_rows=(w1 >> 8) & 0xFF,
            num_ops=num_ops,
            size=size,
        )

    @property
    def device(self) -> tuple[int, ...]:
        return (
            self.major,
            self.minor,
            self.dev_gen,
            self.num_rows,
            self.num_cols,
            self.num_mem_tile_rows,
        )

    def tile_kind(self, row: int) -> str:
        if row == 0:
            return "shim"
        return "mem" if row <= self.num_mem_tile_rows else "core"


# AIE2 tile addressing.
_COL_SHIFT = 25
_ROW_SHIFT = 20
_REG_MASK = (1 << _ROW_SHIFT) - 1

_BD_STRIDE = 0x20


class _Registers(ctypes.LittleEndianStructure):
    """Consecutive 32-bit registers decoded into named bit fields."""

    @classmethod
    def of(cls, words: Sequence[int]):
        n = ctypes.sizeof(cls) // 4
        if len(words) < n:
            raise ValueError(f"{cls.__name__} has {n} words")
        return cls.from_buffer_copy(np.asarray(words[:n], dtype="<u4").tobytes())


def _words(*words: list[tuple[str | None, int]]) -> list:
    """`_fields_` for registers given as (name, bits) lists, each LSB first.

    A None name is reserved. Every word must add up to 32 bits, so no field
    can straddle two registers.
    """
    fields = []
    for i, word in enumerate(words):
        if sum(bits for _, bits in word) != 32:
            raise ValueError(f"register word {i} does not add up to 32 bits")
        fields += [
            (name or f"_reserved{i}_{j}", ctypes.c_uint32, bits)
            for j, (name, bits) in enumerate(word)
        ]
    return fields


_PACKET = [
    ("packet_type", 3),
    ("packet_id", 5),
    ("out_of_order_bd_id", 6),
    ("enable_packet", 1),
]
_LOCKS = [
    ("lock_acq_id", 4),
    (None, 1),
    ("lock_acq_value", 7),
    ("lock_acq_enable", 1),
    ("lock_rel_id", 4),
    (None, 1),
    ("lock_rel_value", 7),
]
_CHAIN_FLAGS = [
    ("valid_bd", 1),
    ("use_next_bd", 1),
    ("next_bd", 4),
    ("tlast_suppress", 1),
]


class _ShimBd(_Registers):
    _fields_ = _words(
        [("buffer_length", 32)],
        [(None, 2), ("base_address_low", 30)],
        [("base_address_high", 16), *_PACKET, (None, 1)],
        [("d0_stepsize", 20), ("d0_wrap", 10), ("secure_access", 1), (None, 1)],
        [("d1_stepsize", 20), ("d1_wrap", 10), ("burst_length", 2)],
        [("d2_stepsize", 20), ("axqos", 4), ("axcache", 4), ("smid", 4)],
        [("iteration_stepsize", 20), ("iteration_wrap", 6), ("iteration_current", 6)],
        [*_LOCKS, *_CHAIN_FLAGS],
    )


class _MemBd(_Registers):
    _fields_ = _words(
        [
            ("buffer_length", 17),
            ("out_of_order_bd_id", 6),
            ("packet_id", 5),
            ("packet_type", 3),
            ("enable_packet", 1),
        ],
        [
            ("base_address", 19),
            ("use_next_bd", 1),
            ("next_bd", 6),
            ("d0_zero_before", 6),
        ],
        [("d0_stepsize", 17), ("d0_wrap", 10), (None, 4), ("tlast_suppress", 1)],
        [("d1_stepsize", 17), ("d1_wrap", 10), ("d1_zero_before", 5)],
        [
            ("d2_stepsize", 17),
            ("d2_wrap", 10),
            ("d2_zero_before", 4),
            ("enable_compression", 1),
        ],
        [
            ("d3_stepsize", 17),
            ("d0_zero_after", 6),
            ("d1_zero_after", 5),
            ("d2_zero_after", 4),
        ],
        [
            ("iteration_stepsize", 17),
            ("iteration_wrap", 6),
            ("iteration_current", 6),
            (None, 3),
        ],
        [
            ("lock_acq_id", 8),
            ("lock_acq_value", 7),
            ("lock_acq_enable", 1),
            ("lock_rel_id", 8),
            ("lock_rel_value", 7),
            ("valid_bd", 1),
        ],
    )


class _CoreBd(_Registers):
    _fields_ = _words(
        [("buffer_length", 14), ("base_address", 14), (None, 4)],
        [(None, 16), *_PACKET, ("enable_compression", 1)],
        [("d0_stepsize", 13), ("d1_stepsize", 13), (None, 6)],
        [("d2_stepsize", 13), ("d0_wrap", 8), ("d1_wrap", 8), (None, 3)],
        [
            ("iteration_stepsize", 13),
            ("iteration_wrap", 6),
            ("iteration_current", 6),
            (None, 7),
        ],
        [*_LOCKS, *_CHAIN_FLAGS],
    )


def _queue_push(bd_id_bits: int) -> type[_Registers]:
    class _QueuePush(_Registers):
        _fields_ = _words(
            [
                ("start_bd_id", bd_id_bits),
                (None, 16 - bd_id_bits),
                ("repeat_count", 8),
                (None, 7),
                ("enable_token_issue", 1),
            ]
        )

    return _QueuePush


class _DmaLayout(NamedTuple):
    bd_base: int
    bd: type[_Registers]
    num_bds: int
    ctrl_base: int
    mm2s_delta: int
    channels: int
    queue_push: type[_Registers]

    def owns(self, reg: int) -> bool:
        """Whether reg is one of this DMA's BD or channel registers."""
        bds_end = self.bd_base + self.num_bds * _BD_STRIDE
        ctrl_end = self.ctrl_base + self.mm2s_delta + self.channels * 8
        return self.bd_base <= reg < bds_end or self.ctrl_base <= reg < ctrl_end


# Identical on AIE2 and AIE2p.
_LAYOUT = {
    "shim": _DmaLayout(0x1D000, _ShimBd, 16, 0x1D200, 0x10, 2, _queue_push(4)),
    "mem": _DmaLayout(0xA0000, _MemBd, 48, 0xA0600, 0x30, 6, _queue_push(6)),
    "core": _DmaLayout(0x1D000, _CoreBd, 16, 0x1DE00, 0x10, 2, _queue_push(4)),
}

# The BD fields `Transfer.packet` and `Transfer.flags` compare. A shim BD has
# no compression bit, which reads as compression off.
_PACKET_FIELDS = tuple(name for name, _ in _PACKET)
_FLAG_FIELDS = (
    *(name for name, _ in _LOCKS if name),
    "valid_bd",
    "tlast_suppress",
    "enable_compression",
)


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
        if self.kind == "update_reg":
            return (
                f"update_reg {self.addr:#010x} <- table[{self.data[0]}] "
                f"func {self.data[1]} arg {self.value:#x}"
            )
        return f"{self.kind} {self.words}"


def decode(words: Sequence[int] | np.ndarray) -> list[Op]:
    """Split a TXN stream (header included) into instructions.

    The header's size and op count must match the stream.
    """
    w = [int(x) & 0xFFFFFFFF for x in words]
    if len(w) < 4:
        raise ValueError("stream shorter than the 4-word TXN header")
    header = Header.of(w)
    if header.size != 4 * len(w):
        raise ValueError(
            f"TXN header gives {header.size} bytes, the stream has {4 * len(w)}"
        )
    ops: list[Op] = [Op("header", 0, tuple(w[:4]))]
    pos = 4
    n = len(w)
    while pos < n:
        opc = w[pos]
        base = opc & 0xFF
        if base == Opcode.WRITE:
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
        elif base in (Opcode.MASKWRITE, Opcode.MASKPOLL):
            ops.append(
                Op(
                    "maskwrite32" if base == Opcode.MASKWRITE else "maskpoll32",
                    pos,
                    tuple(w[pos : pos + 7]),
                    addr=w[pos + 2],
                    value=w[pos + 4],
                    mask=w[pos + 5],
                )
            )
            pos += 7
        elif base == Opcode.BLOCKWRITE:
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
        elif base == Opcode.TCT:
            total = w[pos + 1] // 4
            ops.append(Op("tct", pos, tuple(w[pos : pos + total])))
            pos += total
        elif base == Opcode.DDR_PATCH:
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
        elif base == Opcode.LOADPDI:
            ops.append(Op("loadpdi", pos, tuple(w[pos : pos + 4])))
            pos += 4
        elif base == Opcode.PREEMPT:
            ops.append(Op("preempt", pos, (opc,), value=opc >> 8))
            pos += 1
        elif base == Opcode.CREATE_SCRATCHPAD:
            # Words 2-3 are the DDR address the runtime patches in.
            ops.append(
                Op(
                    "scratchpad",
                    pos,
                    tuple(w[pos : pos + 4]),
                    value=w[pos + 1],
                    data=((opc >> 8) & 0xFF,),
                )
            )
            pos += 4
        elif base == Opcode.UPDATE_REG:
            ops.append(
                Op(
                    "update_reg",
                    pos,
                    tuple(w[pos : pos + 3]),
                    addr=w[pos + 2],
                    value=w[pos + 1],
                    data=((opc >> 8) & 0xFF, (opc >> 16) & 0xFF),
                )
            )
            pos += 3
        else:
            raise ValueError(f"unknown TXN opcode {opc:#x} at word {pos}")
    if pos != n:
        raise ValueError(f"the op at word {ops[-1].pos} runs past the stream")
    if header.num_ops != len(ops) - 1:
        raise ValueError(
            f"TXN header counts {header.num_ops} ops, the stream has {len(ops) - 1}"
        )
    return ops


@dataclass(frozen=True)
class Transfer:
    """A buffer descriptor as the DMA sees it.

    The form is independent of how the stream encoded it.
    """

    length: int  # buffer_length register value, in 32-bit words
    address: tuple  # ("arg", idx, byte_offset) or ("abs", address)
    dims: tuple[
        tuple[int, int], ...
    ]  # (wrap, stride) innermost first; unit wraps dropped unless padded
    outer_stride: int  # stride applied when every dim wraps
    iteration: tuple[int, int]  # (wrap, stride)
    packet: tuple[int, ...]  # type, id, out-of-order BD id, enable
    flags: tuple[int, ...]  # locks, valid, TLAST suppress, compression; no chaining
    burst_axcache: tuple[int, int]
    padding: tuple[tuple[int, int], ...] = ()  # memtile (before, after) per dim

    @staticmethod
    def from_words(kind: str, w: Sequence[int], patched: tuple | None) -> "Transfer":
        bd = _LAYOUT[kind].bd.of(w)
        dims = ((bd.d0_wrap, bd.d0_stepsize + 1), (bd.d1_wrap, bd.d1_stepsize + 1))
        outer_stride = bd.d2_stepsize + 1
        padding = ()
        burst_axcache = (0, 0)
        if kind == "shim":
            address = bd.base_address_low << 2 | bd.base_address_high << 32
            burst_axcache = (bd.burst_length, bd.axcache)
        else:
            address = bd.base_address
        if kind == "mem":
            # A zero d2 wrap leaves a memtile's d2 unbounded, as the shim's
            # d2 is: buffer_length ends the transfer and d3 never steps.
            if bd.d2_wrap:
                dims += ((bd.d2_wrap, bd.d2_stepsize + 1),)
                outer_stride = bd.d3_stepsize + 1
            pads = tuple(
                (getattr(bd, f"d{d}_zero_before"), getattr(bd, f"d{d}_zero_after"))
                for d in range(3)
            )
            if any(b or a for b, a in pads):
                padding = pads
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
            if dims and _contiguous(dims, outer_stride, bd.buffer_length):
                dims, outer_stride = (), 1
        return Transfer(
            length=bd.buffer_length,
            address=patched or ("abs", address),
            dims=dims,
            outer_stride=outer_stride,
            iteration=(bd.iteration_wrap, bd.iteration_stepsize + 1),
            packet=tuple(getattr(bd, f) for f in _PACKET_FIELDS),
            flags=tuple(getattr(bd, f, 0) for f in _FLAG_FIELDS),
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
    kind: str  # push | wait | write | loadpdi | preempt | scratchpad | update_reg
    col: int = 0
    row: int = 0
    direction: str = ""  # S2MM | MM2S
    channel: int = 0
    bd: Transfer | None = None
    chain: tuple[Transfer, ...] = ()  # the BDs after `bd`, in chain order
    chain_loop: int | None = None  # where in (bd, *chain) the last BD continues
    repeat: int = 0
    issue_token: bool = False
    ctrl: int = 0
    raw: tuple[int, ...] = ()

    def __str__(self) -> str:
        where = f"({self.col},{self.row}) {self.direction} ch{self.channel}"
        if self.kind == "push":
            tok = " token" if self.issue_token else ""
            rep = f" repeat={self.repeat}" if self.repeat else ""
            chain = "".join(f" -> {t}" for t in self.chain)
            loop = (
                f" -> back to BD {self.chain_loop} of the chain"
                if self.chain_loop is not None
                else ""
            )
            return f"push {where}{rep}{tok} ctrl={self.ctrl:#x}: {self.bd}{chain}{loop}"
        if self.kind == "wait":
            return f"wait {where} cols={self.raw[0]} rows={self.raw[1]}"
        if self.kind == "write":
            reg, value, mask = self.raw
            masked = f" & {mask:#010x}" if mask != 0xFFFFFFFF else ""
            return f"write ({self.col},{self.row}) {reg:#07x} <- {value:#010x}{masked}"
        return f"{self.kind} {self.raw}"


def trace(words: Sequence[int] | np.ndarray) -> list[Event]:
    """Replay a stream and return the events it triggers, in order."""
    ops = decode(words)
    header = Header.of(ops[0].words)
    regs: dict[int, int] = {}
    patches: dict[int, tuple] = {}  # BD address-word register -> ("arg", idx, plus)
    events: list[Event] = []

    def locate(addr: int) -> tuple[int, int, str, int]:
        row = (addr >> _ROW_SHIFT) & 0x1F
        return (addr >> _COL_SHIFT) & 0x7F, row, header.tile_kind(row), addr & _REG_MASK

    def queue(kind: str, reg: int) -> tuple[str, int] | None:
        """Return the (direction, channel) whose queue register `reg` is, if any."""
        layout = _LAYOUT[kind]
        for direction, delta in (("S2MM", 0), ("MM2S", layout.mm2s_delta)):
            for ch in range(layout.channels):
                if reg == layout.ctrl_base + delta + ch * 8 + 4:
                    return direction, ch
        return None

    def bd_chain(
        op: Op, col: int, row: int, kind: str, bd_id: int
    ) -> tuple[list[Transfer], int | None]:
        layout = _LAYOUT[kind]
        tile = (col << _COL_SHIFT) | (row << _ROW_SHIFT)
        transfers: list[Transfer] = []
        seen: dict[int, int] = {}
        while bd_id not in seen:
            seen[bd_id] = len(transfers)
            bd_base = tile | (layout.bd_base + bd_id * _BD_STRIDE)
            addrs = [bd_base + 4 * i for i in range(ctypes.sizeof(layout.bd) // 4)]
            # Only shim BDs address host memory, through the word-1 patch.
            patched = patches.get(bd_base + 4) if kind == "shim" else None
            missing = [
                i
                for i, a in enumerate(addrs)
                if a not in regs and not (patched and i == 1)
            ]
            if missing:
                raise ValueError(
                    f"the push at word {op.pos} runs BD {bd_id} of tile "
                    f"({col},{row}), but the stream never writes its words {missing}"
                )
            w = [regs.get(a, 0) for a in addrs]
            transfers.append(Transfer.from_words(kind, w, patched))
            bd = layout.bd.of(w)
            if not bd.use_next_bd:
                return transfers, None
            bd_id = bd.next_bd
        return transfers, seen[bd_id]

    def write(addr: int, value: int, mask: int) -> None:
        col, row, kind, reg = locate(addr)
        if not _LAYOUT[kind].owns(reg):
            events.append(Event("write", col=col, row=row, raw=(reg, value, mask)))

    for op in ops:
        if op.kind == "write32":
            regs[op.addr] = op.value
        elif op.kind == "maskwrite32":
            regs[op.addr] = (regs.get(op.addr, 0) & ~op.mask) | (op.value & op.mask)
            write(op.addr, op.value & op.mask, op.mask)
        elif op.kind == "blockwrite":
            # A fresh BD image supersedes an earlier patch of its address word.
            for i, d in enumerate(op.data):
                regs[op.addr + 4 * i] = d
                patches.pop(op.addr + 4 * i, None)
                write(op.addr + 4 * i, d, 0xFFFFFFFF)
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
        elif op.kind == "scratchpad":
            events.append(Event(op.kind, raw=(op.data[0], op.value)))
        elif op.kind == "update_reg":
            events.append(Event(op.kind, raw=(*op.data, op.value, op.addr)))
        elif op.kind in ("loadpdi", "preempt"):
            events.append(Event(op.kind, raw=op.words))
        if op.kind != "write32":
            continue
        col, row, kind, reg = locate(op.addr)
        hit = queue(kind, reg)
        if hit is None:
            write(op.addr, op.value, 0xFFFFFFFF)
            continue
        direction, ch = hit
        push = _LAYOUT[kind].queue_push.of([op.value])
        repeat = push.repeat_count
        (bd, *chain), loop = bd_chain(op, col, row, kind, push.start_bd_id)
        # A linear BD re-run repeat+1 times with an iteration dimension that
        # advances by its own length is one linear transfer that long; the
        # static emitter folds a contiguous repeat dimension into the length.
        iter_wrap, iter_stride = bd.iteration[0] + 1, bd.iteration[1]
        if (
            kind == "shim"
            and not chain
            and loop is None
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
                chain=tuple(chain),
                chain_loop=loop,
                repeat=repeat,
                issue_token=bool(push.enable_token_issue),
                ctrl=regs.get(op.addr - 4, 0),
            )
        )
    return events


def compare(
    a: Sequence[int] | np.ndarray, b: Sequence[int] | np.ndarray, *, names=("a", "b")
) -> list[str]:
    """Return the differences between two streams' devices and events.

    Empty when the streams are equivalent.
    """
    ha, hb = Header.of(decode(a)[0].words), Header.of(decode(b)[0].words)
    if ha.device != hb.device:
        return [f"headers differ:\n  {names[0]}: {ha}\n  {names[1]}: {hb}"]
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
    """Human-readable listing of a stream's events (or raw ops)."""
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
