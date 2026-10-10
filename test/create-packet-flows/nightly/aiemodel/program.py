#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""The deadlock model of docs/DeadlockModel.md, run in every order on small
designs: the oracle the deadlock engine is checked against.

It reads a design's MLIR into programs of its own rather than through
Design, which keeps only what the router's rules read, so a mistake in one
front end cannot hide the same mistake in the other. It explores every
interleaving, one word or one lock event at a time, so it is exact and slow:
a design whose state space passes `max_states` is reported undecided.
"""

import re
from collections import deque
from dataclasses import dataclass, field

from aie.ir import Location, Module
import aie.dialects.aie  # noqa: F401
import aie.dialects.aiex  # noqa: F401

from .design import _attrs, _context, _elem_bits, _int, _num_elems, _ops, _pkt

ACQUIRE_EQ, RELEASE, ACQUIRE_GE = 0, 1, 2
S2MM, MM2S = 0, 1
DMA, TRACE = 1, 9  # WireBundle
WORD = 4
# A loop at least this long in a core is how a design says "forever".
FOREVER = 1 << 30
# Loops in a core or a runtime sequence are unrolled up to this many ops.
MAX_UNROLL = 4096
# Ops that place, configure or name things and run no program.
INERT = {
    "aie.tile",
    "aie.lock",
    "aie.buffer",
    "aie.external_buffer",
    "aie.end",
    "aie.switchbox",
    "aie.shim_mux",
    "aie.wire",
    "aie.cascade_flow",
    "arith.constant",
    "func.func",
}


@dataclass(frozen=True)
class BD:
    """One buffer descriptor: its acquire, the words it moves, its packet id
    and whether a receiver keeps the header, and its release."""

    words: int
    acquire: tuple | None = None  # (action, lock, value)
    release: tuple | None = None
    packet: int | None = None


@dataclass(frozen=True)
class Chain:
    """BDs a channel runs in order, `passes` times, or forever from
    `loop_to` when the last BD leads back into the chain."""

    bds: tuple
    passes: int = 1
    loop_to: int | None = None
    token: bool = False
    name: str = ""


@dataclass
class System:
    locks: dict = field(default_factory=dict)  # name -> init
    cores: dict = field(default_factory=dict)  # tile -> (prefix ops, loop ops)
    static: dict = field(default_factory=dict)  # channel -> Chain
    # Stream connectivity: (tile, dir MM2S, ch, packet id or None) -> [receivers]
    # and per receiver (tile, S2MM, ch) whether it keeps headers.
    sends: dict = field(default_factory=dict)
    keeps: dict = field(default_factory=dict)
    allocs: dict = field(default_factory=dict)  # symbol -> (channel, packet)
    sequences: dict = field(default_factory=dict)  # name -> [host op]
    unsupported: list = field(default_factory=list)


def _words(nbytes):
    return -(-nbytes // WORD)


def _sizes(attr):
    return [int(v) for v in re.findall(r"-?\d+", str(attr).split(":")[-1])]


class _Unsupported(Exception):
    pass


class _Forever(Exception):
    """A loop that never exits: the events before it repeats, and the
    events it repeats forever."""

    def __init__(self, loop):
        super().__init__()
        self.loop = loop


UNKNOWN = None
# arith.cmpi predicates, in their enum order.
CMP = [
    lambda a, b: a == b,
    lambda a, b: a != b,
    lambda a, b: a < b,
    lambda a, b: a <= b,
    lambda a, b: a > b,
    lambda a, b: a >= b,
    lambda a, b: a < b,
    lambda a, b: a <= b,
    lambda a, b: a > b,
    lambda a, b: a >= b,
]
BINARY = {
    "arith.addi": lambda a, b: a + b,
    "arith.subi": lambda a, b: a - b,
    "arith.muli": lambda a, b: a * b,
    "arith.divsi": lambda a, b: int(a / b) if b else UNKNOWN,
    "arith.divui": lambda a, b: a // b if b else UNKNOWN,
    "arith.floordivsi": lambda a, b: a // b if b else UNKNOWN,
    "arith.ceildivsi": lambda a, b: -(-a // b) if b else UNKNOWN,
    "arith.remsi": lambda a, b: a - b * int(a / b) if b else UNKNOWN,
    "arith.remui": lambda a, b: a % b if b else UNKNOWN,
    "arith.maxsi": max,
    "arith.minsi": min,
    "arith.maxui": max,
    "arith.minui": min,
    "arith.andi": lambda a, b: a & b,
    "arith.ori": lambda a, b: a | b,
    "arith.xori": lambda a, b: a ^ b,
    "arith.shli": lambda a, b: a << b,
    "arith.shrsi": lambda a, b: a >> b,
    "arith.shrui": lambda a, b: a >> b,
}
CASTS = (
    "arith.index_cast",
    "arith.index_castui",
    "arith.extsi",
    "arith.extui",
    "arith.trunci",
)
# Ops the interpreter steps over: they move data, not tokens.
PURE = (
    "memref.load",
    "memref.store",
    "memref.assume_alignment",
    "memref.subview",
    "memref.reinterpret_cast",
    "memref.get_global",
    "func.call",
    "aie.end",
    "cf.assert",
    "aiex.npu.rtp_write",
)


class _Interp:
    """Runs a core body or a runtime sequence on the integers it can know,
    turning the ops `event` recognizes into events. A value read from memory
    or computed from one is unknown; control flow that depends on it, around
    an event, is outside the model."""

    def __init__(self, event):
        self.event = event
        self.env = {}

    def val(self, v):
        return self.env.get(hash(v), UNKNOWN)

    def bind(self, values, args):
        for v, a in zip(values, args):
            self.env[hash(a)] = v

    def has_events(self, op):
        return any(
            self.event(x, None) or self.has_events(x)
            for region in op.regions
            for b in region.blocks
            for x in _ops(b)
        )

    def block(self, block, out):
        """Runs `block`, returning its terminator's operand values."""
        for op in _ops(block):
            if op.name in ("scf.yield", "scf.condition"):
                return [self.val(v) for v in op.operands]
            self.op(op, out)
        return []

    def op(self, op, out):
        name = op.name
        a = _attrs(op)
        if name == "arith.constant":
            try:
                self.env[hash(op.results[0])] = _int(a["value"])
            except (KeyError, ValueError):
                self.env[hash(op.results[0])] = UNKNOWN
        elif name in BINARY:
            x, y = (self.val(v) for v in op.operands)
            r = UNKNOWN if UNKNOWN in (x, y) else BINARY[name](x, y)
            self.env[hash(op.results[0])] = r
        elif name in CASTS:
            self.env[hash(op.results[0])] = self.val(op.operands[0])
        elif name == "arith.cmpi":
            x, y = (self.val(v) for v in op.operands)
            r = UNKNOWN if UNKNOWN in (x, y) else int(CMP[_int(a["predicate"])](x, y))
            self.env[hash(op.results[0])] = r
        elif name == "arith.select":
            c, x, y = (self.val(v) for v in op.operands)
            self.env[hash(op.results[0])] = UNKNOWN if c is UNKNOWN else (x if c else y)
        elif self.event(op, out):
            pass
        elif name == "scf.for":
            self.loop_for(op, out)
        elif name == "scf.while":
            self.loop_while(op, out)
        elif name == "scf.if":
            c = self.val(op.operands[0])
            if c is UNKNOWN:
                if self.has_events(op):
                    raise _Unsupported("tokens under data-dependent control")
                self.bind([UNKNOWN] * len(op.results), op.results)
                return
            region = op.regions[0 if c else 1]
            vals = self.block(region.blocks[0], out) if region.blocks else []
            self.bind(vals, op.results)
        elif name == "scf.index_switch":
            sel = self.val(op.operands[0])
            cases = _sizes(a["cases"])
            if sel is UNKNOWN:
                if self.has_events(op):
                    raise _Unsupported("tokens under data-dependent control")
                self.bind([UNKNOWN] * len(op.results), op.results)
                return
            region = op.regions[1 + cases.index(sel)] if sel in cases else op.regions[0]
            self.bind(self.block(region.blocks[0], out), op.results)
        elif name in PURE or (name.startswith("arith.") and not op.regions):
            self.bind([UNKNOWN] * len(op.results), op.results)
        else:
            raise _Unsupported(name)

    def loop_for(self, op, out):
        lb, ub, step = (self.val(v) for v in list(op.operands)[:3])
        carried = [self.val(v) for v in list(op.operands)[3:]]
        body = op.regions[0].blocks[0]
        args = list(body.arguments)
        if UNKNOWN in (lb, ub, step) or step <= 0:
            if self.has_events(op):
                raise _Unsupported("a loop whose bounds are not known, around tokens")
            self.bind([UNKNOWN] * len(op.results), op.results)
            return
        n = max(0, -(-(ub - lb) // step))
        if not self.has_events(op) and not carried:
            return
        forever = n >= FOREVER
        seen = {}
        i = 0
        while forever or i < n:
            if forever:
                state = tuple(carried)
                if UNKNOWN not in state and state in seen:
                    raise _Forever(tuple(out[seen[state] :]))
                seen[state] = len(out)
            self.bind([lb + i * step] + carried, args)
            carried = self.block(body, out)
            if len(out) > MAX_UNROLL:
                raise _Unsupported("a loop too long to unroll")
            i += 1
        self.bind(carried, op.results)

    def loop_while(self, op, out):
        carried = [self.val(v) for v in op.operands]
        before, after = op.regions[0].blocks[0], op.regions[1].blocks[0]
        seen = {}
        while True:
            state = tuple(carried)
            if UNKNOWN not in state and state in seen:
                raise _Forever(tuple(out[seen[state] :]))
            seen[state] = len(out)
            self.bind(carried, list(before.arguments))
            cond = self.block(before, out)
            if cond[0] is UNKNOWN:
                raise _Unsupported("a while loop whose condition is not known")
            if not cond[0]:
                self.bind(cond[1:], op.results)
                return
            self.bind(cond[1:], list(after.arguments))
            carried = self.block(after, out)
            if len(out) > MAX_UNROLL:
                raise _Unsupported("a loop too long to unroll")


def load_system(text):
    """Parse one aie.device into a System. Constructs the model does not
    cover are listed in `unsupported`, never guessed."""
    s = System()
    tiles, locks, tasks = {}, {}, {}

    def key(v):
        return hash(v)

    def consts_in(region):
        env = {}
        for b in region.blocks:
            for x in _ops(b):
                if x.name == "arith.constant":
                    try:
                        env[key(x.results[0])] = _int(_attrs(x)["value"])
                    except (KeyError, ValueError):
                        pass
        return env

    def lock_event(op, env):
        a = _attrs(op)
        n = env.get(key(op.operands[1])) if len(op.operands) > 1 else None
        if n is None:
            raise _Unsupported("a lock value the model cannot know")
        return (_int(a["action"]), locks[key(op.operands[0])], n)

    def bd_words(op):
        a = _attrs(op)
        ty = str(op.operands[0].type)
        bits = _elem_bits(ty) or 32
        if "static_len" in a:
            n = _int(a["static_len"])
        elif "static_sizes" in a:
            n = 1
            for v in _sizes(a["static_sizes"]):
                n *= v
        else:
            n = _num_elems(ty) or 0
        return _words(n * max(bits // 8, 1))

    def bd_of(block, env):
        acq = rel = None
        bd = None
        header = None
        for op in _ops(block):
            if op.name == "aie.dma_bd_packet":
                header = _int(_attrs(op)["packet_id"])
            elif op.name == "aie.use_lock":
                ev = lock_event(op, env)
                if ev[0] == RELEASE:
                    rel = ev
                else:
                    acq = ev
            elif op.name == "aie.dma_bd":
                a = _attrs(op)
                bd = (bd_words(op), _pkt(a["packet"]) if "packet" in a else None)
        if bd is None:
            return None
        return BD(bd[0], acq, rel, bd[1] if bd[1] is not None else header)

    def chain_from(blocks, first, env):
        bds, seen, loop_to = [], [], None
        blk = first
        while blk is not None:
            idx = blocks.index(blk)
            if idx in seen:
                loop_to = seen.index(idx)
                break
            seen.append(idx)
            bd = bd_of(blocks[idx], env)
            if bd is not None:
                bds.append(bd)
            nxt = None
            for op in _ops(blocks[idx]):
                if op.name == "aie.next_bd":
                    nxt = list(op.successors)[0]
            blk = nxt
        return bds, loop_to

    def dma_region(op, tile):
        blocks = list(op.regions[0].blocks)
        env = consts_in(op.regions[0])
        for b in blocks:
            for x in _ops(b):
                if x.name == "aie.dma_start":
                    a = _attrs(x)
                    bds, loop_to = chain_from(blocks, list(x.successors)[0], env)
                    rep = _int(a["repeat_count"]) if "repeat_count" in a else 0
                    ch = (tile, _int(a["channel_dir"]), _int(a["channel"]))
                    s.static[ch] = Chain(tuple(bds), rep + 1, loop_to)
                elif x.name == "aie.dma":
                    raise _Unsupported("aie.dma")

    def core(op, tile):
        interp = None

        def event(x, out):
            if x.name != "aie.use_lock":
                return False
            if out is not None:
                out.append(("lock",) + lock_event(x, interp.env))
            return True

        interp = _Interp(event)
        out = []
        try:
            for b in op.regions[0].blocks:
                interp.block(b, out)
            s.cores[tile] = (tuple(out), ())
        except _Forever as f:
            s.cores[tile] = (tuple(out[: len(out) - len(f.loop)]), f.loop)

    def task_chain(op, a, name, env):
        blocks = list(op.regions[0].blocks)
        env = {**env, **consts_in(op.regions[0])}
        bds, loop_to = chain_from(blocks, blocks[0], env)
        rep = _int(a["repeat_count"]) if "repeat_count" in a else 0
        token = "issue_token" in a and str(a["issue_token"]) == "true"
        return Chain(tuple(bds), rep + 1, loop_to, token, name)

    def memcpy_sizes(op, interp):
        a = _attrs(op)
        static = _sizes(a["static_sizes"])
        segs = _sizes(a["operandSegmentSizes"])
        operands = list(op.operands)
        start = segs[0] + segs[1]
        dyn = [interp.val(v) for v in operands[start : start + segs[2]]]
        sizes = []
        for v in static:
            sizes.append(dyn.pop(0) if v < 0 else v)
        return sizes

    def sequence(op):
        interp = None

        def event(x, out):
            if not x.name.startswith("aiex.") or x.name == "aiex.npu.rtp_write":
                return False
            if out is None:
                return True
            a = _attrs(x)
            if x.name == "aiex.npu.dma_memcpy_nd":
                sym = str(a["metadata"]).lstrip("@")
                if sym not in s.allocs:
                    raise _Unsupported(f"a transfer on {sym}, which has no allocation")
                channel, alloc_pkt = s.allocs[sym]
                sizes = memcpy_sizes(x, interp)
                bits = _elem_bits(str(x.operands[0].type))
                if UNKNOWN in sizes or bits is None:
                    raise _Unsupported("a runtime transfer of runtime size")
                total = 1
                for v in sizes:
                    total *= v
                runs = sizes[0]
                pkt = _pkt(a["packet"]) if "packet" in a else alloc_pkt
                bd = BD(_words(total // runs * bits // 8), packet=pkt)
                token = channel[1] == S2MM or (
                    "issue_token" in a and str(a["issue_token"]) == "true"
                )
                out.append(("push", channel, Chain((bd,), runs, None, token, sym)))
            elif x.name in ("aiex.dma_configure_task_for", "aiex.dma_configure_task"):
                if x.name == "aiex.dma_configure_task_for":
                    sym = str(a["alloc"]).lstrip("@")
                    if sym not in s.allocs:
                        raise _Unsupported(f"a task on {sym}, which has no allocation")
                    channel = s.allocs[sym][0]
                else:
                    tile = tiles[key(x.operands[0])]
                    channel = (tile, _int(a["direction"]), _int(a["channel"]))
                tasks[key(x.results[0])] = (
                    channel,
                    task_chain(x, a, f"task{len(tasks)}", interp.env),
                )
            elif x.name == "aiex.dma_start_task":
                t = tasks.get(key(x.operands[0]))
                if t is None:
                    raise _Unsupported("a started task the model cannot see")
                out.append(("push", t[0], t[1]))
            elif x.name == "aiex.dma_await_task":
                t = tasks.get(key(x.operands[0]))
                out.append(("await", t[1].name if t else None))
            elif x.name == "aiex.dma_free_task":
                t = tasks.get(key(x.operands[0]))
                out.append(("free", t[1].name if t else None))
            elif x.name == "aiex.npu.dma_wait":
                out.append(("wait", str(a["symbol"]).lstrip("@")))
            elif x.name == "aiex.set_lock":
                out.append(("set", locks[key(x.operands[0])], _int(a["value"])))
            else:
                raise _Unsupported(f"{x.name} in a runtime sequence")
            return True

        interp = _Interp(event)
        out = []
        interp.block(op.regions[0].blocks[0], out)
        return out

    def register(op):
        a = _attrs(op)
        if op.name == "aie.tile":
            tiles[key(op.results[0])] = (_int(a["col"]), _int(a["row"]))
        elif op.name == "aie.lock":
            name = (
                str(a["sym_name"]).strip('"')
                if "sym_name" in a
                else f"lock{len(locks)}"
            )
            locks[key(op.results[0])] = name
            s.locks[name] = _int(a["init"]) if "init" in a else 0
        elif op.name == "aie.shim_dma_allocation":
            tile = tiles[key(op.operands[0])]
            ch = (tile, _int(a["channel_dir"]), _int(a["channel_index"]))
            pkt = _pkt(a["packet"]) if "packet" in a else None
            s.allocs[str(a["sym_name"]).strip('"')] = (ch, pkt)
        for region in op.regions:
            for b in region.blocks:
                for x in _ops(b):
                    register(x)

    def flow_end(x, xa, sending):
        bundle = _int(xa["bundle"])
        if bundle == TRACE and sending:
            return "trace"
        if bundle != DMA:
            raise _Unsupported("a packet flow end that is not a DMA")
        t = tiles[key(x.operands[0])]
        return (t, MM2S if sending else S2MM, _int(xa["channel"]))

    with _context(), Location.unknown():
        module = Module.parse(text)
        device = next(op for op in _ops(module.body) if op.name == "aie.device")
        body = device.regions[0].blocks[0]
        for op in _ops(body):
            register(op)
        for op in _ops(body):
            a = _attrs(op)
            try:
                if op.name in ("aie.mem", "aie.memtile_dma", "aie.shim_dma"):
                    dma_region(op, tiles[key(op.operands[0])])
                elif op.name == "aie.core":
                    core(op, tiles[key(op.operands[0])])
                elif op.name == "aie.flow":
                    src = tiles[key(op.operands[0])]
                    dst = tiles[key(op.operands[1])]
                    if _int(a["source_bundle"]) == TRACE:
                        continue
                    if _int(a["source_bundle"]) != DMA or _int(a["dest_bundle"]) != DMA:
                        raise _Unsupported("a flow that does not run DMA to DMA")
                    snd = (src, MM2S, _int(a["source_channel"]), None)
                    s.sends.setdefault(snd, []).append(
                        (dst, S2MM, _int(a["dest_channel"]))
                    )
                elif op.name == "aie.packet_flow":
                    pid = _int(a["ID"])
                    keep = (
                        "keep_pkt_header" in a and str(a["keep_pkt_header"]) == "true"
                    )
                    srcs, dsts = [], []
                    for x in _ops(op.regions[0].blocks[0]):
                        xa = _attrs(x)
                        if x.name == "aie.packet_source":
                            srcs.append(flow_end(x, xa, True))
                        elif x.name == "aie.packet_dest":
                            dsts.append(flow_end(x, xa, False))
                    # Trace streams carry nothing any agent waits on.
                    if "trace" in srcs:
                        continue
                    for src in srcs:
                        s.sends.setdefault(src + (pid,), []).extend(dsts)
                    for dst in dsts:
                        s.keeps[dst] = keep
                elif op.name == "aie.runtime_sequence":
                    name = str(a["sym_name"]).strip('"') if "sym_name" in a else ""
                    s.sequences[name] = sequence(op)
                elif op.name not in INERT and op.name not in (
                    "aie.shim_dma_allocation",
                    "aie.trace",
                    "aie.packet_flow",
                ):
                    raise _Unsupported(op.name)
            except _Unsupported as e:
                s.unsupported.append(str(e))
    return s


@dataclass
class Verdict:
    """What the model says of one dispatch of one runtime sequence."""

    outcome: str  # "accept", "deadlock", "unquiesced", "undecided"
    reason: str = ""
    schedule: list = field(default_factory=list)  # moves to the first deadlock
    states: int = 0
    unquiesced: list = field(default_factory=list)  # at the runs' ends


class _Run:
    """The state space of one dispatch, explored breadth first.

    A state is a tuple: lock values, each core's position, each channel's
    progress and queue, each receiver's buffered words and the packet it is
    taking, the host's position, and finished task names."""

    def __init__(self, system, sequence, capacity):
        self.s = system
        self.cap = capacity
        self.host = [] if sequence is None else system.sequences[sequence]
        self.locks = sorted(system.locks)
        self.lock_idx = {n: i for i, n in enumerate(self.locks)}
        self.cores = sorted(system.cores)
        channels = set(system.static)
        for op in self.host:
            if op[0] == "push":
                channels.add(op[1])
        self.channels = sorted(channels)
        receivers = set()
        for dsts in system.sends.values():
            receivers.update(dsts)
        self.receivers = sorted(receivers)
        self.rx_idx = {r: i for i, r in enumerate(self.receivers)}

    def initial(self):
        locks = tuple(self.s.locks[n] for n in self.locks)
        cores = tuple(0 for _ in self.cores)
        # A channel: (running chain or None, bd index, phase, words done,
        # pass, queued chains).
        chans = tuple((self.s.static.get(c), 0, 0, 0, 0, ()) for c in self.channels)
        # A receiver: (buffered words, sending channel and packet or None,
        # words of that packet still to come).
        rxs = tuple((0, None, 0) for _ in self.receivers)
        return (locks, cores, chans, rxs, 0, frozenset())

    # Moves.

    def lock_try(self, locks, ev):
        action, name, v = ev
        i = self.lock_idx[name]
        val = locks[i]
        if action == ACQUIRE_GE:
            if val < v:
                return None
            val -= v
        elif action == ACQUIRE_EQ:
            if val != v:
                return None
        else:
            val += v
        return locks[:i] + (val,) + locks[i + 1 :]

    def core_moves(self, st):
        locks, cores = st[0], st[1]
        for ci, tile in enumerate(self.cores):
            prefix, loop = self.s.cores[tile]
            pc = cores[ci]
            if pc < len(prefix):
                op = prefix[pc]
            elif loop:
                op = loop[(pc - len(prefix)) % len(loop)]
            else:
                continue
            nl = self.lock_try(locks, op[1:])
            if nl is None:
                continue
            nxt = pc + 1
            if loop and nxt >= len(prefix) + len(loop):
                nxt -= len(loop)
            yield (
                f"core {tile} {op}",
                (nl, cores[:ci] + (nxt,) + cores[ci + 1 :]) + st[2:],
            )

    def receivers_of(self, chan, bd):
        return self.s.sends.get(chan + (bd.packet,), [])

    def taking(self, chans, dst):
        """Whether the channel at `dst` is mid-transfer on a BD, so a word
        sent now passes straight into it."""
        if dst not in self.channels:
            return False
        chain, bi, phase, w, pss, queue = chans[self.channels.index(dst)]
        return chain is not None and bool(chain.bds) and phase == 1

    def channel_moves(self, st):
        locks, cores, chans, rxs, hpc, done = st
        for ki, ch in enumerate(self.channels):
            chain, bi, phase, w, pss, queue = chans[ki]
            if chain is None:
                if queue:
                    nc = list(chans)
                    nc[ki] = (queue[0], 0, 0, 0, 0, queue[1:])
                    yield (
                        f"{ch} starts {queue[0].name}",
                        (locks, cores, tuple(nc), rxs, hpc, done),
                    )
                continue
            if not chain.bds:
                nc = list(chans)
                nc[ki] = (None, 0, 0, 0, 0, queue)
                nd = done | {chain.name} if chain.name else done
                yield (
                    f"{ch} ends {chain.name}",
                    (locks, cores, tuple(nc), rxs, hpc, nd),
                )
                continue
            bd = chain.bds[bi]

            def advance(nlocks, nphase, nw, nrxs):
                nb, npass = bi, pss
                if nphase == 3:
                    nb, nphase, nw = bi + 1, 0, 0
                    if nb == len(chain.bds):
                        if chain.loop_to is not None:
                            nb = chain.loop_to
                        else:
                            npass, nb = pss + 1, 0
                nc = list(chans)
                nd = done
                if npass == chain.passes:
                    nc[ki] = (None, 0, 0, 0, 0, queue)
                    if chain.name:
                        nd = done | {chain.name}
                else:
                    nc[ki] = (chain, nb, nphase, nw, npass, queue)
                return (nlocks, cores, tuple(nc), nrxs, hpc, nd)

            if phase == 0:
                if bd.acquire is None:
                    yield (f"{ch} bd {bi}", advance(locks, 1, 0, rxs))
                else:
                    nl = self.lock_try(locks, bd.acquire)
                    if nl is not None:
                        yield (f"{ch} acquires {bd.acquire}", advance(nl, 1, 0, rxs))
            elif phase == 2:
                nl = locks if bd.release is None else self.lock_try(locks, bd.release)
                yield (f"{ch} releases {bd.release}", advance(nl, 3, 0, rxs))
            elif ch[1] == MM2S:
                dsts = self.receivers_of(ch, bd)
                header = bd.packet is not None
                total = bd.words + (1 if header else 0)
                if not dsts:
                    # Nothing the model knows takes it: it waits forever.
                    continue
                nrx = list(rxs)
                ok = True
                for dst in dsts:
                    i = self.rx_idx[dst]
                    buf, owner, left = rxs[i]
                    me = (ch, bd.packet)
                    if owner not in (None, me) or buf >= max(self.cap, 1):
                        ok = False
                        break
                    if self.cap == 0 and not self.taking(chans, dst):
                        ok = False
                        break
                    keep = self.s.keeps.get(dst, False)
                    counts = not (header and w == 0 and not keep)
                    nleft = total - w - 1
                    nrx[i] = (buf + (1 if counts else 0), me if nleft else None, nleft)
                if ok:
                    nw = w + 1
                    yield (
                        f"{ch} sends",
                        advance(locks, 2 if nw == total else 1, nw, tuple(nrx)),
                    )
            else:
                i = self.rx_idx.get(ch)
                if i is None:
                    continue
                buf, owner, left = rxs[i]
                if buf == 0:
                    continue
                nrx = list(rxs)
                nrx[i] = (buf - 1, owner, left)
                nw = w + 1
                yield (
                    f"{ch} receives",
                    advance(locks, 2 if nw == bd.words else 1, nw, tuple(nrx)),
                )

    def host_moves(self, st):
        locks, cores, chans, rxs, hpc, done = st
        if hpc >= len(self.host):
            return
        op = self.host[hpc]
        nxt = (locks, cores, chans, rxs, hpc + 1, done)
        if op[0] == "push":
            ki = self.channels.index(op[1])
            chain, bi, phase, w, pss, queue = chans[ki]
            if len(queue) + (chain is not None) >= 4:
                return
            nc = list(chans)
            nc[ki] = (chain, bi, phase, w, pss, queue + (op[2],))
            yield (
                f"host pushes {op[2].name} to {op[1]}",
                (locks, cores, tuple(nc), rxs, hpc + 1, done),
            )
        elif op[0] == "wait":
            pending = [
                c
                for k, c in self._chains_on(st, op[1])
                if c.token and c.name not in done
            ]
            if not pending:
                yield (f"host waits on {op[1]}", nxt)
        elif op[0] == "await":
            if op[1] in done:
                yield (f"host awaits {op[1]}", nxt)
        elif op[0] == "free":
            yield (f"host frees {op[1]}", nxt)
        elif op[0] == "set":
            i = self.lock_idx[op[1]]
            yield (
                f"host sets {op[1]}",
                (locks[:i] + (op[2],) + locks[i + 1 :],) + nxt[1:],
            )

    def _chains_on(self, st, sym):
        chans = st[2]
        out = []
        for ki, ch in enumerate(self.channels):
            chain, _, _, _, _, queue = chans[ki]
            for c in ((chain,) if chain else ()) + queue:
                if c.name == sym:
                    out.append((ch, c))
        return out

    def moves(self, st):
        yield from self.host_moves(st)
        yield from self.core_moves(st)
        yield from self.channel_moves(st)

    def unquiesced(self, st):
        locks, cores, chans, rxs, hpc, done = st
        out = []
        for ki, ch in enumerate(self.channels):
            chain, bi, phase, w, pss, queue = chans[ki]
            if queue or (chain is not None and ch not in self.s.static):
                out.append(f"{ch} still has work")
            elif chain is not None and phase == 1 and w:
                out.append(f"{ch} stopped partway through a BD")
        for i, r in enumerate(self.receivers):
            if rxs[i][0]:
                out.append(f"{r} still holds {rxs[i][0]} words")
        return out


def explore(system, sequence=None, capacity=0, max_states=200000):
    """Every order one dispatch of `sequence` can run in, with `capacity`
    words of buffering in front of each receiver (0: a word passes only
    straight from a sender to a receiver taking it). With no sequence, the
    design runs on its own until nothing can move."""
    if system.unsupported:
        return Verdict("undecided", "; ".join(sorted(set(system.unsupported))))
    if sequence is not None and sequence not in system.sequences:
        return Verdict("undecided", f"no runtime sequence {sequence!r}")
    driven = set(system.static) | {ch for ch, _ in system.allocs.values()}
    ends = {snd[:3] for snd in system.sends}
    ends.update(r for dsts in system.sends.values() for r in dsts)
    hidden = sorted(e for e in ends if e not in driven)
    if hidden:
        return Verdict(
            "undecided",
            f"a flow names {hidden[0]}, whose channel the design does not program",
        )
    run = _Run(system, sequence, capacity)
    start = run.initial()
    parent = {start: None}
    frontier = deque([start])
    deadlocks, ends = [], []
    while frontier:
        st = frontier.popleft()
        succ = list(run.moves(st))
        if not succ:
            (ends if st[4] >= len(run.host) else deadlocks).append(st)
            continue
        for label, nst in succ:
            if nst not in parent:
                parent[nst] = (st, label)
                if len(parent) > max_states:
                    return Verdict(
                        "undecided",
                        f"more than {max_states} states",
                        states=len(parent),
                    )
                frontier.append(nst)
    unq = sorted({u for st in ends for u in run.unquiesced(st)})
    if not deadlocks:
        return Verdict(
            "unquiesced" if unq else "accept", states=len(parent), unquiesced=unq
        )
    schedule = []
    st = deadlocks[0]
    while parent[st] is not None:
        st, label = parent[st]
        schedule.append(label)
    schedule.reverse()
    if ends:
        return Verdict(
            "undecided",
            "some orders deadlock and some finish: outside the deterministic subset",
            schedule,
            len(parent),
            unq,
        )
    return Verdict("deadlock", "", schedule, len(parent), unq)
