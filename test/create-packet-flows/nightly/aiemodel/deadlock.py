#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Which streams can deadlock, mirrored from
AIEStreamDependencyAnalysis.cpp line for line."""

from collections import defaultdict, deque

from .params import PARAMS
from .fabric import (
    CORE,
    CTRL,
    DIRECTIONAL,
    DIRS,
    DMA,
    MM2S,
    NORTH,
    S2MM,
    SOUTH,
    STEP,
    fmt_ep,
    fmt_port,
)
from .design import keeps_header, last_keep

# The deadlock rules, mirrored from AIEStreamDependencyAnalysis.cpp. Names
# follow the C++ so the two read side by side.


class Stream:
    __slots__ = ("src", "dst", "pid", "keep", "hops", "mask")

    def __init__(self, src, dst, pid=None, keep=False, hops=(), mask=None):
        self.src, self.dst, self.pid, self.keep = src, dst, pid, keep
        self.hops = list(hops)  # [(tile, input port, arbiter or None)]
        self.mask = -1 if mask is None else mask

    def carries(self, pid):
        """carriesID: whether the stream carries packets with id `pid`."""
        return self.pid is None or (pid ^ self.pid) & self.mask == 0

    def key(self):
        return (self.src, self.dst, self.pid)

    def __repr__(self):
        return describe_stream(self)


def describe_stream(s):
    text = ("packet flow " if s.pid is not None else "flow ") + (
        f"{fmt_ep(s.src)} -> {fmt_ep(s.dst)}"
    )
    return text + (f" (id {s.pid})" if s.pid is not None else "")


def program_ops(p):
    return [op for block in p["seq"] for op in block]


def cycle_start(p):
    return p.get("loop_to", 0) if p["loops"] else 0


def block_at(p, step):
    """blockAt: the blocks before the cycle once, then the cycle over and over."""
    start = cycle_start(p)
    if step < start:
        return step
    return start + (step - start) % (len(p["seq"]) - start)


def requested_streams(d):
    streams = [Stream(s, t) for s, t in d.flows]
    keeps = last_keep(d)
    for f in d.packet_flows:
        for s in f["srcs"]:
            for t in f["dsts"]:
                keep = keeps_header(t[:2], t[2:], keeps[t])
                streams.append(Stream(s, t, f["id"], keep, mask=f["mask"]))
    return streams


def sent_packet_ids(d):
    ids = defaultdict(set)
    for p in d.programs:
        if p["dir"] != MM2S:
            continue
        for op in program_ops(p):
            if op[0] == "bd" and op[2] is not None:
                ids[d.program_key(p)].add(op[2])
    for a in d.allocs.values():
        if a["pkt"] is not None:
            ids[(*a["tile"], a["dir"], a["ch"])].add(a["pkt"])
    for events in d.sequences:
        for ev in events:
            if ev[0] == "memcpy" and ev[2] is not None and ev[1] in d.allocs:
                a = d.allocs[ev[1]]
                ids[(*a["tile"], a["dir"], a["ch"])].add(ev[2])
    return {k: sorted(v) for k, v in ids.items()}


def trace_routed_streams(d):
    """StreamTracer: every stream the switchboxes and shim muxes carry."""
    boxes = {t: d.boxes[t] for t in sorted(d.boxes)}
    muxes = {t: [("connect", s, m) for s, m in d.muxes[t]] for t in sorted(d.muxes)}
    sent = sent_packet_ids(d)
    streams = []

    def input_ports(ops):
        ports = []
        for op in ops:
            p = op[1] if op[0] in ("connect", "rules") else None
            if p is not None and p not in ports:
                ports.append(p)
        return ports

    def follow(here, is_mux, out):
        if is_mux:
            if out[0] == NORTH and here in boxes:
                return ("hop", here, False, (SOUTH, out[1]))
            return ("end", (*here, *out))
        if out[0] not in DIRECTIONAL:
            return ("end", (*here, *out))
        if out[0] == SOUTH and here in muxes:
            return ("hop", here, True, (NORTH, out[1]))
        dc, dr, into = STEP[out[0]]
        nxt = (here[0] + dc, here[1] + dr)
        if nxt in boxes:
            return ("hop", nxt, False, (into, out[1]))
        return ("end", (*here, *out))

    def step(src, tile, is_mux, inp, pid, visited, path):
        key = (tile, is_mux, inp)
        if key in visited:
            return
        visited = visited | {key}
        ops = muxes[tile] if is_mux else boxes[tile]

        def nxt(out, next_id, arbiter, keep=False):
            next_path = path if is_mux else path + [(tile, inp, arbiter)]
            to = follow(tile, is_mux, out)
            if to[0] == "end":
                streams.append(Stream(src, to[1], next_id, keep, next_path))
                return
            step(src, to[1], to[2], to[3], next_id, visited, next_path)

        for op in ops:
            if op[0] == "connect" and op[1] == inp:
                nxt(op[2], pid, None)
        amsels = {op[1]: op[2] for op in ops if op[0] == "amsel"}
        for op in ops:
            if op[0] != "rules" or op[1] != inp:
                continue

            def route(rule, rule_id):
                arbiter = amsels.get(rule[2])
                for ms in ops:
                    if ms[0] == "masterset" and rule[2] in ms[2]:
                        nxt(ms[1], rule_id, arbiter, keeps_header(tile, ms[1], ms[3]))

            ids = range(PARAMS["max_id"] + 1) if pid is None else (pid,)
            for packet_id in ids:
                for rule in op[2]:
                    mask, value = rule[0], rule[1]
                    if (packet_id & mask) == (value & mask):
                        route(rule, packet_id)
                        break

    def trace_from(src, tile, is_mux, inp):
        ids = None
        if src[2] == DMA:
            ids = sent.get((src[0], src[1], MM2S, src[3]))
        if ids is None:
            step(src, tile, is_mux, inp, None, frozenset(), [])
            return
        for pid in ids:
            step(src, tile, is_mux, inp, pid, frozenset(), [])

    for tile, ops in muxes.items():
        for p in input_ports(ops):
            if p[0] != NORTH:
                trace_from((*tile, *p), tile, True, p)
    for tile, ops in boxes.items():
        for p in input_ports(ops):
            if p[0] not in DIRECTIONAL:
                trace_from((*tile, *p), tile, False, p)
    return streams


class Volumes:
    """StreamVolumeAnalysis."""

    def __init__(self, d, streams):
        self.d = d
        self.streams = streams
        self.visiting = set()
        self.programs = defaultdict(list)
        for p in d.programs:
            self.programs[d.program_key(p)].append(p)
        self.memcpys = []
        for events in d.sequences:
            for ev in events:
                if ev[0] == "memcpy":
                    self.memcpys.append(ev)

    def send_volume(self, s):
        if s.src[2] != DMA:
            return None
        key = (s.src[0], s.src[1], MM2S, s.src[3])

        def carries(pkt):
            return pkt is None or s.carries(pkt)

        def header(pkt):
            return PARAMS["header_bytes"] if pkt is not None and s.keep else 0

        known, total = False, 0
        for p in self.programs.get(key, []):
            bds = [op for op in program_ops(p) if op[0] == "bd" and carries(op[2])]
            known = True
            if not bds:
                continue
            if p["loops"]:
                looped = self.looped_volume(
                    p, lambda op: op[1] + header(op[2]) if carries(op[2]) else 0
                )
                if looped is None:
                    return None
                total += looped
                continue
            nbytes = sum(op[1] + header(op[2]) for op in bds)
            if p["kind"] == "start":
                runs = p["repeat"] + 1
            else:
                if p["dyn_repeat"]:
                    return None
                runs = 0
                for in_loop in p["users"]:
                    if in_loop:
                        return None
                    runs += p["repeat"] + 1
            total += nbytes * runs
        for _, sym, pkt, nbytes, in_loop, runs in self.memcpys:
            a = self.d.allocs.get(sym)
            if a is None or (*a["tile"], a["dir"], a["ch"]) != key:
                continue
            if pkt is None:
                pkt = a["pkt"]
            if not carries(pkt):
                continue
            if in_loop or nbytes is None:
                return None
            known = True
            total += nbytes + runs * header(pkt)
        return total if known else None

    def looped_volume(self, p, bytes_of):
        """StreamVolumeAnalysis::loopedVolume."""
        nbytes = 0

        def visit(op):
            nonlocal nbytes
            nbytes += bytes_of(op)

        return nbytes if self.walk_loop(p, visit) else None

    def walk_loop(self, p, visit):
        """StreamVolumeAnalysis::walkLoop."""
        if self.d.target.aie1 or id(p) in self.visiting or not p["seq"]:
            return False
        self.visiting.add(id(p))
        try:
            tokens, seq = {}, p["seq"]
            for step in range(PARAMS["max_bd_steps"]):
                for op in seq[block_at(p, step)]:
                    if op[0] != "lock":
                        visit(op)
                        continue
                    _, action, lock, n = op
                    if lock is None or n is None or n < 0 or action == 0:
                        return False
                    if lock not in tokens:
                        others = self.tokens_from_others(lock, p)
                        if others is None:
                            return False
                        tokens[lock] = self.d.locks[lock][3] + others
                    if action == 1:
                        tokens[lock] += n
                    elif tokens[lock] < n:
                        return True
                    else:
                        tokens[lock] -= n
            return False
        finally:
            self.visiting.discard(id(p))

    def may_send_after(self, first, then):
        """StreamVolumeAnalysis::maySendAfter."""
        if first.src != then.src:
            return None
        if (
            first.pid is None
            or then.pid is None
            or (first.pid ^ then.pid) & first.mask & then.mask == 0
        ):
            return True
        if first.src[2] != DMA:
            return None
        key = (first.src[0], first.src[1], MM2S, first.src[3])
        progs = self.programs.get(key, [])
        if len(progs) != 1 or progs[0]["kind"] != "start":
            return None
        for _, sym, _, _, _, _ in self.memcpys:
            a = self.d.allocs.get(sym)
            if a is not None and (*a["tile"], a["dir"], a["ch"]) == key:
                return None
        p = progs[0]
        sent = after = False

        def visit(op):
            nonlocal sent, after
            after |= sent and (op[2] is None or then.carries(op[2]))
            sent |= op[2] is None or first.carries(op[2])

        if p["loops"]:
            return after if self.walk_loop(p, visit) else None
        for _ in range(p["repeat"] + 1):
            if after:
                break
            for op in program_ops(p):
                if op[0] == "bd":
                    visit(op)
        return after

    def tokens_from_others(self, lock, me):
        """StreamVolumeAnalysis::tokensFromOthers."""
        total = 0
        for uses in self.d.cores.values():
            for a, l, n in uses:
                if a != 1 or l != lock:
                    continue
                if lock in self.d.core_repeated_releases or n is None or n < 0:
                    return None
                total += n
        for p in self.d.programs:
            if p is me or not any(
                op[0] == "lock" and op[1] == 1 and op[2] == lock
                for op in program_ops(p)
            ):
                continue
            n = self.releases_over(p, lock)
            if n is None:
                return None
            total += n
        return total

    def releases_over(self, p, lock):
        """StreamVolumeAnalysis::releasesOver."""
        if p["kind"] != "start":
            return None
        seq = p["seq"]

        def released(op):
            if op[0] != "lock" or op[1] != 1 or op[2] != lock:
                return 0
            return op[3]

        if not p["loops"]:
            amounts = [released(op) for op in program_ops(p)]
            if any(n is None or n < 0 for n in amounts):
                return None
            return sum(amounts) * (p["repeat"] + 1)
        if p["dir"] != S2MM or not seq:
            return None
        ep = (*p["tile"], DMA, p["ch"])
        received = 0
        for s in self.streams:
            if s.dst != ep:
                continue
            v = self.send_volume(s)
            if v is None:
                return None
            received += v
        filled = tokens = 0
        for step in range(PARAMS["max_bd_steps"]):
            for op in seq[block_at(p, step)]:
                if op[0] == "bd":
                    if filled + op[1] > received:
                        return tokens
                    filled += op[1]
                    continue
                n = released(op)
                if n is None or n < 0:
                    return None
                tokens += n
        return None

    def receive_capacity(self, ep):
        if ep[2] != DMA:
            return 0
        progs = self.programs.get((ep[0], ep[1], S2MM, ep[3]))
        if not progs:
            return 0
        caps = [c for c in map(self.program_capacity, progs) if c is not None]
        return min(caps, default=None)

    def program_capacity(self, p):
        if p["kind"] != "start":
            return 0
        passes = 1 + p["repeat"]
        seq = p["seq"]
        if not seq:
            return 0
        values = {}
        nbytes = 0
        # A cycle that leaves every lock where it found it, or with more
        # tokens, runs again the same way.
        cycle_values, cycle_bytes, start = None, 0, cycle_start(p)
        step = 0
        while True:
            if not p["loops"] and step >= passes * len(seq):
                return nbytes
            i = block_at(p, step)
            if i == start:
                if (
                    cycle_values is not None
                    and cycle_values.keys() == values.keys()
                    and all(
                        (
                            v == cycle_values[k]
                            if self.d.target.aie1
                            else v >= cycle_values[k]
                        )
                        for k, v in values.items()
                    )
                ):
                    if p["loops"]:
                        return None
                    return nbytes + (passes - step // len(seq)) * (nbytes - cycle_bytes)
                cycle_values, cycle_bytes = dict(values), nbytes
            # Past the analysis limit, what it took in so far.
            if step >= PARAMS["max_bd_steps"]:
                return nbytes
            step += 1
            for op in seq[i]:
                if op[0] == "lock":
                    _, action, lock, n = op
                    if lock is None or n is None:
                        return 0
                    v = values.setdefault(lock, self.d.locks[lock][3])
                    if self.d.target.aie1:
                        if action == 1:
                            values[lock] = n
                        elif v != n:
                            return nbytes
                    elif action == 1:
                        values[lock] = v + n
                    elif action == 0 or v < n:
                        return nbytes
                    else:
                        values[lock] = v - n
                else:
                    nbytes += op[1]

    def can_fill(self, ep, streams):
        cap = self.receive_capacity(ep)
        if cap is None:
            return False
        sent = 0
        for s in streams:
            if s.dst != ep:
                continue
            v = self.send_volume(s)
            if v is None:
                return True
            sent += v
        return sent > cap


class WaitGraph:
    """StreamWaitGraph. Agents are (col, row, kind, dir, channel)."""

    LOCK, STREAM, HOST = range(3)
    CHANNEL, CORE, CONTROLLER = range(3)

    def __init__(self, d, streams, volumes):
        self.agents, self.edges, self.ids, self.modeled = [], [], {}, set()
        # Stream edges from a sender to its receiver, and the reverse.
        self.sends, self.receives = set(), set()
        never_full = {
            (*s.dst[:2], s.dst[3])
            for s in streams
            if s.dst[2] == DMA and not volumes.can_fill(s.dst, streams)
        }

        def waits_on_locks(a):
            c, r, kind, dr, ch = self.agents[a]
            return kind == self.CORE or dr != S2MM or (c, r, ch) not in never_full

        acquirers, releasers = {}, {}

        def note(use, agent):
            _, action, lock, _ = use
            if lock is None:
                return
            m = releasers if action == 1 else acquirers
            s = m.setdefault(lock, [])
            if agent not in s:
                s.append(agent)

        for tile, uses in d.cores.items():
            agent = self.get_or_create(tile, self.CORE, MM2S, 0)
            self.modeled.add(agent)
            for u in uses:
                note(("lock", *u), agent)
        for p in d.programs:
            agent = self.get_or_create(p["tile"], self.CHANNEL, p["dir"], p["ch"])
            self.modeled.add(agent)
            for op in program_ops(p):
                if op[0] == "lock":
                    note(op, agent)
        for lock in d.locks:
            if lock not in acquirers or lock not in releasers:
                continue
            for p in acquirers[lock]:
                for q in releasers[lock]:
                    if p != q and waits_on_locks(p):
                        self.add_edge(p, q, self.LOCK)

        def endpoint_agent(ep, sending):
            if ep[2] == CORE:
                return self.get_or_create(ep[:2], self.CORE, MM2S, 0)
            if ep[2] == CTRL and sending:
                return self.get_or_create(ep[:2], self.CONTROLLER, MM2S, 0)
            if ep[2] == DMA:
                return self.get_or_create(
                    ep[:2], self.CHANNEL, MM2S if sending else S2MM, ep[3]
                )
            return None

        for s in streams:
            a = endpoint_agent(s.src, True)
            b = endpoint_agent(s.dst, False)
            if a is None or b is None or a == b:
                continue
            self.add_edge(a, b, self.STREAM)
            self.add_edge(b, a, self.STREAM)
            self.sends.add((a, b))
            self.receives.add((b, a))

        def by_symbol(sym):
            a = d.allocs.get(sym)
            return None if a is None else (*a["tile"], a["dir"], a["ch"])

        num_stream_agents = len(self.agents)
        for events in d.sequences:
            waited = []

            def agent_of(key):
                if key is None or key[2] is None:
                    return None
                return self.get_or_create(key[:2], self.CHANNEL, key[2], key[3])

            def chain_key(ev):
                return by_symbol(ev[2]) if ev[2] else ev[1]

            def keys_of(ev):
                if ev[0] == "host_alts":
                    return list(ev[2])
                if ev[0] in ("memcpy", "wait"):
                    key = by_symbol(ev[1])
                elif ev[0] in ("start", "await"):
                    key = d.program_key(d.programs[ev[1]])
                elif ev[0] == "chain":
                    key = chain_key(ev)
                elif ev[0] == "await_chain":
                    key = chain_key(events[ev[1]])
                else:
                    return []
                return [] if key is None or key[2] is None else [key]

            def issues(ev):
                return ev[0] in ("memcpy", "start", "chain") or ev[0:2] == (
                    "host_alts",
                    "start",
                )

            # walkLoopsTwice: waits later in a loop body hold its next trip.
            def replay(i, out):
                while i < len(events):
                    ev = events[i]
                    if ev[0] == "loop_end":
                        return i + 1
                    if ev[0] == "loop_begin":
                        replay(i + 1, out)
                        i = replay(i + 1, out)
                        continue
                    # A host_alts event stands for the one before it.
                    if i + 1 == len(events) or events[i + 1][0] != "host_alts":
                        out.append(ev)
                    i += 1
                return i

            ordered = []
            replay(0, ordered)
            issued = list(
                dict.fromkeys(k for ev in events if issues(ev) for k in keys_of(ev))
            )
            for ev in ordered:
                keys = keys_of(ev)
                if issues(ev):
                    for key in keys:
                        agent = agent_of(key)
                        self.modeled.add(agent)
                        for w in waited:
                            if w != agent:
                                self.add_edge(agent, w, self.HOST)
                    # An issue on a channel the model cannot see may start any.
                    if not keys:
                        for a in range(num_stream_agents):
                            if self.agents[a][2] == self.CHANNEL:
                                for w in waited:
                                    if w != a:
                                        self.add_edge(a, w, self.HOST)
                elif ev[0] in (
                    "wait",
                    "await",
                    "await_chain",
                    "await_any",
                    "host_alts",
                ):
                    # A wait on a channel the model cannot see may be on any.
                    # The host learns a channel is done from a task-complete
                    # token its column's controllers send.
                    for key in keys or issued:
                        for agent in [agent_of(key)] + [
                            a
                            for a in range(num_stream_agents)
                            if self.agents[a][2] == self.CONTROLLER
                            and self.agents[a][0] == key[0]
                        ]:
                            if agent not in waited:
                                waited.append(agent)
        # The host drives a shim channel nothing programs, and may wait on any
        # other shim tile first.
        self.on_shim = [
            kind == self.CHANNEL and d.target.kind((c, r)) == "shim"
            for c, r, kind, _, _ in self.agents
        ]
        for a in range(len(self.agents)):
            if (
                a in self.modeled
                or self.agents[a][2] == self.CONTROLLER
                or not waits_on_locks(a)
            ):
                continue
            for b in range(len(self.agents)):
                if (
                    b != a
                    and self.agents[b][2] != self.CONTROLLER
                    and (
                        self.agents[b][:2] == self.agents[a][:2]
                        or (self.on_shim[a] and self.on_shim[b])
                    )
                ):
                    self.add_edge(a, b, self.LOCK)

    def _key(self, tile, kind, dr, ch):
        tile_only = kind != self.CHANNEL
        return (tile[0], tile[1], kind, 0 if tile_only else dr, 0 if tile_only else ch)

    def get_or_create(self, tile, kind, dr, ch):
        k = self._key(tile, kind, dr, ch)
        if k not in self.ids:
            self.ids[k] = len(self.agents)
            self.agents.append(k)
            self.edges.append([])
        return self.ids[k]

    def add_edge(self, a, b, kind):
        if (b, kind) not in self.edges[a]:
            self.edges[a].append((b, kind))

    def agent_at(self, ep, sending):
        if ep[2] == CORE:
            return self.ids.get(self._key(ep[:2], self.CORE, MM2S, 0))
        if ep[2] == CTRL and sending:
            return self.ids.get(self._key(ep[:2], self.CONTROLLER, MM2S, 0))
        if ep[2] == DMA:
            return self.ids.get(
                self._key(ep[:2], self.CHANNEL, MM2S if sending else S2MM, ep[3])
            )
        return None

    def drainers_of(self, a):
        if self.agents[a][2] == self.CORE:
            return [a]
        return [b for b, kind in self.edges[a] if kind != self.STREAM]

    def wait_chain(self, frm, targets, avoid):
        # A state is an agent and whether the chain reached it as a receiver
        # taking a sender's data, where its other DMA senders are no wait of
        # its: a DMA channel sends a packet whole once it starts.
        parent = {}
        for a in avoid:
            parent[(a, False)] = None
            parent[(a, True)] = None
        work = deque()
        for a in frm:
            if (a, False) not in parent:
                parent[(a, False)] = None
                work.append((a, False))
        while work:
            state = work.popleft()
            a, taking = state
            if a in targets:
                chain = []
                at = state
                while at is not None:
                    chain.append(at[0])
                    at = parent[at]
                return chain[::-1]
            if taking and (a, False) in parent:
                continue
            on_chain = set()
            at = state
            while at is not None:
                on_chain.add(at[0])
                at = parent[at]
            for b, kind in self.edges[a]:
                if (
                    taking
                    and (a, b) in self.receives
                    and (a, b) not in self.sends
                    and self.agents[b][2] == self.CHANNEL
                ):
                    continue
                if (b, False) in parent or b in on_chain:
                    continue
                nxt = (b, kind == self.STREAM and (a, b) in self.sends)
                if nxt not in parent:
                    parent[nxt] = state
                    work.append(nxt)
        return []

    def describe(self, a):
        c, r, kind, dr, ch = self.agents[a]
        if kind == self.CORE:
            return f"({c}, {r}) core"
        if kind == self.CONTROLLER:
            return f"({c}, {r}) TileControl"
        return f"({c}, {r}) {DIRS[dr]} {ch}"


class Analysis:
    """StreamConflicts: requested streams first, then those already routed."""

    def __init__(self, d):
        self.d = d
        self.streams = requested_streams(d)
        self.num_requested = len(self.streams)
        self.streams += trace_routed_streams(d)
        self._graph = None
        self._stalls, self._blocks, self._conflicts, self._waits = {}, {}, {}, {}
        tree_ids, self.tree_members, self.tree_of = {}, [], []
        for i, s in enumerate(self.streams):
            tree = len(self.tree_members)
            if s.pid is not None:
                tree = tree_ids.setdefault((s.src, s.pid, i < self.num_requested), tree)
            if tree == len(self.tree_members):
                self.tree_members.append([])
            self.tree_members[tree].append(i)
            self.tree_of.append(tree)

    @property
    def graph(self):
        if self._graph is None:
            self.volumes = Volumes(self.d, self.streams)
            self._graph = WaitGraph(self.d, self.streams, self.volumes)
        return self._graph

    def can_stall(self, f):
        if f not in self._stalls:
            self._stalls[f] = self.volumes.can_fill(self.streams[f].dst, self.streams)
        return self._stalls[f]

    def blocking_chain(self, f, g):
        g_ = self.graph
        fs, gs = self.streams[f], self.streams[g]
        f_dst = g_.agent_at(fs.dst, False)
        if f_dst is None:
            return []
        own = []
        f_src = g_.agent_at(fs.src, True)
        if f_src is not None:
            own.append(f_src)
        own.append(f_dst)
        drainers = g_.drainers_of(f_dst)
        avoid = [a for a in own if a not in drainers]
        targets = [
            a
            for a in (g_.agent_at(gs.src, True), g_.agent_at(gs.dst, False))
            if a is not None and a not in own
        ]
        if not targets:
            return []
        return g_.wait_chain(drainers, targets, avoid)

    def can_block(self, f, g):
        if (f, g) not in self._blocks:
            self.graph
            self._blocks[(f, g)] = (
                self.can_stall(f)
                and not self.silent(f)
                and not self.silent(g)
                and self.volumes.may_send_after(self.streams[f], self.streams[g])
                is not False
                and bool(self.blocking_chain(f, g))
            )
        return self._blocks[(f, g)]

    def silent(self, f):
        self.graph
        return self.volumes.send_volume(self.streams[f]) == 0

    def assumptions(self, f, g):
        fs = self.streams[f]
        out = []
        for other in self.streams:
            if other.dst == fs.dst and self.volumes.send_volume(other) is None:
                out.append(
                    f"The volume {describe_stream(other)} carries is unknown, so it "
                    "is assumed to overrun its receiver."
                )
                break
        if (
            fs.src == self.streams[g].src
            and self.volumes.may_send_after(fs, self.streams[g]) is None
        ):
            out.append(
                f"Both come from {fmt_ep(fs.src)}, and the order it sends in is "
                "not modeled."
            )
        chain = self.blocking_chain(f, g)
        waiters = [self.graph.agent_at(fs.dst, False)] + chain[:-1]
        for i, a in enumerate(waiters):
            if a not in self.graph.modeled and a not in waiters[:i]:
                out.append(
                    f"Nothing in the design programs {self.graph.describe(a)}, so it "
                    "is assumed to wait on anything on its tile"
                    + (" or on another shim tile." if self.graph.on_shim[a] else ".")
                )
        return out

    def explain_block(self, f, g):
        fs, gs = self.streams[f], self.streams[g]
        chain = self.blocking_chain(f, g)
        s = (
            describe_stream(fs)
            + " can fill its receiver, and draining that waits on "
            + ", then ".join(self.graph.describe(a) for a in chain)
        )
        s += (
            ", which receives "
            if self.graph.agent_at(gs.dst, False) == chain[-1]
            else ", which sends "
        )
        s += describe_stream(gs) + "."
        for a in self.assumptions(f, g):
            s += " " + a
        return s

    def blocks(self, s, t):
        a, b = self.streams[s], self.streams[t]
        if a.src == b.src or a.dst == b.dst:
            return False
        return self.can_block(s, t)

    def related(self, s, t):
        if self.streams[s].src == self.streams[t].src:
            return True
        return any(
            self.streams[m].dst == self.streams[n].dst
            for m in self.tree_members[self.tree_of[s]]
            for n in self.tree_members[self.tree_of[t]]
        )

    def conflict(self, s, t):
        if (s, t) not in self._conflicts:
            self._conflicts[(s, t)] = not self.related(s, t) and (
                self.blocks(s, t) or self.blocks(t, s)
            )
        return self._conflicts[(s, t)]

    def waits_from(self, a):
        """StreamConflicts::waitsFrom."""
        if a not in self._waits:
            reached, work = {}, deque([a])
            while work:
                u = work.popleft()
                for g, members in enumerate(self.tree_members):
                    if g == a or g in reached or self.streams[members[0]].pid is None:
                        continue
                    by = next(
                        (
                            (x, m)
                            for x in self.tree_members[u]
                            for m in members
                            if self.blocks(x, m)
                        ),
                        None,
                    )
                    if by is not None:
                        reached[g] = by
                        work.append(g)
            self._waits[a] = reached
        return self._waits[a]

    def must_separate(self, s, t):
        """StreamConflicts::mustSeparate."""
        if self.related(s, t):
            return False
        return (
            self.blocks(s, t)
            or self.blocks(t, s)
            or self.tree_of[t] in self.waits_from(self.tree_of[s])
            or self.tree_of[s] in self.waits_from(self.tree_of[t])
        )

    def unavoidable(self):
        """StreamConflicts::unavoidable."""
        n = self.num_requested
        return [
            (s, t)
            for s in range(n)
            for t in range(n)
            if s != t
            and (self.streams[s].pid is not None or self.streams[t].pid is not None)
            and self.related(s, t)
            and self.can_block(s, t)
            and not self.assumptions(s, t)
        ]

    def explain(self, s, t):
        self.graph
        if self.can_block(s, t):
            return self.explain_block(s, t)
        if self.can_block(t, s):
            return self.explain_block(t, s)
        if self.tree_of[t] not in self.waits_from(self.tree_of[s]):
            s, t = t, s
        steps, g = [], self.tree_of[t]
        while g != self.tree_of[s]:
            x, m = self.waits_from(self.tree_of[s])[g]
            steps.append(self.explain_block(x, m))
            g = self.tree_of[x]
        return " ".join(reversed(steps))

    def hold_cycle(self, routes, forced_waits=False, definite=False):
        """StreamConflicts::holdCycle. routes[i] is [(tile, input, arbiter)]
        for streams[i]. Steps are (wait, waiting, sharer, holding, tile,
        sharer input, holder input, arbiter, forced)."""
        streams, nreq = self.streams, self.num_requested
        trees, tree_ids = [], {}
        for i, s in enumerate(streams):
            if s.pid is None or self.silent(i):
                continue
            k = (s.src[:2], s.src[2:], s.pid, i < nreq)
            if k not in tree_ids:
                tree_ids[k] = len(trees)
                trees.append(dict(members=[], hops=[], parent=[], arbiter=[], base=0))
            tree = trees[tree_ids[k]]
            tree["members"].append(i)
            prev = -1
            for tile, inp, arb in routes[i]:
                key = (tile, inp)
                if key in tree["hops"]:
                    h = tree["hops"].index(key)
                else:
                    h = len(tree["hops"])
                    tree["hops"].append(key)
                    tree["parent"].append(prev)
                    tree["arbiter"].append(arb)
                prev = h

        # One source sends one tree's packet at a time. Trees into one receiver
        # are both in flight at once, and wait on each other at its master port
        # however they are routed, where both enter its switchbox for it; and
        # trees with one id wait on each other where they meet and below, going
        # on as one. Those waits count only with `forced_waits`.
        def related(a, b):
            return (
                streams[trees[a]["members"][0]].src
                == streams[trees[b]["members"][0]].src
            )

        def forced(a, ha, b, hb, tile):
            ina, inb = trees[a]["hops"][ha][1], trees[b]["hops"][hb][1]

            def into(m, inp):
                return (
                    streams[m].dst[:2] == tile
                    and bool(routes[m])
                    and routes[m][-1][:2] == (tile, inp)
                )

            if any(
                streams[m].dst == streams[n].dst and into(m, ina) and into(n, inb)
                for m in trees[a]["members"]
                for n in trees[b]["members"]
            ):
                return True
            if (
                streams[trees[a]["members"][0]].pid
                != streams[trees[b]["members"][0]].pid
            ):
                return False
            return any(
                parent == ha
                and any(
                    pb == hb and trees[b]["hops"][k] == trees[a]["hops"][h]
                    for k, pb in enumerate(trees[b]["parent"])
                )
                for h, parent in enumerate(trees[a]["parent"])
            )

        nodes = []
        for t, tree in enumerate(trees):
            tree["base"] = len(nodes)
            nodes += [(t, "hop", h) for h in range(len(tree["hops"]))]
            nodes += [(t, "recv", m) for m in range(len(tree["members"]))]
            nodes.append((t, "any", 0))
            nodes += [(t, "past", h) for h in range(len(tree["hops"]))]

        def receiver_node(t, m):
            return trees[t]["base"] + len(trees[t]["hops"]) + m

        def anywhere_node(t):
            return receiver_node(t, len(trees[t]["members"]))

        def past_node(t, h):
            return anywhere_node(t) + 1 + h

        entering, passing = defaultdict(list), defaultdict(list)
        for t, tree in enumerate(trees):
            for h, hop in enumerate(tree["hops"]):
                entering[hop].append((t, h))
                passing[hop[0]].append((t, h))

        # The arbiter both trees took to leave the switchbox above a hop they
        # share where they came into it apart.
        def merged_at(a, ha, b, hb):
            while True:
                pa, pb = trees[a]["parent"][ha], trees[b]["parent"][hb]
                if pa < 0 or pb < 0:
                    return None
                if trees[a]["hops"][pa] != trees[b]["hops"][pb]:
                    arbiter = trees[a]["arbiter"][pa]
                    if (
                        trees[a]["hops"][pa][0] != trees[b]["hops"][pb][0]
                        or arbiter is None
                        or arbiter != trees[b]["arbiter"][pb]
                    ):
                        return None
                    return (trees[a]["hops"][pa][0], arbiter)
                ha, hb = pa, pb

        cache = {}

        def successors(n):
            if n in cache:
                return cache[n]
            out = []
            t0, kind, index = nodes[n]
            u = trees[t0]
            if kind == "hop":
                tile, inp = u["hops"][index]
                waiting = u["members"][0]
                for t, ht in entering[(tile, inp)]:
                    if t != t0 and related(t0, t):
                        continue
                    sharer = trees[t]["members"][0]
                    same_id = streams[waiting].pid == streams[sharer].pid
                    if t != t0 and (not same_id or forced_waits):
                        out.append(
                            (
                                past_node(t, ht),
                                (
                                    "link",
                                    waiting,
                                    sharer,
                                    sharer,
                                    tile,
                                    inp,
                                    inp,
                                    -1,
                                    same_id,
                                ),
                                merged_at(t0, index, t, ht),
                            )
                        )
                    arbiter = trees[t]["arbiter"][ht]
                    if arbiter is None:
                        continue
                    # A tree's branches move as one, so where two come into a
                    # switchbox apart onto one arbiter, the one holding it
                    # waits on the other.
                    for v, hv in passing[tile]:
                        holder_input = trees[v]["hops"][hv][1]
                        own = v == t0 and t == t0
                        if (
                            (not own and (v == t0 or v == t or related(t, v)))
                            or holder_input == inp
                            or trees[v]["arbiter"][hv] != arbiter
                        ):
                            continue
                        is_forced = not own and forced(t, ht, v, hv, tile)
                        if is_forced and not forced_waits:
                            continue
                        out.append(
                            (
                                past_node(v, hv),
                                (
                                    "arbiter",
                                    waiting,
                                    sharer,
                                    trees[v]["members"][0],
                                    tile,
                                    inp,
                                    holder_input,
                                    arbiter,
                                    is_forced,
                                ),
                                None,
                            )
                        )
            elif kind == "recv":
                s = u["members"][index]
                for g in range(len(trees)):
                    if g == t0 or any(
                        streams[m].dst == streams[s].dst for m in trees[g]["members"]
                    ):
                        continue
                    for m in trees[g]["members"]:
                        if self.blocks(s, m) and not (
                            definite and self.assumptions(s, m)
                        ):
                            out.append(
                                (
                                    anywhere_node(g),
                                    ("drain", s, s, m, None, None, None, -1, False),
                                    None,
                                )
                            )
                            break
            else:
                behind = set()
                if kind == "past":
                    h = index
                    while h >= 0:
                        behind.add(h)
                        h = u["parent"][h]
                for h in range(len(u["hops"])):
                    if h not in behind:
                        out.append((u["base"] + h, None, None))
                for m in range(len(u["members"])):
                    out.append((receiver_node(t0, m), None, None))
            cache[n] = out
            return out

        def counts(e):
            st = e[1]
            if st is None or st[0] == "drain":
                return False
            return st[1] < nreq or st[2] < nreq or st[3] < nreq

        roots = [
            n
            for n in range(len(nodes))
            if nodes[n][1] == "hop" and any(counts(e) for e in successors(n))
        ]
        index = [-1] * len(nodes)
        low = [0] * len(nodes)
        comp = [-1] * len(nodes)
        on_stack = [False] * len(nodes)
        stack, frames = [], []
        counter = [0]
        ncomp = 0

        def visit(n):
            index[n] = low[n] = counter[0]
            counter[0] += 1
            stack.append(n)
            on_stack[n] = True
            frames.append([n, 0])

        for root in roots:
            if index[root] >= 0:
                continue
            visit(root)
            while frames:
                n, nx = frames[-1]
                out = successors(n)
                if nx < len(out):
                    frames[-1][1] += 1
                    w = out[nx][0]
                    if index[w] < 0:
                        visit(w)
                    elif on_stack[w]:
                        low[n] = min(low[n], index[w])
                    continue
                frames.pop()
                if frames:
                    low[frames[-1][0]] = min(low[frames[-1][0]], low[n])
                if low[n] != index[n]:
                    continue
                while True:
                    w = stack.pop()
                    on_stack[w] = False
                    comp[w] = ncomp
                    if w == n:
                        break
                ncomp += 1

        def allowed(edge, fixed, excluded):
            st = edge[1]
            if st is None or st[0] != "arbiter":
                return True
            grant = (st[4], st[7])
            if grant in fixed:
                return fixed[grant] == st[3]
            return st[3] not in excluded.get(grant, ())

        # A packet holds its arbiter until its tail passes, so no state has two
        # trees holding one arbiter. Nor is a walk where each packet is behind
        # the next on links below one arbiter they merged at, which granted
        # them those links in one order, a deadlock, so a walk from a link wait
        # below one must also wait otherwise.
        def close_walk(x, e, fixed, excluded):
            if not allowed(e, fixed, excluded):
                return None

            # A state is a node and whether the walk there has waited other
            # than on a link below e's merge.
            def ordered(edge):
                return edge[1] is None or edge[2] == e[2]

            start, end = (e[0], e[2] is None), (x, True)
            via = {start: (start, None)}
            work = deque([start])
            while end not in via:
                if not work:
                    return None
                n = work.popleft()
                for nxt in successors(n[0]):
                    to = (nxt[0], n[1] or not ordered(nxt))
                    if (
                        comp[nxt[0]] == comp[x]
                        and allowed(nxt, fixed, excluded)
                        and to not in via
                    ):
                        via[to] = (n, nxt)
                        work.append(to)
            path = []
            n = end
            while n != start:
                path.append(via[n][1])
                n = via[n][0]
            path.append(e)
            return list(reversed(path))

        def steps_of(path):
            return [edge[1] for edge in path if edge[1] is not None]

        for x in roots:
            for e in successors(x):
                if not counts(e) or comp[e[0]] != comp[x]:
                    continue
                pending, first, search = [({}, {})], None, 0
                while pending:
                    if search == PARAMS["walk_searches"]:
                        return steps_of(first)
                    search += 1
                    fixed, excluded = pending.pop()
                    path = close_walk(x, e, fixed, excluded)
                    if path is None:
                        continue
                    first = first or path
                    held, clash = {}, None
                    for st in steps_of(path):
                        if st[0] != "arbiter":
                            continue
                        grant = (st[4], st[7])
                        if held.setdefault(grant, st[3]) != st[3]:
                            clash = (grant, held[grant])
                            break
                    if clash is None:
                        return steps_of(path)
                    grant, holder = clash
                    pending.append(
                        (
                            fixed,
                            {**excluded, grant: excluded.get(grant, ()) + (holder,)},
                        )
                    )
                    pending.append(({**fixed, grant: holder}, excluded))
        return None

    def explain_cycle(self, steps):
        out = []
        for wait, waiting, sharer, holding, tile, sin, hin, arb, _ in steps:
            if wait == "link":
                out.append(
                    f"{describe_stream(self.streams[waiting])} can queue behind "
                    f"{describe_stream(self.streams[holding])} on {fmt_port(sin)} into "
                    f"tile ({tile[0]}, {tile[1]})."
                )
            elif wait == "arbiter":
                s = (
                    f"{describe_stream(self.streams[holding])} can hold arbiter {arb} at "
                    f"tile ({tile[0]}, {tile[1]}) that "
                    f"{describe_stream(self.streams[sharer])} needs"
                )
                if waiting != sharer:
                    s += f", and {describe_stream(self.streams[waiting])} can queue behind it"
                out.append(s + ".")
            else:
                self.graph
                out.append(self.explain_block(waiting, holding))
        return " ".join(out)
