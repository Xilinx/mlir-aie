#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""A design: tiles, flows, programs and runtime sequence, built in Python
or loaded from MLIR, and emitted back as MLIR."""

import re
import threading

from aie.ir import Context, Location, Module
import aie.dialects.aie  # noqa: F401
import aie.dialects.aiex  # noqa: F401

from .params import DEVICE_IDS
from .fabric import ACTIONS, BUNDLES, DIRS, DMA, LOOP_OPS, SOUTH, Target


class Design:
    """Everything in a device the router and its deadlock rules read.

    Ports are (bundle, channel) and endpoints (col, row, bundle, channel),
    bundles as WireBundle numbers so tuples order as the C++ Port and TileID
    do. A DMA program's `seq` is its BD chain as the analysis walks it: for
    aie.dma_start the blocks from its first BD, including the terminal block
    when the chain ends; its ops are ("lock", action, lock, amount) and
    ("bd", bytes, packet id).
    """

    def __init__(self, dev):
        self.dev = dev
        self.tiles = []
        self.locks = {}  # name -> (col, row, id, init)
        self.programs = []  # document order
        self.cores = {}  # tile -> [(action, lock, amount)]
        self.core_repeated_releases = set()  # locks a core can release twice
        self.boxes = {}  # tile -> [op]
        self.muxes = {}  # tile -> [(src port, dst port)]
        self.flows = []  # (src, dst)
        self.packet_flows = []
        self.allocs = {}  # symbol -> dict(tile, dir, ch, pkt)
        self.sequences = []  # [[event]]
        self.unsupported = []
        self.reload = False  # has_ctrl_pkt_overlay: reloaded by control packets

    @property
    def target(self):
        return Target(self.dev)

    def copy(self):
        d = Design(self.dev)
        d.tiles = list(self.tiles)
        d.locks = dict(self.locks)
        d.programs = [
            dict(p, seq=[list(b) for b in p["seq"]], users=list(p.get("users", [])))
            for p in self.programs
        ]
        d.cores = {k: list(v) for k, v in self.cores.items()}
        d.core_repeated_releases = set(self.core_repeated_releases)
        d.boxes = {k: [tuple(o) for o in v] for k, v in self.boxes.items()}
        d.muxes = {k: list(v) for k, v in self.muxes.items()}
        d.flows = list(self.flows)
        d.packet_flows = [
            dict(f, srcs=list(f["srcs"]), dsts=list(f["dsts"]))
            for f in self.packet_flows
        ]
        d.allocs = {k: dict(v) for k, v in self.allocs.items()}
        d.sequences = [list(s) for s in self.sequences]
        d.unsupported = list(self.unsupported)
        d.reload = self.reload
        return d

    def add_packet_flow(self, pid, srcs, dsts, keep=None, priority=None, mask=None):
        self.packet_flows.append(
            dict(
                id=pid,
                mask=mask,
                keep=keep,
                priority=priority,
                srcs=list(srcs),
                dsts=list(dsts),
            )
        )

    def lock(self, tile, init):
        n = sum(1 for v in self.locks.values() if v[:2] == tile)
        name = f"l_{tile[0]}_{tile[1]}_{n}"
        self.locks[name] = (tile[0], tile[1], n, init)
        return name

    def program_key(self, p):
        return (*p["tile"], p["dir"], p["ch"])

    def used_tiles(self):
        tiles = set(self.tiles)
        for s, d in self.flows:
            tiles |= {s[:2], d[:2]}
        for f in self.packet_flows:
            tiles |= {e[:2] for e in f["srcs"] + f["dsts"]}
        tiles |= {p["tile"] for p in self.programs}
        tiles |= set(self.cores) | set(self.boxes) | set(self.muxes)
        tiles |= {v[:2] for v in self.locks.values()}
        tiles |= {a["tile"] for a in self.allocs.values()}
        return sorted(tiles)

    def emit(self, tag=None):
        out = [
            "module {" if tag is None else f"module @s{tag} {{",
            f"  aie.device({self.dev}) {{",
        ]
        for c, r in self.used_tiles():
            out.append(f"    %t_{c}_{r} = aie.tile({c}, {r})")
        for name, (c, r, lid, init) in self.locks.items():
            out.append(
                f"    %{name} = aie.lock(%t_{c}_{r}, {lid}) "
                f'{{init = {init} : i32, sym_name = "{name}"}}'
            )
        nbuf = [0]

        def buffer(tile, nbytes):
            name = f"b_{tile[0]}_{tile[1]}_{nbuf[0]}"
            nbuf[0] += 1
            elem, n = ("i32", nbytes // 4) if nbytes % 4 == 0 else ("i8", nbytes)
            op = (
                "aie.external_buffer"
                if self.target.kind(tile) == "shim"
                else f"aie.buffer(%t_{tile[0]}_{tile[1]})"
            )
            out.append(
                f'    %{name} = {op} {{sym_name = "{name}"}} : memref<{n}x{elem}>'
            )
            return f"%{name} : memref<{n}x{elem}> offset = 0 len = {n}"

        def lock_ops(ops, indent, consts):
            lines = []
            for op in ops:
                if op[0] == "lock":
                    _, action, lock, n = op
                    consts.add(n)
                    lines.append(
                        f"{indent}aie.use_lock(%{lock}, {ACTIONS[action]}, %c{n})"
                    )
            return lines

        def pkt_attr(pkt, bd_id=None):
            attrs = []
            if bd_id is not None:
                attrs.append(f"bd_id = {bd_id} : i32")
            if pkt is not None:
                attrs.append(f"packet = #aie.packet_info<pkt_type = 0, pkt_id = {pkt}>")
            return f" {{{', '.join(attrs)}}}" if attrs else ""

        by_tile = {}
        for p in self.programs:
            if p["kind"] == "start":
                by_tile.setdefault(p["tile"], []).append(p)
        for tile in by_tile:
            progs = by_tile[tile]
            body, consts = [], set()
            for n, p in enumerate(progs):
                blocks = p["seq"] if p["loops"] else p["seq"][:-1]
                head = f"^p{n}b0" if blocks else "^end"
                nxt = f"^p{n + 1}" if n + 1 < len(progs) else "^end"
                if n:
                    body.append(f"    ^p{n}:")
                rep = f", repeat_count = {p['repeat']}" if p["repeat"] else ""
                body.append(
                    f"      %d{n} = aie.dma_start({DIRS[p['dir']]}, {p['ch']}, "
                    f"{head}, {nxt}{rep})"
                )
                for i, ops in enumerate(blocks):
                    body.append(f"    ^p{n}b{i}:")
                    for op in ops:
                        if op[0] == "lock":
                            body += lock_ops([op], "      ", consts)
                        else:
                            body.append(
                                f"      aie.dma_bd({buffer(tile, op[1])})"
                                + pkt_attr(op[2])
                            )
                    if i + 1 < len(blocks):
                        body.append(f"      aie.next_bd ^p{n}b{i + 1}")
                    elif p["loops"]:
                        body.append(f"      aie.next_bd ^p{n}b{p.get('loop_to', 0)}")
                    else:
                        body.append("      aie.next_bd ^end")
            op = {"mem": "aie.memtile_dma", "shim": "aie.shim_dma"}.get(
                self.target.kind(tile), "aie.mem"
            )
            out.append(
                f"    %dma_{tile[0]}_{tile[1]} = {op}(%t_{tile[0]}_{tile[1]}) {{"
            )
            out += [f"      %c{n} = arith.constant {n} : i32" for n in sorted(consts)]
            out += body
            out += ["    ^end:", "      aie.end", "    }"]
        for (c, r), uses in self.cores.items():
            out.append(f"    %core_{c}_{r} = aie.core(%t_{c}_{r}) {{")
            consts = {n for _, _, n in uses}
            out += [f"      %c{n} = arith.constant {n} : i32" for n in sorted(consts)]
            for action, lock, n in uses:
                out.append(f"      aie.use_lock(%{lock}, {ACTIONS[action]}, %c{n})")
            out += ["      aie.end", "    }"]
        for (c, r), ops in sorted(self.boxes.items()):
            out.append(f"    %sb_{c}_{r} = aie.switchbox(%t_{c}_{r}) {{")
            for op in ops:
                if op[0] == "connect":
                    (sb, si), (db, di) = op[1], op[2]
                    out.append(
                        f"      aie.connect<{BUNDLES[sb]} : {si}, {BUNDLES[db]} : {di}>"
                    )
                elif op[0] == "amsel":
                    out.append(f"      %{op[1]} = aie.amsel<{op[2]}> ({op[3]})")
                elif op[0] == "masterset":
                    _, port, names, keep, ctrl = op
                    attrs = []
                    if keep is not None:
                        attrs.append(f"keep_pkt_header = {str(keep).lower()}")
                    if ctrl:
                        attrs.append("is_ctrl_pkt_overlay")
                    out.append(
                        f"      %m_{BUNDLES[port[0]]}_{port[1]} = aie.masterset("
                        f"{BUNDLES[port[0]]} : {port[1]}, "
                        + ", ".join(f"%{n}" for n in names)
                        + ")"
                        + (f" {{{', '.join(attrs)}}}" if attrs else "")
                    )
                else:
                    _, port, rules, ctrl, *tagged = op
                    tagged = tagged[0] if tagged else [False] * len(rules)
                    out.append(
                        f"      aie.packet_rules({BUNDLES[port[0]]} : {port[1]}) {{"
                    )
                    for (mask, value, name), tag in zip(rules, tagged):
                        out.append(
                            f"        aie.rule({mask}, {value}, %{name})"
                            + (" {is_ctrl_pkt_overlay}" if tag else "")
                        )
                    out.append("      }" + (" {is_ctrl_pkt_overlay}" if ctrl else ""))
            out.append("    }")
        for (c, r), conns in sorted(self.muxes.items()):
            out.append(f"    %mux_{c}_{r} = aie.shim_mux(%t_{c}_{r}) {{")
            for (sb, si), (db, di) in conns:
                out.append(
                    f"      aie.connect<{BUNDLES[sb]} : {si}, {BUNDLES[db]} : {di}>"
                )
            out.append("    }")
        for s, d in self.flows:
            out.append(
                f"    aie.flow(%t_{s[0]}_{s[1]}, {BUNDLES[s[2]]} : {s[3]}, "
                f"%t_{d[0]}_{d[1]}, {BUNDLES[d[2]]} : {d[3]})"
            )
        for f in self.packet_flows:
            ports = [
                f"aie.packet_source<%t_{s[0]}_{s[1]}, {BUNDLES[s[2]]} : {s[3]}>"
                for s in f["srcs"]
            ] + [
                f"aie.packet_dest<%t_{d[0]}_{d[1]}, {BUNDLES[d[2]]} : {d[3]}>"
                for d in f["dsts"]
            ]
            attrs = [
                f"{k} = {str(f[a]).lower()}"
                for k, a in (
                    ("keep_pkt_header", "keep"),
                    ("priority_route", "priority"),
                )
                if f[a] is not None
            ]
            head = f"{f['id']}" + (
                f", mask = {f['mask']}" if f["mask"] is not None else ""
            )
            out.append(
                f"    aie.packet_flow({head}) {{ {' '.join(ports)} }}"
                + (f" {{{', '.join(attrs)}}}" if attrs else "")
            )
        for sym, a in self.allocs.items():
            c, r = a["tile"]
            pkt = (
                f", <pkt_id = {a['pkt']}, pkt_type = 0>" if a["pkt"] is not None else ""
            )
            out.append(
                f"    aie.shim_dma_allocation @{sym}(%t_{c}_{r}, "
                f"{DIRS[a['dir']]}, {a['ch']}{pkt})"
            )
        if any(ev[0] == "chain" for events in self.sequences for ev in events):
            out += [
                "    aie.bd_chain @chain(%x: memref<16xi32>) {",
                "      aie.dma_bd(%x : memref<16xi32> offset = 0 len = 16)",
                "      aie.end",
                "    }",
            ]
        tasks = [p for p in self.programs if p["kind"] in ("task", "task_for")]
        for k, events in enumerate(self.sequences):
            args, body = [], []
            ntask = {}
            for p in tasks:
                if p.get("sequence", 0) != k:
                    continue
                t = len(ntask)
                ntask[id(p)] = t
                if p["kind"] == "task":
                    head = (
                        f"aiex.dma_configure_task(%t_{p['tile'][0]}_{p['tile'][1]}, "
                        f"{DIRS[p['dir']]}, {p['ch']})"
                    )
                else:
                    head = f"aiex.dma_configure_task_for @{p['alloc']}"
                body.append(f"      %task{t} = {head} {{")
                blocks = p["seq"]
                for i, ops in enumerate(blocks):
                    if i:
                        body.append(f"      ^bb{i}:")
                    for op in ops:
                        if op[0] == "bd":
                            a = len(args)
                            n = op[1] // 4
                            args.append(f"%a{a}: memref<{n}xi32>")
                            body.append(
                                f"        aie.dma_bd(%a{a} : memref<{n}xi32> offset = 0 "
                                f"len = {n})" + pkt_attr(op[2], i)
                            )
                    body.append(
                        f"        aie.next_bd ^bb{i + 1}"
                        if i + 1 < len(blocks)
                        else "        aie.end"
                    )
                attrs = ["issue_token = true"]
                if p["repeat"]:
                    attrs.append(f"repeat_count = {p['repeat']} : i32")
                body.append(f"      }} {{{', '.join(attrs)}}}")
            loops = 0
            by_prog = {i: p for i, p in enumerate(self.programs)}
            for ei, ev in enumerate(events):
                indent = "      "
                lines = []
                if ev[0] == "memcpy":
                    _, sym, pkt, nbytes, in_loop, runs = ev
                    a = len(args)
                    n = nbytes // 4
                    run = n // runs
                    args.append(f"%a{a}: memref<{n}xi32>")
                    pk = (
                        f", packet = <pkt_id = {pkt}, pkt_type = 0>"
                        if pkt is not None
                        else ""
                    )
                    lines.append(
                        f"aiex.npu.dma_memcpy_nd(%a{a}[0, 0, 0, 0][{runs}, 1, 1, {run}]"
                        f"[{run}, 0, 0, 1]{pk}) {{ metadata = @{sym}, id = {a} : i64, "
                        f"issue_token = true }} : memref<{n}xi32>"
                    )
                elif ev[0] == "start":
                    in_loop = ev[2]
                    lines.append(
                        f"aiex.dma_start_task(%task{ntask[id(by_prog[ev[1]])]})"
                    )
                elif ev[0] == "await":
                    in_loop = False
                    lines.append(
                        f"aiex.dma_await_task(%task{ntask[id(by_prog[ev[1]])]})"
                    )
                elif ev[0] == "wait":
                    in_loop = False
                    lines.append(f"aiex.npu.dma_wait {{symbol = @{ev[1]}}}")
                elif ev[0] == "chain":
                    _, key, alloc, in_loop = ev
                    a = len(args)
                    args.append(f"%a{a}: memref<16xi32>")
                    on = (
                        f"for @{alloc}"
                        if alloc
                        else f"on (%t_{key[0]}_{key[1]}, {DIRS[key[2]]}, {key[3]})"
                    )
                    lines.append(
                        f"%chain{ei} = aiex.dma_start_bd_chain{'_for' if alloc else ''} "
                        f"@chain(%a{a}) : "
                        f"(memref<16xi32>) {on}"
                    )
                elif ev[0] == "await_chain":
                    in_loop = False
                    lines.append(f"aiex.dma_await_task(%chain{ev[1]})")
                else:
                    continue
                if in_loop:
                    body += [
                        f"{indent}%lb{loops} = arith.constant 0 : index",
                        f"{indent}%ub{loops} = arith.constant 2 : index",
                        f"{indent}%st{loops} = arith.constant 1 : index",
                        f"{indent}scf.for %i{loops} = %lb{loops} to %ub{loops} "
                        f"step %st{loops} {{",
                    ]
                    body += [f"{indent}  {l}" for l in lines]
                    body.append(f"{indent}}}")
                    loops += 1
                else:
                    body += [f"{indent}{l}" for l in lines]
            out.append(f"    aie.runtime_sequence @seq{k}({', '.join(args)}) {{")
            out += body
            out.append("    }")
        out += ["  }" + (" {has_ctrl_pkt_overlay = true}" if self.reload else "")]
        out += ["}", ""]
        return "\n".join(out)


def _int(attr):
    s = str(attr)
    if s in ("true", "false"):
        return s == "true"
    return int(s.split(":")[0].strip())


def _pkt(attr):
    m = re.search(r"pkt_id = (\d+)", str(attr))
    return int(m.group(1)) if m else None


def _elem_bits(type_str):
    m = re.search(r"x(i|f|bf|ui|si)(\d+)>", type_str)
    if m:
        return int(m.group(2))
    return None


def _num_elems(type_str):
    m = re.match(r"memref<([\dx]+)x[a-z]", type_str)
    if not m:
        return None
    n = 1
    for d in m.group(1).split("x"):
        n *= int(d)
    return n


def _attrs(op):
    return {
        op.attributes[i].name: op.attributes[i].attr for i in range(len(op.attributes))
    }


def _ops(block):
    return [x.operation for x in block.operations]


_contexts = threading.local()


def _context():
    if not hasattr(_contexts, "ctx"):
        _contexts.ctx = Context()
    return _contexts.ctx


def load_design(text):
    """Parse MLIR text (one aie.device) into a Design, runtime sequence
    included. Ops the model does not read are listed in design.unsupported."""
    with _context(), Location.unknown():
        module = Module.parse(text)
        device = None
        for op in module.body.operations:
            op = op.operation
            if op.name == "aie.device":
                device = op
                break
        if device is None:
            raise ValueError("no aie.device")
        attrs = _attrs(device)
        d = Design(DEVICE_IDS[_int(attrs["device"])])
        tiles, locks, amsels_of = {}, {}, {}
        consts = {}
        results = {}

        def value_key(v):
            return hash(v)

        def const_of(v):
            return consts.get(value_key(v))

        def tile_of(v):
            return tiles.get(value_key(v))

        def lock_ops(op, out):
            if op.name == "aie.use_lock":
                a = _attrs(op)
                lock = locks.get(value_key(op.operands[0]))
                n = const_of(op.operands[1]) if len(op.operands) > 1 else None
                if n is None and "value" in a:
                    n = _int(a["value"])
                out.append(("lock", _int(a["action"]), lock, n))
            elif op.name == "aie.dma_bd":
                a = _attrs(op)
                ty = str(op.operands[0].type)
                bits = _elem_bits(ty) or 32
                if "static_len" in a:
                    n = _int(a["static_len"])
                else:
                    n = _num_elems(ty) or 0
                pkt = _pkt(a["packet"]) if "packet" in a else None
                out.append(("bd", n * max(bits // 8, 1), pkt))
            elif op.name == "arith.constant":
                try:
                    consts[value_key(op.results[0])] = _int(_attrs(op)["value"])
                except (ValueError, KeyError):
                    pass

        def walk_ops(op, out, repeated=None, once=True):
            start = len(out)
            lock_ops(op, out)
            if repeated is not None and not once:
                repeated.update(u[2] for u in out[start:] if u[:2] == ("lock", 1))
            for region in op.regions:
                blocks = list(region.blocks)
                inner = once and len(blocks) == 1 and op.name not in LOOP_OPS
                for b in blocks:
                    for x in _ops(b):
                        walk_ops(x, out, repeated, inner)

        def dma_programs(op, tile):
            blocks = list(op.regions[0].blocks)
            for b in blocks:
                for x in _ops(b):
                    if x.name == "arith.constant":
                        lock_ops(x, [])
            for b in blocks:
                for x in _ops(b):
                    if x.name == "aie.dma_start":
                        a = _attrs(x)
                        seq, seen, loops, loop_to = [], [], False, 0
                        blk = list(x.successors)[0]
                        while blk is not None:
                            idx = blocks.index(blk)
                            if idx in seen:
                                loops, loop_to = True, seen.index(idx)
                                break
                            seen.append(idx)
                            ops = []
                            nxt = None
                            for y in _ops(blocks[idx]):
                                lock_ops(y, ops)
                                if y.name == "aie.next_bd":
                                    nxt = list(y.successors)[0]
                            seq.append(ops)
                            blk = nxt
                        d.programs.append(
                            dict(
                                tile=tile,
                                kind="start",
                                dir=_int(a["channel_dir"]),
                                ch=_int(a["channel"]),
                                seq=seq,
                                loops=loops,
                                loop_to=loop_to,
                                repeat=(
                                    _int(a["repeat_count"])
                                    if "repeat_count" in a
                                    else 0
                                ),
                                dyn_repeat=False,
                                users=[],
                            )
                        )
                    elif x.name == "aie.dma":
                        d.unsupported.append(x.name)

        def box_ops(op):
            out = []
            names = {}
            for x in _ops(op.regions[0].blocks[0]):
                a = _attrs(x)
                if x.name == "aie.connect":
                    out.append(
                        (
                            "connect",
                            (_int(a["source_bundle"]), _int(a["source_channel"])),
                            (_int(a["dest_bundle"]), _int(a["dest_channel"])),
                        )
                    )
                elif x.name == "aie.amsel":
                    name = f"a{_int(a['arbiterID'])}_{_int(a['msel'])}_{len(amsels_of)}"
                    amsels_of[name] = None
                    names[value_key(x.results[0])] = name
                    out.append(("amsel", name, _int(a["arbiterID"]), _int(a["msel"])))
                elif x.name == "aie.masterset":
                    out.append(
                        (
                            "masterset",
                            (_int(a["dest_bundle"]), _int(a["dest_channel"])),
                            [names.get(value_key(v)) for v in x.operands],
                            (
                                _int(a["keep_pkt_header"])
                                if "keep_pkt_header" in a
                                else None
                            ),
                            "is_ctrl_pkt_overlay" in a,
                        )
                    )
                elif x.name == "aie.packet_rules":
                    rules, tagged = [], []
                    for y in _ops(x.regions[0].blocks[0]):
                        if y.name == "aie.rule":
                            ya = _attrs(y)
                            rules.append(
                                (
                                    _int(ya["mask"]),
                                    _int(ya["value"]),
                                    names.get(value_key(y.operands[0])),
                                )
                            )
                            tagged.append("is_ctrl_pkt_overlay" in ya)
                    out.append(
                        (
                            "rules",
                            (_int(a["source_bundle"]), _int(a["source_channel"])),
                            rules,
                            "is_ctrl_pkt_overlay" in a,
                            tagged,
                        )
                    )
                elif x.name != "aie.end":
                    d.unsupported.append(x.name)
            return out

        def sequence(op):
            events = []
            k = len(d.sequences)

            # Every task a value may hold, None for one the model cannot
            # see (StreamWaitGraph's taskCreators).
            creators = {}

            def forward(src, dst):
                p = results.get(value_key(src))
                if p is not None:
                    results.setdefault(value_key(dst), p)
                ps = creators.setdefault(value_key(dst), [])
                for q in creators.get(
                    value_key(src), [p if isinstance(p, int) else None]
                ):
                    if q not in ps:
                        ps.append(q)

            def forward_yields(x):
                for region in x.regions:
                    for b in region.blocks:
                        ops = _ops(b)
                        if ops and ops[-1].name == "scf.yield":
                            for v, r in zip(ops[-1].operands, x.results):
                                forward(v, r)

            def walk(x, in_loop):
                a = _attrs(x)
                if x.name not in (
                    "aiex.dma_start_task",
                    "aiex.dma_await_task",
                    "aiex.dma_free_task",
                ):
                    for v in x.operands:
                        p = results.get(value_key(v))
                        if isinstance(p, int):
                            # It may reach starts that are not counted.
                            d.programs[p]["users"].append(True)
                if x.name == "scf.for":
                    body = x.regions[0].blocks[0]
                    for i, init in enumerate(list(x.operands)[3:]):
                        forward(init, body.arguments[i + 1])
                        forward(init, x.results[i])
                if x.name == "aiex.npu.dma_memcpy_nd":
                    sym = str(a["metadata"]).lstrip("@")
                    ty = str(x.operands[0].type)
                    bits = _elem_bits(ty)
                    sizes = [
                        int(v)
                        for v in re.findall(
                            r"-?\d+", str(a["static_sizes"]).split(":")[-1]
                        )
                    ]
                    dynamic = any(v < 0 for v in sizes) or bits is None
                    n = 1
                    for v in sizes:
                        n *= v
                    nbytes = None if dynamic else n * bits // 8
                    pkt = _pkt(a["packet"]) if "packet" in a else None
                    # Each run of the BD's outermost dimension sends a header.
                    runs = None if dynamic else sizes[0]
                    events.append(("memcpy", sym, pkt, nbytes, in_loop, runs))
                elif x.name in (
                    "aiex.dma_configure_task",
                    "aiex.dma_configure_task_for",
                ):
                    if x.name == "aiex.dma_configure_task":
                        tile = tile_of(x.operands[0])
                        dr, ch, alloc = _int(a["direction"]), _int(a["channel"]), None
                        dyn = len(x.operands) > 1
                    else:
                        alloc = str(a["alloc"]).lstrip("@")
                        al = d.allocs.get(alloc)
                        tile = al["tile"] if al else None
                        dr, ch = (al["dir"], al["ch"]) if al else (None, None)
                        dyn = len(x.operands) > 0
                    blocks = []
                    for b in x.regions[0].blocks:
                        ops = []
                        for y in _ops(b):
                            walk_ops(y, ops)
                        blocks.append(ops)
                    if tile is not None:
                        results[value_key(x.results[0])] = len(d.programs)
                        d.programs.append(
                            dict(
                                tile=tile,
                                kind="task" if alloc is None else "task_for",
                                alloc=alloc,
                                dir=dr,
                                ch=ch,
                                seq=blocks,
                                loops=False,
                                repeat=(
                                    _int(a["repeat_count"])
                                    if "repeat_count" in a
                                    else 0
                                ),
                                dyn_repeat=dyn,
                                users=[],
                                sequence=k,
                            )
                        )
                elif x.name in ("aiex.dma_start_task", "aiex.dma_await_task"):
                    kind = "start" if x.name == "aiex.dma_start_task" else "await"
                    p = results.get(value_key(x.operands[0]))
                    if isinstance(p, int):
                        if kind == "start":
                            d.programs[p]["users"].append(in_loop)
                            events.append(("start", p, in_loop))
                        else:
                            events.append(("await", p))
                        ps = creators.get(value_key(x.operands[0]), [p])
                        if ps != [p]:
                            # The host model issues or waits on all of them.
                            keys = [
                                None if q is None else d.program_key(d.programs[q])
                                for q in ps
                            ]
                            known = all(
                                k is not None and k[2] is not None for k in keys
                            )
                            events.append(
                                (
                                    "host_alts",
                                    kind,
                                    tuple(dict.fromkeys(keys)) if known else (),
                                )
                            )
                    elif p is not None and kind == "await":
                        events.append(("await_chain", p[1]))
                    elif kind == "await":
                        events.append(("await_any",))
                elif x.name == "aiex.npu.dma_wait":
                    events.append(("wait", str(a["symbol"]).lstrip("@")))
                elif x.name in (
                    "aiex.dma_start_bd_chain",
                    "aiex.dma_start_bd_chain_for",
                ):
                    if x.name == "aiex.dma_start_bd_chain":
                        tile = next(
                            (tile_of(v) for v in x.operands if tile_of(v) is not None),
                            None,
                        )
                        key = (
                            None
                            if tile is None
                            else (*tile, _int(a["direction"]), _int(a["channel"]))
                        )
                        alloc = None
                    else:
                        key, alloc = None, str(a["alloc"]).lstrip("@")
                    results[value_key(x.results[0])] = ("chain", len(events))
                    events.append(("chain", key, alloc, in_loop))
                elif x.name == "arith.constant":
                    lock_ops(x, [])
                elif x.name in ("scf.for", "scf.while", "scf.parallel", "affine.for"):
                    events.append(("loop_begin",))
                    for region in x.regions:
                        for b in region.blocks:
                            for y in _ops(b):
                                walk(y, True)
                    events.append(("loop_end",))
                    forward_yields(x)
                elif x.name not in ("aie.end", "scf.yield", "aie.next_bd"):
                    if x.regions:
                        for region in x.regions:
                            for b in region.blocks:
                                for y in _ops(b):
                                    walk(y, in_loop)
                        forward_yields(x)

            for b in op.regions[0].blocks:
                for x in _ops(b):
                    walk(x, False)
            d.sequences.append(events)

        for op in _ops(device.regions[0].blocks[0]):
            a = _attrs(op)
            name = op.name
            if name == "aie.tile":
                t = (_int(a["col"]), _int(a["row"]))
                tiles[value_key(op.results[0])] = t
                d.tiles.append(t)
            elif name == "aie.lock":
                t = tile_of(op.operands[0])
                sym = (
                    str(a["sym_name"]).strip('"')
                    if "sym_name" in a
                    else f"lock{len(locks)}"
                )
                lid = _int(a["lockID"]) if "lockID" in a else -1
                init = _int(a["init"]) if "init" in a else 0
                locks[value_key(op.results[0])] = sym
                d.locks[sym] = (t[0], t[1], lid, init)
            elif name in ("aie.mem", "aie.memtile_dma", "aie.shim_dma"):
                dma_programs(op, tile_of(op.operands[0]))
            elif name == "aie.core":
                uses = []
                for region in op.regions:
                    blocks = list(region.blocks)
                    for b in blocks:
                        for x in _ops(b):
                            walk_ops(
                                x, uses, d.core_repeated_releases, len(blocks) == 1
                            )
                d.cores[tile_of(op.operands[0])] = [
                    (u[1], u[2], u[3]) for u in uses if u[0] == "lock"
                ]
            elif name == "aie.switchbox":
                d.boxes.setdefault(tile_of(op.operands[0]), []).extend(box_ops(op))
            elif name == "aie.shim_mux":
                d.muxes.setdefault(tile_of(op.operands[0]), []).extend(
                    (o[1], o[2]) for o in box_ops(op) if o[0] == "connect"
                )
            elif name == "aie.flow":
                s, t = tile_of(op.operands[0]), tile_of(op.operands[1])
                d.flows.append(
                    (
                        (*s, _int(a["source_bundle"]), _int(a["source_channel"])),
                        (*t, _int(a["dest_bundle"]), _int(a["dest_channel"])),
                    )
                )
            elif name == "aie.packet_flow":
                srcs, dsts = [], []
                for x in _ops(op.regions[0].blocks[0]):
                    xa = _attrs(x)
                    if x.name in ("aie.packet_source", "aie.packet_dest"):
                        ep = (
                            *tile_of(x.operands[0]),
                            _int(xa["bundle"]),
                            _int(xa["channel"]),
                        )
                        (srcs if x.name == "aie.packet_source" else dsts).append(ep)
                d.add_packet_flow(
                    _int(a["ID"]),
                    srcs,
                    dsts,
                    keep=_int(a["keep_pkt_header"]) if "keep_pkt_header" in a else None,
                    priority=(
                        _int(a["priority_route"]) if "priority_route" in a else None
                    ),
                    mask=_int(a["mask"]) if "mask" in a else None,
                )
            elif name == "aie.shim_dma_allocation":
                d.allocs[str(a["sym_name"]).strip('"')] = dict(
                    tile=tile_of(op.operands[0]),
                    dir=_int(a["channel_dir"]),
                    ch=_int(a["channel_index"]),
                    pkt=_pkt(a["packet"]) if "packet" in a else None,
                )
            elif name == "aie.runtime_sequence":
                sequence(op)
            elif name in (
                "aie.buffer",
                "aie.external_buffer",
                "aie.wire",
                "aie.end",
                "aie.bd_chain",
            ):
                pass
            elif name == "arith.constant":
                lock_ops(op, [])
            else:
                d.unsupported.append(name)
        for p in d.programs:
            if p["kind"] == "start" and not p["loops"]:
                pass
        return d


def design_signature(d):
    """What the model reads, for comparing two Designs: a snapshot, as the
    model prunes Designs in place."""
    return repr(
        (
            d.dev,
            sorted(d.tiles),
            [(k, v[:2], v[3]) for k, v in d.locks.items()],
            [
                (
                    p["tile"],
                    p["kind"],
                    p["dir"],
                    p["ch"],
                    p["seq"],
                    p["loops"],
                    p["repeat"],
                    p["dyn_repeat"],
                    p["users"],
                )
                for p in d.programs
            ],
            sorted(d.cores.items()),
            sorted(
                (
                    t,
                    [
                        o if o[0] != "masterset" else o[:2] + (len(o[2]),) + o[3:]
                        for o in ops
                    ],
                )
                for t, ops in d.boxes.items()
            ),
            sorted(d.muxes.items()),
            d.flows,
            d.packet_flows,
            sorted(d.allocs.items(), key=lambda kv: kv[0]),
            d.sequences,
        )
    )


def keeps_header(tile, port, keep):
    """AIERT: packets leave a master with their header unless it is a DMA or
    the shim's South; keep_pkt_header overrides."""
    if keep is not None:
        return keep
    return port[0] != DMA and not (tile[1] == 0 and port[0] == SOUTH)


def last_keep(d):
    """keep_pkt_header per destination port: the last packet flow into it
    decides, as keepPktHeaderAttr is written per destination."""
    keep = {}
    for f in d.packet_flows:
        for t in f["dsts"]:
            keep[t] = f["keep"]
    return keep


def overlay_keep_conflicts(d):
    """The ports where the last flow in keeps headers otherwise than the
    prioritized flows into them do, each with that last flow: a control-packet
    reload keeps their keep_pkt_header, so the router rejects the design if
    one reloads it."""
    prio = {(s, f["id"]) for f in d.packet_flows if f["priority"] for s in f["srcs"]}
    last, last_prio = {}, {}
    for f in d.packet_flows:
        for t in f["dsts"]:
            last[t] = f
            if any((s, f["id"]) in prio for s in f["srcs"]):
                last_prio[t] = f
    return [
        (t, last[t])
        for t, p in last_prio.items()
        if keeps_header(t[:2], t[2:], last[t]["keep"])
        != keeps_header(t[:2], t[2:], p["keep"])
    ]


def agree_overlay_keep(d):
    """Drop where the last flow into a port keeps headers there otherwise
    than the prioritized flows into it do."""
    while wrong := overlay_keep_conflicts(d):
        for t, f in wrong:
            f["dsts"].remove(t)
        d.packet_flows = [f for f in d.packet_flows if f["dsts"]]
