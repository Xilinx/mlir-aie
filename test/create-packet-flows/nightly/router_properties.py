#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# RUN: %python %s --device npu1 | FileCheck %s
# RUN: %python %s --device npu2 | FileCheck %s
# RUN: %python %s --device xcvc1902 | FileCheck %s

"""Model-based property test for --aie-create-pathfinder-flows.

The .mlir tests in this directory pin exact switchbox settings on a handful of
designs. This file checks what any routing must satisfy on many generated
ones, against a model of the fabric and of the router's deadlock rules:

 * the fabric: port counts per tile from the TargetModel bindings, legal
   crossbar connections (isLegalTileConnection on AIE1 and AIE2), six
   arbiters of four msels per switchbox, four packet rules per slave port;
 * which streams can deadlock (AIEStreamDependencyAnalysis, mirrored here
   line for line: stream volumes, receiver capacities, the waits-for graph
   of cores, DMA channels and the host, and the global hold-cycle search);
 * the router's own arbiter planning and pre-routing rejection
   (planArbiters, cutTiles, unroutableArbiters, the hold-cycle search), run
   on a routing the generator builds itself as a witness.

Designs come in three tiers:

 * routable: the generator routes every flow on exclusive links, plans its
   arbiters with the router's own rules, and emits only the flows, programs,
   runtime sequence and pre-placed switchbox configuration. The router has
   to succeed with allow-deadlock-prone, and its output is checked hop by
   hop; without it, it has to fail exactly where the model says the flows
   can deadlock.
 * unroutable: more pairwise conflicting streams must take an arbiter at a
   tile than it has (the router's message is predicted exactly), more
   circuit flows must cross a port than it has channels, or conflicting
   streams must merge. The router has to fail.
 * unknown: recorded; whatever the router emits is still checked, with the
   hold-cycle rule applied to the output.

The model lives in the aiemodel package beside this file: Target in
aiemodel.fabric, Design and load_design in aiemodel.design, generate and
verdict in aiemodel.generate, and verify in aiemodel.verify.
"""

import argparse
import json
import random
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from aiemodel.params import PARAMS, SPACE_FLOORS
from aiemodel.fabric import CORE, DMA, MM2S, NORTH, S2MM, SOUTH
from aiemodel.design import Design, design_signature, load_design
from aiemodel.deadlock import Analysis, trace_routed_streams
from aiemodel.run import (
    ALLOW_HINT,
    SHARED_RECEIVER_WARNING,
    aie_opt,
    debug_flags,
    first_error,
    route_batch,
    router_warning,
    shared_receiver_warning,
    unavoidable_warning,
)
from aiemodel.routing import (
    Construction,
    overlay_design,
    overlay_problems,
    pins_hops_fn,
    plan_routing,
    standalone_overlay,
    unroutable_arbiters,
)
from aiemodel.verify import known_bug, verify
from aiemodel.generate import (
    add_program,
    bd_block,
    canonical,
    gen_wormhole,
    random_design,
    routable_case,
    unknown_case,
    unroutable_case,
    verdict,
)
from aiemodel.route_space import PAIRWISE, route_space, space_coverage

# Self checks.


def check_model():
    """Answers the model owes on designs worked by hand."""
    out = []

    def expect(label, got, want):
        if got != want:
            out.append(f"{label}: got {got!r}, want {want!r}")

    # Unknown ids obey first-match rules and can diverge downstream.
    d = Design("npu1_1col")
    d.boxes[(0, 5)] = [
        ("amsel", "a", 0, 0),
        ("amsel", "b", 1, 0),
        ("masterset", (SOUTH, 0), ["a"], False),
        ("masterset", (SOUTH, 1), ["b"], False),
        ("rules", (DMA, 0), [(30, 30, "a"), (31, 31, "b")]),
    ]
    d.boxes[(0, 4)] = [
        ("amsel", "a", 0, 0),
        ("amsel", "b", 1, 0),
        ("masterset", (DMA, 0), ["a"], False),
        ("masterset", (DMA, 1), ["b"], False),
        ("rules", (NORTH, 0), [(31, 30, "a"), (31, 31, "b")]),
        ("connect", (NORTH, 1), (CORE, 0)),
    ]
    expect(
        "unknown ids, ordered and masked",
        [(s.dst, s.pid) for s in trace_routed_streams(d)],
        [((0, 4, DMA, 0), 30), ((0, 4, DMA, 1), 31)],
    )
    add_program(d, (0, 5), MM2S, 0, [bd_block(64, 31)], False)
    expect(
        "known id through masked rule",
        [(s.dst, s.pid) for s in trace_routed_streams(d)],
        [((0, 4, DMA, 1), 31)],
    )
    d = Design("npu1_1col")
    d.boxes[(0, 4)] = [("connect", (DMA, 0), (DMA, 1))]
    expect("circuit has no packet id", [s.pid for s in trace_routed_streams(d)], [None])

    # arbiter_deadlock_exhausted.mlir: nothing programs memtile (1,1), so the
    # six flows into it wait on each other, and flow 6 into (0,1) waits on all.
    d = Design("npu2")
    for ch in range(6):
        d.add_packet_flow(ch, [(0, 1, DMA, ch)], [(1, 1, DMA, ch)])
    d.add_packet_flow(6, [(0, 0, DMA, 0)], [(0, 1, DMA, 0)])
    an = Analysis(d)
    reason = unroutable_arbiters(d, an, pins_hops_fn(d, False))
    expect("unprogrammed tile", (reason or "")[:14], "at tile (0, 1)")
    expect("unprogrammed clique", len(getattr(an, "clique", [])), 7)
    expect(
        "unprogrammed hops on",
        unroutable_arbiters(d, Analysis(d), pins_hops_fn(d, True)),
        None,
    )
    expect("unprogrammed verdict", verdict(d, hops_on=False)["routable"], False)
    # Its second design: (1,1) drains each channel into a buffer nothing
    # frees, so none of the six waits on another; flow 6 still waits on all.
    for ch in range(6):
        prod, cons = d.lock((1, 1), 1), d.lock((1, 1), 0)
        add_program(d, (1, 1), S2MM, ch, [bd_block(1024, None, prod, cons)], True)
    an = Analysis(d)
    an.graph
    expect("buffered capacity", an.volumes.receive_capacity((1, 1, DMA, 0)), 1024)
    expect("buffered stalls", an.can_stall(0), True)
    expect(
        "buffered conflicts",
        [(i, j) for i in range(7) for j in range(i + 1, 7) if an.conflict(i, j)],
        [(i, 6) for i in range(6)],
    )
    # A receiver that never blocks cannot stall, whatever is sent to it.
    d.programs[0]["seq"] = [bd_block(64)]
    an = Analysis(d)
    an.graph
    expect("sink capacity", an.volumes.receive_capacity((1, 1, DMA, 0)), None)
    expect("sink stalls", an.can_stall(0), False)
    # Volume counts only the BDs carrying the stream's id, a finite send that
    # fits the receiver does not stall it, a kept header does, and repeats
    # multiply.
    d = Design("npu2")
    d.add_packet_flow(3, [(0, 2, DMA, 0)], [(0, 3, DMA, 0)])
    send = add_program(d, (0, 2), MM2S, 0, [bd_block(64, 3), bd_block(128, 4)], False)
    p, c = d.lock((0, 3), 2), d.lock((0, 3), 0)
    add_program(d, (0, 3), S2MM, 0, [bd_block(32, None, p, c)], True)
    an = Analysis(d)
    an.graph
    expect("volume by id", an.volumes.send_volume(an.streams[0]), 64)
    expect(
        "finite fits",
        (an.volumes.receive_capacity((0, 3, DMA, 0)), an.can_stall(0)),
        (64, False),
    )
    d.packet_flows[0]["keep"] = True
    an = Analysis(d)
    an.graph
    expect(
        "kept header",
        (an.volumes.send_volume(an.streams[0]), an.can_stall(0)),
        (68, True),
    )
    d.packet_flows[0]["keep"] = None
    send["repeat"] = 2
    an = Analysis(d)
    an.graph
    expect("repeats", an.volumes.send_volume(an.streams[0]), 192)
    # A chain through a core: f stalls at (0,3) S2MM 0, which the core
    # drains; the core feeds MM2S 1, which sends g. g does not stall at a sink.
    send.update(repeat=0, loops=True, seq=send["seq"][:-1])
    full, empty = d.lock((0, 3), 0), d.lock((0, 3), 1)
    d.cores[(0, 3)] = [(2, c, 1), (1, p, 1), (1, full, 1), (2, empty, 1)]
    add_program(d, (0, 3), MM2S, 1, [bd_block(32, None, full, empty)], True)
    d.flows.append(((0, 3, DMA, 1), (0, 4, DMA, 0)))
    add_program(d, (0, 4), S2MM, 0, [bd_block(32)], True)
    an = Analysis(d)
    g, f = 0, 1
    expect("relayed through core", (an.blocks(f, g), an.blocks(g, f)), (True, False))
    expect("relayed conflict", an.conflict(f, g), True)
    # A channel takes no more than its least program does.
    add_program(d, (0, 4), S2MM, 0, [bd_block(16)], False)
    an = Analysis(d)
    an.graph
    expect("least program", an.volumes.receive_capacity((0, 4, DMA, 0)), 16)
    # arbiter_looped_send_volume.mlir: a looping chain sends one pass per
    # token of %go, so as many as its feeders ever release.
    d = Design("npu1_1col")
    d.add_packet_flow(1, [(0, 2, DMA, 0)], [(0, 3, DMA, 0)])
    d.add_packet_flow(2, [(0, 2, DMA, 0)], [(0, 3, DMA, 1)])
    go = d.lock((0, 2), 1)
    send = add_program(d, (0, 2), MM2S, 0, [bd_block(32, 1, go), bd_block(64, 2)], True)

    def looped(label, want):
        an = Analysis(d)
        an.graph
        got = [an.volumes.send_volume(s) for s in an.streams if s.pid == 1]
        expect(label, got, [want])

    looped("looped, nothing refills", 32)
    feed = add_program(d, (0, 2), S2MM, 0, [bd_block(32, None, None, go)], False)
    looped("looped, one-shot feeder", 64)
    feed.update(loops=True, seq=[bd_block(64, None, None, go)])
    d.flows.append(((0, 4, DMA, 0), (0, 2, DMA, 0)))
    add_program(d, (0, 4), MM2S, 0, [bd_block(32)], False)
    looped("looped, feeder fills no BD", 32)
    d.cores[(0, 2)] = [(1, go, 1)]
    looped("looped, core refills once", 64)
    d.core_repeated_releases.add(go)
    looped("looped, core refills in a loop", None)
    del d.cores[(0, 2)]
    send["seq"][0][0] = ("lock", 0, go, 1)
    looped("looped, acquire-equal", None)
    # arbiter_hold_cycle_one_holder.mlir: flows 0 and 3 share arbiter 0 at
    # (0,1). The walk through it needs both to hold it at once, which a
    # packet held until tlast rules out.
    d = Design("npu1_1col")
    for k in range(4):
        core, ch = (0, 2 + k // 2), k % 2
        d.add_packet_flow(k, [(0, 1, DMA, k)], [(*core, DMA, ch)])
        add_program(d, (0, 1), MM2S, k, [bd_block(256, k)], False)
    for r in (2, 3):
        pa, ca, pb, cb = (d.lock((0, r), init) for init in (1, 0, 1, 0))
        add_program(d, (0, r), S2MM, 0, [bd_block(64, None, pa, ca)], True)
        add_program(d, (0, r), S2MM, 1, [bd_block(64, None, pb, cb)], True)
        d.cores[(0, r)] = [(2, ca, 1), (2, cb, 1), (1, pa, 1), (1, pb, 1)]
    an = Analysis(d)
    an.graph

    def cycles(*arbiters):
        routes = [[((0, 1), (DMA, s.src[3]), arbiters[s.pid])] for s in an.streams]
        return an.hold_cycle(routes) is not None

    expect("one holder", cycles(0, 1, 2, 0), False)
    expect("same-core sharers", cycles(0, 0, 1, 2), True)
    # arbiter_hold_cycle_wormhole.mlir: no pair conflicts, but with every hop
    # packet switched the arbiters close a cycle.
    d = gen_wormhole(random.Random(0), "npu1_1col")
    an = Analysis(d)
    expect(
        "wormhole pre-routing", unroutable_arbiters(d, an, pins_hops_fn(d, False)), None
    )
    con = Construction(d, random.Random(0))
    for fl in d.packet_flows:
        con.route(fl["srcs"][0], fl["dsts"], True)
    reason = plan_routing(d, an, con.solution(), hops_on=False)["reason"] or ""
    want = "packet flows can deadlock holding arbiters across switchboxes"
    expect("wormhole", reason[: len(want)], want)
    # What the model reads survives printing and parsing.
    for seed in range(8):
        d = canonical(
            random_design(random.Random(seed), ("npu1_2col", "npu2")[seed % 2])
        )
        again = canonical(d)
        expect(
            f"roundtrip {seed}", design_signature(again) == design_signature(d), True
        )
    return out


def check_verifier():
    """The verifier has to notice a routing that lost a hop: delete each
    connect and rule of a routed design in turn, and every deletion must be
    reported."""
    d = Design("npu1_2col")
    d.flows.append(((0, 0, DMA, 0), (1, 3, DMA, 0)))
    d.add_packet_flow(5, [(1, 2, DMA, 1)], [(0, 1, DMA, 2), (0, 4, CORE, 0)])
    d.add_packet_flow(6, [(0, 1, DMA, 3)], [(0, 0, DMA, 1)])
    an = Analysis(d)
    p = aie_opt(d.emit())
    if p.returncode:
        return [f"routing the verifier's design failed: {first_error(p.stderr)}"]
    out = []
    problems, _ = verify(d, an, p.stdout, True)
    out += [f"clean routing flagged: {x}" for x in problems]
    lines = p.stdout.splitlines()
    mutable = [
        i for i, l in enumerate(lines) if "aie.connect<" in l or "aie.rule(" in l
    ]
    for i in mutable:
        broken = "\n".join(lines[:i] + lines[i + 1 :])
        if not verify(d, an, broken, True)[0]:
            out.append(f"missed deletion of: {lines[i].strip()}")
    if len(mutable) < 5:
        out.append(f"only {len(mutable)} lines to delete")
    return out


# The test.


def main(argv=None):
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--device", default="npu2", help="npu1, npu2 or a device name")
    cli.add_argument("--seeds", type=int, default=None, help="routable designs")
    cli.add_argument("--first-seed", type=int, default=0)
    cli.add_argument("--out", type=Path, default=None, help="save failing cases here")
    cli.add_argument("--param", action="append", default=[], metavar="KEY=VALUE")
    cli.add_argument(
        "--space-out", type=Path, default=None, help="write route spaces (JSON)"
    )
    args = cli.parse_args(argv)
    for kv in args.param:
        k, v = kv.split("=", 1)
        PARAMS[k] = type(PARAMS[k])(v)
    if args.seeds is not None:
        PARAMS["routable"] = args.seeds
        PARAMS["unroutable"] = PARAMS["unknown"] = max(1, args.seeds // 4)
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
    t_start = time.monotonic()
    first = args.first_seed

    regressions = []

    def report(label, ok, detail):
        print(f"{label}: {detail} : {'OK' if ok else 'REGRESSION'}", flush=True)
        if not ok:
            regressions.append(label)

    def save(case, what, text, stderr=""):
        if args.out:
            name = f"{case['tier']}-{case['shape']}-{case['seed']}-{what}"
            (args.out / f"{name}.mlir").write_text(text)
            if stderr:
                (args.out / f"{name}.err").write_text(stderr)

    problems = check_model()
    for line in problems:
        print("WRONG MODEL:", line)
    report("model-validation", not problems, f"{len(problems)} wrong answer(s)")
    problems = check_verifier()
    for line in problems:
        print("WRONG VERIFIER:", line)
    report("verifier-validation", not problems, f"{len(problems)} problem(s)")

    # Every aie-opt run goes to one pool as soon as its designs exist, so
    # generating and checking designs overlaps routing them.
    pool = ThreadPoolExecutor(max_workers=PARAMS["jobs"])
    debug = debug_flags()
    spaces = []

    def route_one(c):
        p = aie_opt(c["design"].emit(), c["hops_on"], extra=debug)
        spaces.append(
            route_space(
                c["design"], p.returncode, p.stdout, p.stderr, p.stderr, c["hops_on"]
            )
        )
        return p

    ucases = [
        unroutable_case(s, args.device)
        for s in range(first, first + PARAMS["unroutable"])
    ]
    kcases = [
        unknown_case(s, args.device) for s in range(first, first + PARAMS["unknown"])
    ]
    upending = [pool.submit(route_one, c) for c in ucases]
    kpending = [pool.submit(route_one, c) for c in kcases]

    def route(hops_on, group):
        tagged = [(c["seed"], c["design"].emit(c["seed"])) for c in group]
        t0 = time.monotonic()
        outs, errs, _ = route_batch(tagged, hops_on)
        spent = time.monotonic() - t0
        strict = route_batch(tagged, hops_on, strict=True)
        return outs, errs, route_batch(tagged, hops_on, debug), spent, strict

    cases, gave_up, pending = [], 0, {}
    size = PARAMS["batch"]

    def submit(k):
        for hops_on in (True, False):
            pending[hops_on, k] = pool.submit(route, hops_on, cases[k : k + size])

    for seed in range(first, first + PARAMS["routable"]):
        case = routable_case(seed, args.device)
        if case is None:
            gave_up += 1
            continue
        cases.append(case)
        if len(cases) % size == 0:
            submit(len(cases) - size)
    if len(cases) % size:
        submit(len(cases) - len(cases) % size)
    shapes = Counter(c["shape"] for c in cases)

    failed, illegal, nondet = [], [], 0
    inherent, inherent_wrong = Counter(), []
    shared, shared_wrong = Counter(), []
    strict, strict_wrong = Counter(), []

    def check_strict(tag, c, hops_on, allowed, warned, out, err):
        """Without allow-deadlock-prone, flows that can deadlock however they
        are routed fail, as does a design the router leaves only a hold cycle
        through receivers flows share for; the rest route as with it. warned
        is None where the routing with it gave no log of its own."""
        unavoidable = unavoidable_warning(c["analysis"])
        if unavoidable:
            kind = "inherent"
            ok = out is None and f"error: {unavoidable}{ALLOW_HINT}" in err
        elif out is None:
            kind = "shared"
            ok = (
                warned is not False
                and SHARED_RECEIVER_WARNING in err
                and ALLOW_HINT in err
            )
        else:
            kind = "same" if out == allowed else "rerouted"
            if warned is False:
                ok = kind == "same"
            else:
                problems, stats = verify(c["design"], c["analysis"], out, hops_on)
                ok = not problems and stats["shared_receiver_cycle"] is None
        strict[kind] += 1
        if not ok:
            got = "routed" if out is not None else first_error(err)[:300]
            strict_wrong.append(f"{tag}: {kind}: {got}")
            save(
                c, "strict" + ("" if hops_on else " hops-off"), c["design"].emit(), err
            )

    def check_inherent(tag, c, stderr, what="warning"):
        want, got = unavoidable_warning(c["analysis"]), router_warning(stderr)
        inherent[want is not None] += 1
        if want != got:
            inherent_wrong.append(f"{tag}: router {got!r}, model {want!r}")
            save(c, what, c["design"].emit(), stderr)

    known = Counter()
    totals = defaultdict(int)
    route_time = 0.0
    overlay_todo = []
    for hops_on, k in sorted(pending, key=lambda b: (not b[0], b[1])):
        outs, errs, again, spent, (s_outs, s_errs, _) = pending[hops_on, k].result()
        group = cases[k : k + size]
        route_time += spent
        mode = "" if hops_on else " hops-off"
        for c in group:
            seed, d = c["seed"], c["design"]
            rc, stderr = errs.get(seed, (0, ""))
            spaces.append(
                route_space(
                    d, rc, outs.get(seed, ""), stderr, again[2].get(seed, ""), hops_on
                )
            )
            if seed not in outs:
                rc, stderr = errs[seed]
                if rc > 0 and (label := known_bug(d, [first_error(stderr)])):
                    known[label] += 1
                    save(c, "known" + mode.strip(), d.emit(), stderr)
                    continue
                failed.append(
                    f"seed {seed}{mode} ({c['shape']}): rc {rc}: {first_error(stderr)[:300]}"
                )
                save(c, "failed" + mode.strip(), d.emit(), stderr)
                continue
            if outs[seed] != again[0].get(seed):
                nondet += 1
            if seed in again[2]:
                check_inherent(
                    f"seed {seed}{mode}", c, again[2][seed], "warning" + mode.strip()
                )
            problems, stats = verify(d, c["analysis"], outs[seed], hops_on)
            want = stats.pop("shared_receiver_cycle")
            if not problems:
                check_strict(
                    f"seed {seed}{mode}",
                    c,
                    hops_on,
                    outs[seed],
                    (
                        shared_receiver_warning(again[2][seed]) is not None
                        if seed in again[2]
                        else None
                    ),
                    s_outs.get(seed),
                    s_errs.get(seed, (0, ""))[1],
                )
            if seed in again[2] and not problems:
                got = shared_receiver_warning(again[2][seed])
                shared[want is not None] += 1
                if want != got:
                    shared_wrong.append(
                        f"seed {seed}{mode}: router {got!r}, model {want!r}"
                    )
                    save(c, "shared-warning" + mode.strip(), d.emit(), again[2][seed])
            overlay_todo.append((c, mode, outs[seed]))
            if problems and (label := known_bug(d, problems)):
                known[label] += 1
                save(c, "known" + mode.strip(), d.emit())
            elif problems:
                illegal.append(f"seed {seed}{mode} ({c['shape']}): {problems[0]}")
                save(c, "illegal" + mode.strip(), d.emit())
            if hops_on:
                for k, v in stats.items():
                    totals[k] += v
                totals["constructed"] += c["truth"]["witness_hops"]
                totals["designs"] += 1
    runs = 2 * len(cases)
    for line in failed[:10]:
        print("FAILED:", line)
    for line in illegal[:10]:
        print("ILLEGAL:", line)
    report(
        "completeness",
        runs
        and (runs - len(failed)) / runs >= PARAMS["min_routed_rate"]
        and len(cases) >= PARAMS["routable"] * 0.9,
        f"{runs - len(failed)}/{runs} routed ({len(cases)} designs, "
        f"{shapes['fixed-ir']} with fixed switchboxes, both hop modes, "
        f"{gave_up} generator gave up)",
    )
    report(
        "legality",
        not illegal,
        f"{len(illegal)} illegal of {runs - len(failed)} routed",
    )
    report("determinism", nondet == 0, f"{nondet} unstable")

    # Without a control-packet reload the router may move the overlay, but a
    # design routes only if its overlay routes by itself.
    alone = {c["seed"]: o for c in cases if (o := overlay_design(c["design"]))}
    alone_outs, alone_errs, _ = route_batch(
        [(k, o.emit(k)) for k, o in alone.items()], False
    )
    overlay_wrong, overlay_checked, overlay_moved = [], 0, 0
    for c, mode, text in overlay_todo:
        seed = c["seed"]
        if seed not in alone:
            continue
        overlay_checked += 1
        tag = f"seed {seed}{mode} ({c['shape']})"
        if seed not in alone_outs:
            err = first_error(alone_errs[seed][1])[:300]
            overlay_wrong.append(f"{tag}: routed, but its overlay alone fails: {err}")
            save(c, "overlay" + mode.strip(), c["design"].emit())
        elif overlay_problems(load_design(text), load_design(alone_outs[seed])):
            overlay_moved += 1
    for line in overlay_wrong[:10]:
        print("OVERLAY FAILS:", line)
    report(
        "overlay",
        not overlay_wrong,
        f"{overlay_checked - len(overlay_wrong)}/{overlay_checked} routings have an "
        f"overlay that routes by itself, {overlay_moved} move it",
    )

    # A control-packet reload installs @ctrl_pkt_overlay, routed without the
    # design's own switchboxes, so the design has to set the overlay's
    # switches as that does or fail.
    reloads = {}
    for c in cases:
        if s := standalone_overlay(c["design"]):
            d = c["design"].copy()
            d.reload = True
            reloads[c["seed"]] = (c, d, s)
    r_outs, r_errs, _ = route_batch(
        [(k, d.emit(k)) for k, (_, d, _) in reloads.items()], True
    )
    s_outs, s_errs, _ = route_batch(
        [(k, s.emit(k)) for k, (_, _, s) in reloads.items()], True
    )
    reload_wrong, reload_unroutable = [], 0
    for seed, (c, d, s) in reloads.items():
        tag = f"seed {seed} ({c['shape']})"
        if seed not in r_outs:
            reload_unroutable += 1
            continue
        if seed not in s_outs:
            err = first_error(s_errs[seed][1])[:300]
            reload_wrong.append(f"{tag}: routed, but @ctrl_pkt_overlay fails: {err}")
        elif problems := overlay_problems(
            load_design(r_outs[seed]), load_design(s_outs[seed])
        ):
            reload_wrong.append(f"{tag}: {problems[0]}")
        else:
            continue
        save(c, "reload", d.emit())
    for line in reload_wrong[:10]:
        print("RELOAD MOVED:", line)
    report(
        "reload",
        not reload_wrong,
        f"{len(reloads) - len(reload_wrong) - reload_unroutable}/{len(reloads)} "
        f"reloaded designs keep @ctrl_pkt_overlay's switch settings, "
        f"{reload_unroutable} fail",
    )

    wrong, exact, undecided = [], 0, 0
    for c, f in zip(ucases, upending):
        p = f.result()
        want = c["truth"]["expect"]
        tag = f"seed {c['seed']} {c['shape']} ({c['design'].dev})"
        if want is None:
            undecided += 1
            if "no two of" in p.stderr:
                wrong.append(
                    f"{tag}: model finds no overfull tile, router: {first_error(p.stderr)[:300]}"
                )
            continue
        if p.returncode == 0:
            wrong.append(f"{tag}: routed, wants {want[:200]}")
        elif p.returncode < 0 or want not in p.stderr:
            wrong.append(
                f"{tag}: rc {p.returncode}: {first_error(p.stderr)[:300]}, wants {want[:300]}"
            )
        else:
            exact += c["shape"] in ("arbiters", "wormhole")
            continue
        save(c, "wrong", c["design"].emit(), p.stderr)
    for line in wrong[:10]:
        print("WRONG VERDICT:", line)
    decided = len(ucases) - undecided
    report(
        "unroutable",
        not wrong and decided,
        f"{decided - len(wrong)}/{decided} rejected as predicted ({exact} with the "
        f"exact message), {undecided} left to the model",
    )

    outcomes = defaultdict(lambda: [0, 0])
    unknown_illegal = []
    for c, f in zip(kcases, kpending):
        p = f.result()
        outcomes[c["shape"]][p.returncode != 0] += 1
        tag = f"seed {c['seed']} ({c['shape']})"
        if p.returncode >= 0:
            check_inherent(tag, c, p.stderr)
        if p.returncode < 0:
            unknown_illegal.append(f"{tag}: crashed: {first_error(p.stderr)[:300]}")
            save(c, "crash", c["design"].emit(), p.stderr)
        elif p.returncode == 0:
            problems, _ = verify(c["design"], c["analysis"], p.stdout, c["hops_on"])
            if problems and (label := known_bug(c["design"], problems)):
                known[label] += 1
                save(c, "known", c["design"].emit())
            elif problems:
                unknown_illegal.append(f"{tag}: {problems[0]}")
                save(c, "illegal", c["design"].emit())
    pool.shutdown()
    for line in unknown_illegal[:10]:
        print("ILLEGAL:", line)
    print(
        "unknown: "
        + ", ".join(
            f"{k} {v[0]} routed {v[1]} failed" for k, v in sorted(outcomes.items())
        )
    )
    report(
        "unknown-legality",
        not unknown_illegal,
        f"{len(unknown_illegal)} illegal or crashed of {len(kcases)}",
    )
    for line in inherent_wrong[:10]:
        print("WRONG WARNING:", line)
    report(
        "inherent",
        not inherent_wrong,
        f"{sum(inherent.values()) - len(inherent_wrong)}/{sum(inherent.values())} "
        f"warn as the model predicts ({inherent[True]} warn)",
    )
    for line in shared_wrong[:10]:
        print("WRONG SHARED-RECEIVER WARNING:", line)
    report(
        "shared-receiver",
        not shared_wrong,
        f"{sum(shared.values()) - len(shared_wrong)}/{sum(shared.values())} "
        f"warn of shared-receiver hold cycles as the model predicts "
        f"({shared[True]} warn)",
    )
    for line in strict_wrong[:10]:
        print("WRONG STRICT:", line)
    report(
        "strict",
        not strict_wrong,
        f"{sum(strict.values()) - len(strict_wrong)}/{sum(strict.values())} route "
        f"or fail without allow-deadlock-prone as they should ({strict['same']} "
        f"route the same, {strict['rerouted']} reroute around a shared-receiver "
        f"hold cycle, {strict['shared']} fail on one, {strict['inherent']} fail "
        f"on flows that deadlock however they are routed)",
    )

    for label, count in sorted(known.items()):
        print(f"known: {count} {label}")
    n = max(totals["designs"], 1)
    ratio = totals["hops"] / max(totals["constructed"], 1)
    report(
        "hops",
        ratio <= PARAMS["max_hop_ratio"],
        f"{totals['hops']} switchbox hops, {ratio:.3f}x the constructed routing, "
        f"{totals['hops'] / max(totals['bound'], 1):.3f}x the Manhattan bound "
        f"(max {PARAMS['max_hop_ratio']}x constructed)",
    )
    max_amsels = PARAMS["max_amsels"][args.device]
    report(
        "arbiters",
        totals["amsels"] / n <= max_amsels,
        f"{totals['amsels'] / n:.2f} amsels on {totals['arbiters'] / n:.2f} arbiters per "
        f"design, {totals['promoted']} hops circuit switched (max {max_amsels} amsels)",
    )
    report(
        "priority",
        totals["low_priority"] / n <= PARAMS["max_low_priority"],
        f"{totals['low_priority'] / n:.3f} priority hops per design under a non-priority "
        f"msel, {totals['mixed_ctrl']} ctrl mastersets also carrying non-priority flows "
        f"(max {PARAMS['max_low_priority']})",
    )
    dims, pairs, hit, _ = space_coverage(spaces, args.device)
    floors = SPACE_FLOORS.get(args.device, {})
    low = [f"{k} {dims[k][0]} < {n}" for k, n in floors.items() if dims[k][0] < n]
    ph = sum(h for h, _ in pairs.values())
    pn = sum(n for _, n in pairs.values())
    report(
        "route-space",
        not low,
        f"{sum(h for h, _ in dims.values())}/{sum(n for _, n in dims.values())} "
        f"values over {len(dims)} dimensions, {ph}/{pn} pairs over "
        f"{'/'.join(PAIRWISE)}"
        + ("" if debug else ", no router internals (no debug build)")
        + (f"; below floor: {', '.join(low)}" if low else ""),
    )
    if args.space_out:
        args.space_out.write_text(
            json.dumps(
                [
                    dict(sp, dims={k: sorted(v) for k, v in sp["dims"].items()})
                    for sp in spaces
                ]
            )
        )

    per_design = route_time / max(runs, 1) * 1000
    max_ms = PARAMS["max_ms_per_design"][args.device]
    report(
        "time",
        per_design <= max_ms,
        f"{per_design:.1f} ms per design (max {max_ms})",
    )
    print(f"total: {time.monotonic() - t_start:.1f}s")
    return 1 if regressions else 0


# CHECK: model-validation: {{.*}} : OK
# CHECK: verifier-validation: {{.*}} : OK

# CHECK: completeness: {{.*}} : OK
# CHECK: legality: {{.*}} : OK
# CHECK: determinism: {{.*}} : OK
# CHECK: overlay: {{.*}} : OK
# CHECK: unroutable: {{.*}} : OK
# CHECK: unknown-legality: {{.*}} : OK
# CHECK: hops: {{.*}} : OK
# CHECK: arbiters: {{.*}} : OK
# CHECK: priority: {{.*}} : OK
# CHECK: route-space: {{.*}} : OK
# CHECK: time: {{.*}} : OK
# CHECK-NOT: REGRESSION

if __name__ == "__main__":
    sys.exit(main())
