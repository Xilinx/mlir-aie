#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# RUN: %python %s | FileCheck %s

# Property-based test for --aie-assign-buffer-addresses.
#
# The individual .mlir tests in this directory pin exact addresses, which fixes
# behavior on a handful of designs. This file measures legality, completeness,
# determinism and quality (bank spread, contiguity) over many generated designs.
#
# Feasibility comes from a construct-then-hide oracle: the generator builds a
# valid layout, then hides a random subset of it behind `address`, `mem_bank` or
# nothing. The constructed layout is a solution by construction, so a failure to
# allocate is an allocator bug and not an infeasible input.
#
# Completeness and quality are ratchets, not absolutes: packing around fixed
# obstacles is NP-hard, so the allocator is a heuristic and a few adversarial
# layouts defeat it. Tighten the bounds when the allocator improves; a drop
# below them is a regression.

import argparse
import random
import re
import subprocess
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

SEEDS = 200
# All 200 fixed seeds solve, and placement backtracks rather than giving up on
# the first ranked choice, so there is no slack here to absorb an adversarial
# seed: one that defeats the search should fail this and be looked at.
# Note these 200 are solvable without backtracking -- search coverage comes from
# the bank-reservation corpus below, which search-exercised counts.
MAX_CROSSINGS = 0
# Co-residency: how many buffer pairs share a bank, against the fewest the bank
# count allows. Two buffers a kernel reads together serialize on one bank, and
# byte spread across banks does not report that, so this measures it directly.
# 0 is the most even split available, 1 puts every buffer in one bank.
MAX_BANK_PAIR_SHARING = 0.27

# `vec` is the widest alignment a core access can demand (npu2: 512 bits), which
# is also the data region's alignment. A memtile is reached by DMA rather than
# by core vector load and store, so its bus width covers both.
DEVICES = [
    dict(
        name="core",
        dev="npu2",
        tile=(0, 2),
        cap=65536,
        banks=4,
        bus=32,
        vec=64,
        stack=1024,
    ),
    dict(
        name="memtile",
        dev="npu2",
        tile=(0, 1),
        cap=524288,
        banks=8,
        bus=4,
        vec=4,
        stack=0,
    ),
]

BUF_RE = re.compile(r"aie\.buffer.*?\{(.*?)\}\s*:\s*memref<(\d+)xi8>")


def align_up(v, a):
    return ((v + a - 1) // a) * a


def oracle_largest_free_run(cap, stack, blocks, vec):
    """Largest contiguous free run above the stack in the as-constructed
    (pre-hide) oracle layout. Caps a randomly chosen data_size to a
    value the oracle's own placement supports. Aligned to `vec` for the same
    reason largest_free_run is: that is the run the allocator has to find."""
    taken = sorted((b["addr"], b["addr"] + b["size"]) for b in blocks if b["size"])
    best, cursor = 0, stack

    def run(lo, hi):
        return max(0, hi - align_up(lo, vec))

    for lo, hi in taken:
        if lo > cursor:
            best = max(best, run(cursor, lo))
        cursor = max(cursor, hi)
    return max(best, run(cursor, cap))


def build_design(rng, cfg):
    """A random valid layout, then hide part of it.

    Returns (mlir, blocks, core_data_name): the name of the block the core
    claims as its data_size, or None. Only a "core" config claims one, because
    data_size is a CoreOp attribute and the memtile config has no aie.core.
    """
    cap, bus, stack = cfg["cap"], cfg["bus"], cfg["stack"]
    bank = cap // cfg["banks"]
    cursor, blocks = stack, []
    for _ in range(rng.randint(1, 14)):
        if rng.random() < 0.30:  # leave a hole for a later buffer to find
            cursor += rng.randint(1, max(1, bank // 2))
        aligned = rng.random() > 0.15
        if aligned:
            cursor = align_up(cursor, bus)
        roll = rng.random()
        if roll < 0.10:  # zero-sized: covers no bytes, placeable anywhere
            size = 0
        elif roll < 0.60:
            size = rng.randint(1, 8) * bus
        elif roll < 0.88:
            size = rng.randint(1, max(1, bank // (2 * bus))) * bus
        else:  # deliberately larger than one bank
            size = rng.randint(bank + 1, min(3 * bank, cap - stack))
        if cursor + size > cap:
            break
        blocks.append(dict(addr=cursor, size=size, aligned=aligned))
        cursor += size
    if not blocks:
        return None, None, 0

    for i, b in enumerate(blocks):
        b["name"] = f"b{i}"
        # A zero-sized block covers no bytes, so it can never actually leave a
        # bank; the naive addr/(addr+size-1) span is off by one at size 0 and
        # would wrongly disqualify it from the "bank" role at a bank boundary.
        inside_one_bank = b["size"] == 0 or (
            (b["addr"] // bank) == ((b["addr"] + b["size"] - 1) // bank)
        )
        roll = rng.random()
        if roll < 0.30:
            b["role"] = "pin"
        elif roll < 0.45 and inside_one_bank and b["aligned"]:
            # mem_bank is a hard constraint, so request it only when the block
            # lies inside one bank; otherwise the design is infeasible.
            b["role"] = "bank"
        else:
            b["role"] = "free"

    # data_size exercises the allocator's tight packing and its
    # reservation check. Only a config with a core supports it. The value is
    # capped to a fraction of the *oracle's own* largest contiguous gap, not of
    # the total free space, which keeps the construct-then-hide guarantee: the
    # allocator reorders blocks (bank-pinned first, then largest first), so it
    # does not reproduce the oracle's exact packing. The margin below the
    # oracle's number keeps this a legality question, and not a feasibility
    # question this generator cannot answer.
    # Hand one free block to the core as its data_size. The allocator rebuilds
    # it as a core_data aie.buffer, so the layout stays feasible by construction
    # and the block is checked like any other.
    core_data = None
    if cfg["stack"]:
        free = [b for b in blocks if b["role"] == "free" and b["size"] > 0]
        if free and rng.random() < 0.35:
            core_data = rng.choice(free)
            core_data["name"] = f'core_data_{cfg["tile"][0]}_{cfg["tile"][1]}'

    lines = [
        "module {",
        f'  aie.device({cfg["dev"]}) {{',
        f'    %t = aie.tile({cfg["tile"][0]}, {cfg["tile"][1]})',
    ]
    for b in blocks:
        if b is core_data:
            continue
        attrs = [f'sym_name = "{b["name"]}"']
        if b["role"] == "pin":
            attrs.append(f'address = {b["addr"]} : i32')
        if b["role"] == "bank":
            attrs.append(f'mem_bank = {b["addr"] // bank} : i32')
        if not b["aligned"]:
            attrs.append("aligned = false")
        lines.append(
            f'    %{b["name"]} = aie.buffer(%t) {{{", ".join(attrs)}}} '
            f'    : memref<{b["size"]}xi8>'
        )
    if cfg["stack"]:
        core_attrs = f"stack_size = {stack} : i32"
        if core_data:
            core_attrs += f', data_size = {core_data["size"]} : i32'
        lines.append(f"    aie.core(%t) {{ aie.end }} {{{core_attrs}}}")
    else:
        lines.append("    aie.memtile_dma(%t) { aie.end }")
    lines += ["  }", "}", ""]
    return "\n".join(lines), blocks, core_data["name"] if core_data else None


# Object-derived per-bank reservations, on their own seed set so the numbers
# above keep measuring exactly what they measured before.
BANKRES_SEEDS = 12000
# Set at the answer, not at what the allocator manages today: these two bounds
# are the one place here that is not a ratchet. A ratchet guards what works,
# which is right for a healthy metric and wrong for a known defect. Every design
# counted is feasible by construction, so the only correct score is all of them.
#
# Currently ~97%: bank-pinned blocks go first and then largest-first, so on a
# nearly full tile the allocator cannot rebuild the packing the oracle
# constructed. Failures concentrate above 75% density, none at or below 70%.
MIN_BANKRES_RATE = 1.0
# Tiles whose ranked first choice does not work out, so only backtracking
# places them, summed over the corpus from the pass's own statistics. A floor
# rather than a ceiling: it guards the *corpus*, not the allocator. If a
# generator change stops producing contested layouts this drops, and
# bank-reservations would still read 100% while testing nothing hard.
MIN_SEARCHED_TILES = 216
# Designs per aie-opt invocation. Startup dominates the per-design cost, so
# batching is what makes a corpus of thousands fit in the time budget.
BATCH_SIZE = 200

# The reported bug, measured directly. The same designs are placed with their
# objects' bank demands withheld, exactly as the pipeline withholds them today,
# and the resulting layout is asked whether `coreBankRegions` would still leave
# each bank enough room for the section the linker is about to put there. A
# design that fails this places fine and then fails to link.
#


def model_core_objects(rng, cfg):
    """What a core's object files demand of its data memory.

    Modelled rather than compiled: the toolchain's only contribution to
    placement is a set of sizes and alignments, so generating them directly
    keeps this at `aie-opt` speed while still covering the shapes a real link
    produces. A core links one or more objects, and each may carry

      * `.aie.bank<N>` sections, which must land in bank N;
      * ordinary `.data`/`.rodata`/`.bss`, which need one contiguous run;
      * for a prebaked `elf_file` core, ranges already fixed at absolute
        addresses, which nothing may move.

    The linker concatenates same-named sections from every input object, so two
    objects pinning the same bank need the *sum plus the padding between them*,
    not the larger of the two. Getting that wrong under-reserves exactly when a
    core links several kernels, so it is modelled here rather than assumed.

    Returns (per_bank, data_size) with per_bank[b] = (size, align).
    """
    nbanks = cfg["banks"]
    vec = cfg["vec"]
    per_bank, data_size = {}, 0

    def contribute(b, size, align):
        # Concatenation: align this object's contribution onto the running
        # total, then add it. The section's alignment is the strictest any
        # contributor asked for.
        cur_size, cur_align = per_bank.get(b, (0, 1))
        per_bank[b] = (align_up(cur_size, align) + size, max(cur_align, align))

    for _ in range(rng.randint(1, 3)):  # several kernels linked into one core
        if nbanks >= 2 and rng.random() < 0.45:
            # A LUT-shaped kernel: `aie::lut<4>` reads two tables at once, so
            # they are equal-sized, vector-aligned, and must land in different
            # banks. The runtime library ships several such pairs in one object
            # (exp and tanh), and the reported design was exactly this shape.
            for _ in range(rng.randint(1, 2)):
                lo, hi = rng.sample(range(nbanks), 2)
                size = rng.choice([256, 512, 1024])
                contribute(lo, size, 64)
                contribute(hi, size, 64)
        else:
            for b in rng.sample(range(nbanks), rng.randint(0, min(2, nbanks))):
                contribute(b, rng.randint(1, 8) * 64, rng.choice([vec, vec, 64]))
        if rng.random() < 0.7:
            data_size = align_up(data_size, vec) + rng.randint(1, 12) * 64
    return per_bank, data_size


def build_bank_reservation_design(rng, cfg, tag=None):
    """A layout where object-derived per-bank reservations contend with buffers.

    `build_design` models only what the allocator can already see: buffers, and
    one core-owned block standing in for `data_size`. It has no notion of the
    static data a kernel object pins to a bank, so the whole class of "the
    object needs bank space the allocator was never told about" is outside its
    search space.

    That class is what the reported bug was: two bank-filling buffers and two
    512-byte tables pinned to banks 0 and 1, where the only working layout puts
    the buffers in banks 2 and 3. Once the pipeline measures those tables they
    reach the allocator as bank-pinned blocks, which is what this builds.

    Feasible by construction, like `build_design`: reservations sit at the
    bottom of their banks and buffers are sized to the space genuinely left.
    """
    cap, bus, vec, stack = cfg["cap"], cfg["bus"], cfg["vec"], cfg["stack"]
    nbanks = cfg["banks"]
    bank = cap // nbanks
    # Sizes come from the modelled objects, so a bank several kernels pin to
    # carries their combined demand.
    obj_banks, obj_data = model_core_objects(rng, cfg)
    if not obj_banks:
        return None, None, None
    blocks, free_top = [], {}
    n_obstacle = 0
    for i, b in enumerate(sorted(obj_banks)):
        res_size, res_align = obj_banks[b]
        base = align_up(bank * b if b else stack, max(vec, res_align))
        # An address-pinned block at the bank base pushes the reservation off
        # the bottom, leaving the bank with two holes. That is the case where
        # "largest free run in this bank" and "where the reservation actually
        # went" stop being the same answer.
        # A prebaked `elf_file` core brings ranges already fixed by its own
        # link, which the docs say to declare as address-pinned buffers. One at
        # the bank base also pushes the reservation off the bottom, leaving the
        # bank with two holes -- the case where "largest free run in this bank"
        # and "where the reservation actually went" stop being one answer.
        if rng.random() < 0.30:
            osize = align_up(rng.randint(1, 6) * vec, vec)
            if base + osize < (b + 1) * bank:
                blocks.append(
                    dict(
                        addr=base,
                        size=osize,
                        aligned=True,
                        name=f"prebaked{n_obstacle}",
                        role="pin",
                    )
                )
                n_obstacle += 1
                base = align_up(base + osize, max(vec, res_align))
        if base + res_size > (b + 1) * bank:
            continue
        blocks.append(
            dict(
                addr=base, size=res_size, aligned=True, name=f"bankres{i}", role="bank"
            )
        )
        free_top[b] = base + res_size
    if not any(b["role"] == "bank" for b in blocks):
        return None, None, None

    # The core's own static data is measured from the object too, and unlike a
    # reservation it needs one contiguous run wherever it fits. Carve it out of a
    # reserved bank's remainder so the layout stays feasible by construction,
    # then let the allocator rediscover it from `data_size`.
    core_data = None
    if stack and free_top and obj_data:
        host = rng.choice(sorted(free_top))
        lo = align_up(free_top[host], vec)
        nxt = min(
            [a for a, _ in ((x["addr"], x["size"]) for x in blocks) if a >= lo]
            or [(host + 1) * bank]
        )
        room = min((host + 1) * bank, nxt) - lo
        if room > vec:
            size = min(align_up(obj_data, vec), align_up(room, vec) - vec)
            if size > 0 and lo + size <= (host + 1) * bank:
                core_data = dict(
                    addr=lo,
                    size=size,
                    aligned=True,
                    name=f'core_data_{cfg["tile"][0]}_{cfg["tile"][1]}',
                    role="free",
                )
                blocks.append(core_data)
                free_top[host] = lo + size

    # Everything placed so far is immovable, and a buffer may span banks, so
    # sizing has to stop at the next of them rather than at the bank edge.
    # Without this a spanning buffer walks straight through the next bank's
    # reservation and the "solution" the generator claims to have built is not
    # one -- which silently turns generator bugs into allocator failures.
    barriers = sorted((b["addr"], b["addr"] + b["size"]) for b in blocks if b["size"])

    def barrier_after(pos):
        for lo_b, _ in barriers:
            if lo_b >= pos:
                return lo_b
        return cap

    # Fill the rest. A bank with no reservation can take one buffer big enough
    # that it fits nowhere else -- the pressure that made the original design
    # fail -- or several smaller ones, some address-pinned. Pinning mid-bank is
    # what splits a bank into two holes, which is the case where "largest free
    # run in the bank" and "where the reservation actually landed" diverge.
    n = 0
    cursor = stack
    for b in range(nbanks):
        lo = align_up(max(cursor, free_top.get(b, bank * b if b else stack)), vec)
        limit = (b + 1) * bank
        if lo >= limit:
            cursor = max(cursor, lo)
            continue
        if b not in free_top and rng.random() < 0.45:
            # One buffer filling the whole bank: the tight, reported shape.
            size = min(limit, barrier_after(lo)) - lo
            if size <= 0:
                cursor = max(cursor, lo)
                continue
            if rng.random() < 0.5:
                size = align_up(int(size * rng.uniform(0.8, 1.0)), vec)
            blocks.append(
                dict(addr=lo, size=size, aligned=True, name=f"buf{n}", role="free")
            )
            n += 1
            cursor = lo + size
            continue
        # Bias toward near-full banks. Failures live above ~90% density, so a
        # generator that mostly produces half-empty tiles measures very little.
        tight = rng.random() < 0.5
        # Otherwise several blocks, occasionally pinned or spanning into the
        # next bank. A span must stay `free`: mem_bank could not describe it.
        for _ in range(rng.randint(1, 3)):
            lo = align_up(max(lo, cursor), vec)
            room = limit - lo
            if room <= vec:
                break
            span = rng.random() < 0.20 and b + 1 < nbanks
            hi = min((b + 2) * bank if span else limit, barrier_after(lo))
            if hi <= lo:
                break
            if tight:
                size = align_up(int((hi - lo) * rng.uniform(0.5, 1.0)), vec)
                size = min(size, hi - lo)
            else:
                size = align_up(
                    rng.randint(1, max(1, (hi - lo) // (2 * vec))) * vec, vec
                )
            if size <= 0 or lo + size > hi:
                break
            role = "pin" if rng.random() < 0.35 else "free"
            # A buffer the core reaches only by DMA needs no vector alignment;
            # the allocator may place it anywhere the bus width allows.
            aligned = rng.random() > 0.15
            blocks.append(
                dict(addr=lo, size=size, aligned=aligned, name=f"buf{n}", role=role)
            )
            n += 1
            lo = cursor = lo + size
            if not tight and rng.random() < 0.35:  # a hole a later buffer finds
                lo = cursor = lo + align_up(rng.randint(1, 4) * vec, vec)

    def emit(tell_allocator):
        """Render the design as the allocator now receives it.

        The pipeline measures a core's objects before placing its buffers, so
        their bank demands arrive as bank-pinned blocks and a `data_size`.
        """
        # Naming the module lets many designs share one aie-opt invocation:
        # the name comes back in the output and says which design a placement
        # belongs to. Process startup dominates otherwise -- 100 designs cost
        # 1.54s apart and 0.25s batched.
        out = [
            "module {" if tag is None else f"module @s{tag} {{",
            f'  aie.device({cfg["dev"]}) {{',
            f'    %t = aie.tile({cfg["tile"][0]}, {cfg["tile"][1]})',
        ]
        for b in blocks:
            if b is core_data:
                continue  # the allocator rebuilds this one from data_size
            if b["role"] == "bank" and not tell_allocator:
                continue  # object-derived: invisible to placement today
            attrs = [f'sym_name = "{b["name"]}"']
            if b["role"] == "bank":
                attrs.append(f'mem_bank = {b["addr"] // bank} : i32')
            if b["role"] == "pin":
                attrs.append(f'address = {b["addr"]} : i32')
            if not b["aligned"]:
                attrs.append("aligned = false")
            out.append(
                f'    %{b["name"]} = aie.buffer(%t) {{{", ".join(attrs)}}} '
                f'    : memref<{b["size"]}xi8>'
            )
        if stack:
            core_attrs = f"stack_size = {stack} : i32"
            if core_data and tell_allocator:
                core_attrs += f', data_size = {core_data["size"]} : i32'
            out.append(f"    aie.core(%t) {{ aie.end }} {{{core_attrs}}}")
        else:
            out.append("    aie.memtile_dma(%t) { aie.end }")
        out += ["  }", "}", ""]
        return "\n".join(out)

    return (
        emit(True),
        blocks,
        dict(
            blind=emit(False),
            obj_banks=obj_banks,
            core_data=core_data["name"] if core_data else None,
        ),
    )


def bank_free_runs(cfg, placed, stack_run=(0, None)):
    """Largest aligned free run in each bank, the way `coreBankRegions` sees it.

    This is what the linker script hands each `.aie.bank<N>` output section, so
    comparing it against what the objects demand answers "would this core have
    linked?" without compiling or linking anything.
    """
    cap, vec = cfg["cap"], cfg["vec"]
    nbanks = cfg["banks"]
    bank = cap // nbanks
    stack_lo, stack_hi = stack_run
    occupied = [(stack_lo, stack_hi if stack_hi is not None else cfg["stack"])]
    occupied += [(a, a + s) for (a, s, _) in placed.values() if s]
    runs = []
    for b in range(nbanks):
        lo, hi = b * bank, (b + 1) * bank
        best, cursor = 0, lo
        for o_lo, o_hi in sorted(occupied):
            o_lo, o_hi = max(o_lo, lo), min(o_hi, hi)
            if o_lo >= o_hi:
                continue
            best = max(best, o_lo - align_up(cursor, vec))
            cursor = max(cursor, o_hi)
        runs.append(max(best, hi - align_up(cursor, vec)))
    return runs


def unmet_bank_demand(cfg, placed, obj_banks, stack_run=(0, None)):
    """Banks whose leftover run cannot hold what the objects pinned there."""
    runs = bank_free_runs(cfg, placed, stack_run)
    short = []
    for b, (size, align) in sorted(obj_banks.items()):
        have = runs[b] - (align_up(runs[b], align) - runs[b]) if align else runs[b]
        if have < size:
            short.append((b, size, runs[b]))
    return short


def oracle_problems(cfg, blocks):
    """Faults in the generated layout itself, before the allocator is asked.

    Construct-then-hide only proves a design is solvable if the construction is
    a solution. A generator bug therefore reads as an allocator failure, which
    is the most misleading way this test can break, so check it rather than
    assume it.
    """
    bad = []
    bank = cfg["cap"] // cfg["banks"]
    spans = sorted(
        (b["addr"], b["addr"] + b["size"], b["name"]) for b in blocks if b["size"]
    )
    for (a1, e1, n1), (a2, e2, n2) in zip(spans, spans[1:]):
        if a2 < e1:
            bad.append(f"{n1} [{a1},{e1}) overlaps {n2} [{a2},{e2})")
    for b in blocks:
        if b["size"] and b["addr"] < cfg["stack"]:
            bad.append(f'{b["name"]} starts under the stack')
        if b["addr"] + b["size"] > cfg["cap"]:
            bad.append(f'{b["name"]} runs past the tile')
        if b["aligned"] and b["addr"] % cfg["bus"]:
            bad.append(f'{b["name"]} claims alignment it does not have')
        if b["role"] == "bank" and b["size"]:
            lo = b["addr"] // bank
            if lo != (b["addr"] + b["size"] - 1) // bank:
                bad.append(f'{b["name"]} is bank-pinned but crosses a bank')
    return bad


STRESS_BUFFERS = 300
STRESS_MAX_SECONDS = 30


def zero_size_reserved_data_case(cfg):
    """A permanent regression guard, independent of chance: a zero-sized
    buffer pinned strictly inside the tile's only free run must not
    fragment it (a zero-length occupied interval is not a split point)."""
    cap, bus, stack = cfg["cap"], cfg["bus"], cfg["stack"]
    addr = align_up(stack + bus, bus) + bus * 100
    reserved = cap - stack
    name = f'core_data_{cfg["tile"][0]}_{cfg["tile"][1]}'
    blocks = [
        dict(addr=addr, size=0, aligned=True, name="mid", role="pin"),
        dict(addr=stack, size=reserved, aligned=True, name=name, role="free"),
    ]
    lines = [
        "module {",
        f'  aie.device({cfg["dev"]}) {{',
        f'    %t = aie.tile({cfg["tile"][0]}, {cfg["tile"][1]})',
        f'    %mid = aie.buffer(%t) {{sym_name = "mid", address = {addr} : i32}} : memref<0xi8>',
        f"    aie.core(%t) {{ aie.end }} "
        f"{{stack_size = {stack} : i32, data_size = {reserved} : i32}}",
        "  }",
        "}",
        "",
    ]
    return "\n".join(lines), blocks, name


def check_forced_cases():
    """Regressions specific enough that leaving them to the random generator
    would be a coin flip; run every time instead."""
    problems = []
    cfg = DEVICES[0]  # "core": data_size only applies where there's a core
    mlir, blocks, _ = zero_size_reserved_data_case(cfg)
    placed = allocate(mlir)
    if placed is None:
        problems.append(
            "zero_size_reserved_data: allocator rejected a provably-fitting design"
        )
    else:
        problems.extend(
            f"zero_size_reserved_data: {b}"
            for b in legality_violations(cfg, blocks, placed)
        )
    return problems


def stress_design(cfg, n_buffers):
    """Many small buffers on one tile, to catch a real compile-time blow-up
    in a per-buffer scan that costs O(tile size) -- not exercised by the
    random designs above, which cap out at 14 buffers per tile."""
    lines = [
        "module {",
        f'  aie.device({cfg["dev"]}) {{',
        f'    %t = aie.tile({cfg["tile"][0]}, {cfg["tile"][1]})',
    ]
    for i in range(n_buffers):
        lines.append(
            f'    %b{i} = aie.buffer(%t) {{sym_name = "b{i}"}} : memref<16xi8>'
        )
    if cfg["stack"]:
        lines.append(
            f'    aie.core(%t) {{ aie.end }} {{stack_size = {cfg["stack"]} : i32}}'
        )
    else:
        lines.append("    aie.memtile_dma(%t) { aie.end }")
    lines += ["  }", "}", ""]
    return "\n".join(lines)


SEARCH_STAT_RE = re.compile(r"\(S\)\s+(\d+)\s+tiles-needing-search")


def allocate(mlir, stats=None):
    """Run the pass; returns {name: (addr, size, bank)} or None.

    Pass a dict as `stats` to also collect the pass's own search counters, which
    say how much backtracking a design actually cost.
    """
    cmd = ["aie-opt", "--aie-assign-buffer-addresses", "-"]
    if stats is not None:
        cmd.append("--mlir-pass-statistics")
    p = subprocess.run(cmd, input=mlir, capture_output=True, text=True)
    if stats is not None:
        m = SEARCH_STAT_RE.search(p.stderr)
        stats["tiles_searched"] = int(m.group(1)) if m else 0
    if p.returncode != 0:
        return None
    placed = {}
    for line in p.stdout.splitlines():
        m = BUF_RE.search(line)
        if not m:
            continue
        attrs, size = m.group(1), int(m.group(2))
        name = re.search(r'sym_name = "([^"]+)"', attrs)
        addr = re.search(r"address = (\d+)", attrs)
        mb = re.search(r"mem_bank = (\d+)", attrs)
        if name and addr:
            placed[name.group(1)] = (
                int(addr.group(1)),
                size,
                int(mb.group(1)) if mb else None,
            )
    return placed


MODULE_TAG_RE = re.compile(r"^module @s(\d+)\b")


def allocate_batch(tagged, stats=None):
    """Place many tagged designs in one aie-opt run.

    `tagged` is [(tag, mlir)] where each mlir names its module @s<tag>. Returns
    {tag: placed} for the designs that placed; a design aie-opt rejects is
    simply absent, because --split-input-file omits a chunk it could not
    process. Splitting the work this way is what keeps a corpus of thousands
    affordable: nearly all of the per-design cost is process startup.
    """
    cmd = ["aie-opt", "--split-input-file", "--aie-assign-buffer-addresses", "-"]
    if stats is not None:
        cmd.append("--mlir-pass-statistics")
    p = subprocess.run(
        cmd,
        input="\n// -----\n".join(m for _, m in tagged),
        capture_output=True,
        text=True,
    )
    if stats is not None:
        # One statistics report per chunk, so sum them rather than reading the
        # first.
        stats["tiles_searched"] = stats.get("tiles_searched", 0) + sum(
            int(n) for n in SEARCH_STAT_RE.findall(p.stderr)
        )
    results, current = {}, None
    for line in p.stdout.splitlines():
        tag = MODULE_TAG_RE.match(line)
        if tag:
            current = int(tag.group(1))
            results[current] = {}
            continue
        if current is None:
            continue
        m = BUF_RE.search(line)
        if not m:
            continue
        attrs, size = m.group(1), int(m.group(2))
        name = re.search(r'sym_name = "([^"]+)"', attrs)
        addr = re.search(r"address = (\d+)", attrs)
        mb = re.search(r"mem_bank = (\d+)", attrs)
        if name and addr:
            results[current][name.group(1)] = (
                int(addr.group(1)),
                size,
                int(mb.group(1)) if mb else None,
            )
    return results


def legality_violations(cfg, blocks, placed, stack_run=None):
    cap, bus = cfg["cap"], cfg["bus"]
    # The stack sits at offset zero unless a design placed it elsewhere, in
    # which case low memory is ordinary free space and the stack is a hole
    # somewhere above it.
    stack_lo, stack_hi = stack_run if stack_run else (0, cfg["stack"])
    stack = stack_hi
    bank = cap // cfg["banks"]
    out = []
    by_name = {b["name"]: b for b in blocks}
    for b in blocks:
        if b["name"] not in placed:
            out.append(f'{b["name"]} was not placed')
    for name, (addr, size, _) in placed.items():
        spec = by_name[name]
        if size == 0:  # covers no bytes; cannot collide with anything
            continue
        if addr < 0 or addr + size > cap:
            out.append(f"{name} outside tile memory at {addr}+{size}")
        if addr < stack:
            out.append(f"{name} overlaps the stack at {addr}")
        if spec["aligned"] and addr % bus:
            out.append(f"{name} misaligned at {addr}")
        if spec["role"] == "pin" and addr != spec["addr"]:
            out.append(f'{name} pinned at {spec["addr"]} but placed at {addr}')
        if spec["role"] == "bank":
            want = spec["addr"] // bank
            if addr < want * bank or addr + size > (want + 1) * bank:
                out.append(f"{name} left mem_bank {want}")
    spans = sorted((a, a + s, n) for n, (a, s, _) in placed.items() if s)
    for (a1, e1, n1), (a2, e2, n2) in zip(spans, spans[1:]):
        if a2 < e1:
            out.append(f"{n1} [{a1},{e1}) overlaps {n2} [{a2},{e2})")
    return out


def largest_free_run(cfg, placed):
    """(start of largest free run, its size, total free bytes) above the stack.

    The run's start is aligned to `vec`, matching largestFreeRun: the linker
    starts the core's .data at a multiple of its strongest section alignment, so
    an unaligned origin loses the difference to padding. Ties go to the lowest
    address, matching that function's `>` comparison, so this run compares
    against the region the allocator recorded."""
    cap, stack, vec = cfg["cap"], cfg["stack"], cfg["vec"]
    taken = sorted((a, a + s) for a, s, _ in placed.values() if s)
    start = best = total = 0
    cursor = stack

    def consider(lo, hi):
        nonlocal start, best
        aligned = align_up(lo, vec)
        if aligned < hi and hi - aligned > best:
            start, best = aligned, hi - aligned

    for lo, hi in taken:
        if lo > cursor:
            consider(cursor, lo)
            total += lo - cursor
        cursor = max(cursor, hi)
    if cursor < cap:
        consider(cursor, cap)
        total += cap - cursor
    return start, best, total


def bank_pair_sharing(cfg, placed, core_data=None):
    """Fraction of buffer pairs on the tile that share a bank, where 0 is the
    most even split the bank count allows and 1 is every buffer in one bank.

    Contention is pairwise: two buffers a kernel reads together serialize when
    they share a bank. Byte spread across banks does not see that, because two
    buffers can share one bank in a layout whose banks hold equal bytes. The
    data region stays out of this: it is one extent, and its size follows from
    where the buffers land rather than from a request of its own."""
    bank = cfg["cap"] // cfg["banks"]
    per_bank = defaultdict(int)
    total = 0
    for name, (_, size, bk) in placed.items():
        if size == 0 or name == core_data:
            continue
        per_bank[bk] += 1
        total += 1
    if total < 2:
        return None

    def pairs(count):
        return count * (count - 1) // 2

    observed = sum(pairs(c) for c in per_bank.values())
    full, spare = divmod(total, cfg["banks"])
    fewest = spare * pairs(full + 1) + (cfg["banks"] - spare) * pairs(full)
    most = pairs(total)
    if most == fewest:
        return None
    return (observed - fewest) / (most - fewest)


def quality(cfg, blocks, placed):
    bank = cfg["cap"] // cfg["banks"]
    by_name = {b["name"]: b for b in blocks}
    needless, per_bank = 0, defaultdict(int)

    def charge_to_banks(addr, size):
        first, last = addr // bank, (addr + size - 1) // bank
        for bk in range(first, last + 1):
            lo, hi = max(addr, bk * bank), min(addr + size, (bk + 1) * bank)
            per_bank[bk] += hi - lo
        return first, last

    spans = sorted((a, a + s) for a, s, _ in placed.values() if s)
    for name, (addr, size, _) in placed.items():
        if size == 0:
            continue
        first, last = charge_to_banks(addr, size)
        if by_name[name]["role"] != "free" or last == first:
            continue
        # A crossing counts only when some bank had a run this buffer fits.
        # Pinned neighbors can leave every bank too fragmented, and then the
        # crossing is forced rather than chosen.
        others = [sp for sp in spans if sp != (addr, addr + size)]
        for bk in range(cfg["banks"]):
            lo, hi = bk * bank, (bk + 1) * bank
            cursor = max(lo, cfg["stack"])
            for a, e in others:
                if e <= cursor or a >= hi:
                    continue
                if a - cursor >= size:
                    break
                cursor = max(cursor, e)
            else:
                if hi - cursor < size:
                    continue
            needless += 1
            break
    return needless


def check_metrics():
    """Answers the metrics owe on layouts whose answer is known by hand.

    A metric that drifts turns every bound below it into a number about
    nothing, and the suite still prints OK. These layouts pin the answers."""
    cfg = dict(cap=1024, banks=4, bus=4, vec=64, stack=64)
    bank = cfg["cap"] // cfg["banks"]  # 256
    out = []

    def expect(label, got, want):
        if got != want:
            out.append(f"{label}: got {got}, want {want}")

    # Free runs: [64,256) after the stack, [512,768) between the two buffers,
    # and [896,1024) above them. The middle run is the largest, and its start
    # rounds up from 512 to the next multiple of vec, which 512 already is.
    placed = {"a": (256, 256, 1), "b": (768, 128, 3)}
    expect("largest_free_run", largest_free_run(cfg, placed), (512, 256, 576))
    # An unaligned end loses the run's first bytes to padding: the run after a
    # buffer ending at 200 starts at 256, so it is 256 bytes long, not 312.
    expect(
        "largest_free_run aligns the start",
        largest_free_run(cfg, {"a": (64, 136, 0), "b": (512, 512, 2)}),
        (256, 256, 312),
    )
    # A tile with no buffers is one run from the stack to the top.
    expect(
        "largest_free_run on an empty tile", largest_free_run(cfg, {}), (64, 960, 960)
    )

    # Sharing needs at least two buffers, and it needs a spread of layouts to
    # separate: one buffer per bank is the floor, all four in one bank the roof.
    one_each = {n: (i * bank, 16, i) for i, n in enumerate("abcd")}
    expect("bank_pair_sharing at the floor", bank_pair_sharing(cfg, one_each), 0.0)
    one_bank = {n: (i * 16, 16, 0) for i, n in enumerate("abcd")}
    expect("bank_pair_sharing at the roof", bank_pair_sharing(cfg, one_bank), 1.0)
    # 3 of 4 in one bank: 3 observed pairs of the 6 a single bank would give,
    # against a floor of 0 that one buffer per bank reaches.
    lopsided = {"a": (0, 16, 0), "b": (16, 16, 0), "c": (32, 16, 0), "d": (bank, 16, 1)}
    expect("bank_pair_sharing between them", bank_pair_sharing(cfg, lopsided), 0.5)
    # A zero-sized buffer covers no bytes, so it contends with nothing. Naming
    # a buffer as the core's data excludes it for the same reason.
    with_extras = dict(one_each, e=(0, 0, 0), core_data_0_2=(512, 128, 2))
    expect(
        "bank_pair_sharing skips empty and core data",
        bank_pair_sharing(cfg, with_extras, "core_data_0_2"),
        0.0,
    )

    # Crossings. `blocks` carries the role, and only a free buffer can choose
    # where it lands.
    def blocks_for(placed, role="free"):
        return [dict(name=n, role=role) for n in placed]

    inside = {"a": (0, 128, 0)}
    expect(
        "quality on a buffer inside a bank", quality(cfg, blocks_for(inside), inside), 0
    )
    # Straddles [128,384) while banks 2 and 3 stand empty, so it had a choice.
    avoidable = {"a": (128, 256, 0)}
    expect(
        "quality on an avoidable crossing",
        quality(cfg, blocks_for(avoidable), avoidable),
        1,
    )
    # A 224-byte buffer straddling banks 0 and 1. The stack leaves bank 0 with
    # 192 bytes, a pin leaves bank 1 with 128, and pins fill banks 2 and 3, so
    # no bank holds it and the crossing is the only placement left.
    forced = {
        "a": (160, 224, 0),
        "p1": (384, 128, 1),
        "p2": (512, 256, 2),
        "p3": (768, 256, 3),
    }
    roles = blocks_for(forced)
    for b in roles:
        if b["name"] != "a":
            b["role"] = "pin"
    expect("quality on a forced crossing", quality(cfg, roles, forced), 0)
    return out


def main(argv=None):
    cli = argparse.ArgumentParser()
    cli.add_argument("--seeds", type=int, default=SEEDS)
    # Seeds count up from 0, so a larger corpus contains the default one and
    # MIN_SEARCHED_TILES still holds.
    cli.add_argument("--bankres-seeds", type=int, default=BANKRES_SEEDS)
    cli.add_argument("--jobs", type=int, default=4, help="parallel aie-opt runs")
    args = cli.parse_args(argv)
    pool = ThreadPoolExecutor(args.jobs)

    designs = []
    for seed in range(args.seeds):
        cfg = DEVICES[seed % len(DEVICES)]
        mlir, blocks, core_data = build_design(random.Random(seed), cfg)
        if mlir is not None:
            designs.append((seed, cfg, mlir, blocks, core_data))
    total = len(designs)
    solved = needless = nondet = 0
    sharing, illegal, unsolved = [], [], []
    runs = pool.map(lambda d: (allocate(d[2]), allocate(d[2])), designs)
    for (seed, cfg, _, blocks, core_data), (placed, again) in zip(designs, runs):
        if placed is None:
            unsolved.append(f"seed {seed} ({cfg['name']})")
            continue
        solved += 1
        bad = legality_violations(cfg, blocks, placed)
        if bad:
            illegal.append(f"seed {seed} ({cfg['name']}): {bad[0]}")
        needless += quality(cfg, blocks, placed)
        share = bank_pair_sharing(cfg, placed, core_data)
        if share is not None:
            sharing.append(share)
        if again != placed:
            nondet += 1

    # Object-derived per-bank reservations, kept on their own seeds so the
    # metrics above stay comparable with their recorded bounds.
    bankres_solved = bankres_total = 0
    bankres_illegal, bankres_bogus, bankres_unsolved = [], [], []
    bankres_designs = []
    for seed in range(args.bankres_seeds):
        cfg = DEVICES[seed % len(DEVICES)]
        mlir, blocks, extra = build_bank_reservation_design(
            random.Random(seed), cfg, tag=seed
        )
        if mlir is None:
            continue
        faults = oracle_problems(cfg, blocks)
        if faults:
            bankres_bogus.append(f"seed {seed} ({cfg['name']}): {faults[0]}")
            continue
        bankres_total += 1
        bankres_designs.append((seed, cfg, blocks, mlir))

    # One aie-opt run per chunk of designs. Sized so a failure still points
    # at a manageable slice of input, while keeping startup cost negligible.
    def place(group):
        stats = {}
        results = allocate_batch([(seed, mlir) for seed, _, _, mlir in group], stats)
        return results, stats["tiles_searched"]

    groups = [
        bankres_designs[start : start + BATCH_SIZE]
        for start in range(0, len(bankres_designs), BATCH_SIZE)
    ]
    bankres_searched = 0
    for group, (results, searched) in zip(groups, pool.map(place, groups)):
        bankres_searched += searched
        for seed, cfg, blocks, _ in group:
            placed = results.get(seed)
            if placed is None:
                bankres_unsolved.append(f"seed {seed} ({cfg['name']})")
                continue
            bankres_solved += 1
            bad = legality_violations(cfg, blocks, placed)
            if bad:
                bankres_illegal.append(f"seed {seed} ({cfg['name']}): {bad[0]}")
    pool.shutdown()

    # A metric with no samples across solved designs means the property was
    # never exercised. Report that instead of passing it by default.
    share_has_data = bool(sharing) or solved == 0
    mean_share = sum(sharing) / len(sharing) if sharing else 0.0
    for line in illegal[:10]:
        print("ILLEGAL:", line)

    regressions = []

    def report(label, ok, detail):
        print(f"{label}: {detail} : {'OK' if ok else 'REGRESSION'}")
        if not ok:
            regressions.append(label)

    metric_problems = check_metrics()
    for line in metric_problems:
        print("WRONG METRIC:", line)
    report(
        "metric-validation",
        not metric_problems,
        f"{len(metric_problems)} wrong answer(s)",
    )
    report("legality", not illegal, f"{len(illegal)} illegal of {solved} placed")
    report("determinism", nondet == 0, f"{nondet} unstable")
    for line in unsolved[:10]:
        print("UNSOLVED:", line)
    report("completeness", solved >= args.seeds, f"{solved}/{total} solved")
    for line in bankres_illegal[:10]:
        print("ILLEGAL:", line)
    for line in bankres_bogus[:10]:
        print("BOGUS ORACLE:", line)
    for line in bankres_unsolved[:10]:
        print("UNSOLVED:", line)
    bankres_rate = bankres_solved / bankres_total if bankres_total else 0.0
    report(
        "bank-reservations",
        bankres_rate >= MIN_BANKRES_RATE and not bankres_illegal and not bankres_bogus,
        f"{bankres_solved}/{bankres_total} solved ({bankres_rate:.1%}, "
        f"min {MIN_BANKRES_RATE:.0%}), "
        f"{len(bankres_illegal)} illegal, "
        f"{len(bankres_bogus)} unsolvable-by-construction",
    )
    report(
        "search-exercised",
        bankres_searched >= MIN_SEARCHED_TILES,
        f"{bankres_searched} tile(s) needed backtracking "
        f"(min {MIN_SEARCHED_TILES})",
    )
    report(
        "bank-crossings",
        needless <= MAX_CROSSINGS,
        f"{needless} avoidable (max {MAX_CROSSINGS})",
    )
    report(
        "bank-sharing",
        share_has_data and mean_share <= MAX_BANK_PAIR_SHARING,
        (
            f"mean shared pairs {mean_share:.3f} (max {MAX_BANK_PAIR_SHARING})"
            if share_has_data
            else "0 samples despite solved designs"
        ),
    )

    forced_problems = check_forced_cases()
    for line in forced_problems:
        print("ILLEGAL:", line)
    report(
        "forced-regressions",
        not forced_problems,
        f"{len(forced_problems)} problem(s)",
    )

    stress_cfg = DEVICES[1]  # memtile: 512 KB, plenty of room for 300 tiny buffers
    stress_mlir = stress_design(stress_cfg, STRESS_BUFFERS)
    t0 = time.monotonic()
    stress_placed = allocate(stress_mlir)
    stress_elapsed = time.monotonic() - t0
    report(
        "stress",
        stress_placed is not None and stress_elapsed <= STRESS_MAX_SECONDS,
        f"{len(stress_placed) if stress_placed else 0}/{STRESS_BUFFERS} placed "
        f"in {stress_elapsed:.1f}s (max {STRESS_MAX_SECONDS}s)",
    )
    return 1 if regressions else 0


# Every property must report OK; any REGRESSION fails the test.
# CHECK: metric-validation: {{.*}} : OK
# CHECK: legality: {{.*}} : OK
# CHECK: determinism: {{.*}} : OK
# CHECK: completeness: {{.*}} : OK
# CHECK: bank-reservations: {{.*}} : OK
# CHECK: search-exercised: {{.*}} : OK
# CHECK: bank-crossings: {{.*}} : OK
# CHECK: bank-sharing: {{.*}} : OK
# CHECK: forced-regressions: {{.*}} : OK
# CHECK: stress: {{.*}} : OK
# CHECK-NOT: REGRESSION

if __name__ == "__main__":
    sys.exit(main())
