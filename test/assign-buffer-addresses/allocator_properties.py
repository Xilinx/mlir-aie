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

import random
import re
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

SEEDS = 200
# All 200 fixed seeds solve at present. Completeness is a ratchet, like the
# quality metrics below (see the file-level comment). The slack keeps a future
# adversarial seed that defeats the heuristic a tracked gap.
MIN_SOLVED = SEEDS - 2
MAX_CROSSINGS = 0
# Co-residency: how many buffer pairs share a bank, against the fewest the bank
# count allows. Two buffers a kernel reads together serialize on one bank, and
# byte spread across banks does not report that, so this measures it directly.
# 0 is the most even split available, 1 puts every buffer in one bank.
MAX_BANK_PAIR_SHARING = 0.30

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


def check_forced_cases(workdir):
    """Regressions specific enough that leaving them to the random generator
    would be a coin flip; run every time instead."""
    problems = []
    cfg = DEVICES[0]  # "core": data_size only applies where there's a core
    mlir, blocks, _ = zero_size_reserved_data_case(cfg)
    placed = allocate(mlir, workdir)
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


def allocate(mlir, workdir):
    """Run the pass; returns {name: (addr, size, bank)} or None."""
    src = workdir / "case.mlir"
    src.write_text(mlir)
    p = subprocess.run(
        ["aie-opt", "--aie-assign-buffer-addresses=alloc-scheme=bank-aware", str(src)],
        capture_output=True,
        text=True,
    )
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


def legality_violations(cfg, blocks, placed):
    cap, bus, stack = cfg["cap"], cfg["bus"], cfg["stack"]
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


def main():
    with tempfile.TemporaryDirectory(prefix="aie-alloc-props-") as workdir_str:
        workdir = Path(workdir_str)
        solved = total = needless = nondet = 0
        sharing, illegal = [], []
        for seed in range(SEEDS):
            cfg = DEVICES[seed % len(DEVICES)]
            mlir, blocks, core_data = build_design(random.Random(seed), cfg)
            if mlir is None:
                continue
            total += 1
            placed = allocate(mlir, workdir)
            if placed is None:
                continue
            solved += 1
            bad = legality_violations(cfg, blocks, placed)
            if bad:
                illegal.append(f"seed {seed} ({cfg['name']}): {bad[0]}")
            needless += quality(cfg, blocks, placed)
            share = bank_pair_sharing(cfg, placed, core_data)
            if share is not None:
                sharing.append(share)
            if allocate(mlir, workdir) != placed:
                nondet += 1

        # A metric with no samples across solved designs means the property was
        # never exercised. Report that instead of passing it by default.
        share_has_data = bool(sharing) or solved == 0
        mean_share = sum(sharing) / len(sharing) if sharing else 0.0
        for line in illegal[:10]:
            print("ILLEGAL:", line)

        def report(label, ok, detail):
            print(f"{label}: {detail} : {'OK' if ok else 'REGRESSION'}")

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
        report("completeness", solved >= MIN_SOLVED, f"{solved}/{total} solved")
        report(
            "bank-crossings",
            needless <= MAX_CROSSINGS,
            f"{needless} avoidable (max {MAX_CROSSINGS})",
        )
        report(
            "bank-sharing",
            share_has_data and mean_share <= MAX_BANK_PAIR_SHARING,
            (
                f"mean shared pairs {mean_share:.2f} (max {MAX_BANK_PAIR_SHARING})"
                if share_has_data
                else "0 samples despite solved designs"
            ),
        )

        forced_problems = check_forced_cases(workdir)
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
        stress_placed = allocate(stress_mlir, workdir)
        stress_elapsed = time.monotonic() - t0
        report(
            "stress",
            stress_placed is not None and stress_elapsed <= STRESS_MAX_SECONDS,
            f"{len(stress_placed) if stress_placed else 0}/{STRESS_BUFFERS} placed "
            f"in {stress_elapsed:.1f}s (max {STRESS_MAX_SECONDS}s)",
        )
        return 0


# Every property must report OK; any REGRESSION fails the test.
# CHECK: metric-validation: {{.*}} : OK
# CHECK: legality: {{.*}} : OK
# CHECK: determinism: {{.*}} : OK
# CHECK: completeness: {{.*}} : OK
# CHECK: bank-crossings: {{.*}} : OK
# CHECK: bank-sharing: {{.*}} : OK
# CHECK: forced-regressions: {{.*}} : OK
# CHECK: stress: {{.*}} : OK
# CHECK-NOT: REGRESSION

if __name__ == "__main__":
    sys.exit(main())
