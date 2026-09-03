#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# RUN: %python %s | FileCheck %s

# Property-based test for --aie-assign-buffer-addresses.
#
# The individual .mlir tests in this directory pin exact addresses, which fixes
# behaviour on a handful of designs. This file measures legality, completeness,
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
MAX_NEEDLESS_CROSSINGS = 2
# Mean (max-min)/bankSize over banks in use. Counts the core's data region
# alongside the buffers, because the region occupies banks the way a buffer
# does; scoring the banks it covers as empty reports imbalance for a balanced
# layout. Measured at 0.50.
MAX_BANK_IMBALANCE = 0.60
# Fragmentation: the fraction of the free bytes that survive as ONE run. The
# generated linker script grants the core compiler a single region, the largest
# gap between the stack and the buffers, so this fraction, and not the total
# free space, decides whether a core's own .data and .bss fit. 1.0 means
# perfectly contiguous. Measured over the core configs, where the region exists;
# placement scores it directly (see the candidate ranking in AIEAssignBuffers).
# Measured at 0.80.
MIN_MEAN_CONTIGUITY = 0.75

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
    (pre-hide) oracle layout. Caps a randomly chosen reserved_data_size to a
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

    Returns (mlir, blocks, reserved): `reserved` is the reserved_data_size
    (bytes) requested on the core, or 0 for none. Only a "core" config sets it,
    because reserved_data_size is a CoreOp attribute and the memtile config has
    no aie.core.
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

    # reserved_data_size exercises the allocator's tight packing and its
    # reservation check. Only a config with a core supports it. The value is
    # capped to a fraction of the *oracle's own* largest contiguous gap, not of
    # the total free space, which keeps the construct-then-hide guarantee: the
    # allocator reorders blocks (bank-pinned first, then largest first), so it
    # does not reproduce the oracle's exact packing. The margin below the
    # oracle's number keeps this a legality question, and not a feasibility
    # question this generator cannot answer.
    reserved = 0
    if cfg["stack"] and rng.random() < 0.35:
        oracle_run = oracle_largest_free_run(cap, stack, blocks, cfg["vec"])
        if oracle_run > 1:
            reserved = rng.randint(1, max(1, int(oracle_run * 0.7)))

    lines = [
        "module {",
        f'  aie.device({cfg["dev"]}) {{',
        f'    %t = aie.tile({cfg["tile"][0]}, {cfg["tile"][1]})',
    ]
    for b in blocks:
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
        if reserved:
            core_attrs += f", reserved_data_size = {reserved} : i32"
        lines.append(f"    aie.core(%t) {{ aie.end }} {{{core_attrs}}}")
    else:
        lines.append("    aie.memtile_dma(%t) { aie.end }")
    lines += ["  }", "}", ""]
    return "\n".join(lines), blocks, reserved


STRESS_BUFFERS = 300
STRESS_MAX_SECONDS = 30


def zero_size_reserved_data_case(cfg):
    """A permanent regression guard, independent of chance: a zero-sized
    buffer pinned strictly inside the tile's only free run must not
    fragment it (a zero-length occupied interval is not a split point)."""
    cap, bus, stack = cfg["cap"], cfg["bus"], cfg["stack"]
    addr = align_up(stack + bus, bus) + bus * 100
    blocks = [dict(addr=addr, size=0, aligned=True, name="mid", role="pin")]
    reserved = cap - stack
    lines = [
        "module {",
        f'  aie.device({cfg["dev"]}) {{',
        f'    %t = aie.tile({cfg["tile"][0]}, {cfg["tile"][1]})',
        f'    %mid = aie.buffer(%t) {{sym_name = "mid", address = {addr} : i32}} : memref<0xi8>',
        f"    aie.core(%t) {{ aie.end }} "
        f"{{stack_size = {stack} : i32, reserved_data_size = {reserved} : i32}}",
        "  }",
        "}",
        "",
    ]
    return "\n".join(lines), blocks, reserved


def check_forced_cases(workdir):
    """Regressions specific enough that leaving them to the random generator
    would be a coin flip; run every time instead."""
    problems = []
    cfg = DEVICES[0]  # "core": reserved_data_size only applies where there's a core
    mlir, blocks, reserved = zero_size_reserved_data_case(cfg)
    result = allocate(mlir, workdir)
    if result is None:
        problems.append(
            "zero_size_reserved_data: allocator rejected a provably-fitting design"
        )
    else:
        placed, region = result
        bad = legality_violations(cfg, blocks, placed)
        bad += region_violations(cfg, placed, region, reserved)
        problems.extend(f"zero_size_reserved_data: {b}" for b in bad)
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
    """Run the pass; returns ({name: (addr, size, bank)}, region) or None.

    `region` is the (origin, length) the allocator recorded on aie.core for the
    core's own sections, or None on a tile with no core."""
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
    region = None
    for line in p.stdout.splitlines():
        # The core's attribute dict prints on the line closing its body, so this
        # matches it on its own line.
        origin = re.search(r"data_origin = (\d+)", line)
        length = re.search(r"data_length = (\d+)", line)
        if origin and length:
            region = (int(origin.group(1)), int(length.group(1)))
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
    return placed, region


def region_violations(cfg, placed, region, reserved):
    """The data region is a real placed object now, not a number measured after
    the fact, so it gets checked like one: it must satisfy the request, sit
    where nothing else does, and match what the linker-script emitter would
    derive on its own -- that last one is what keeps the emitter's fallback
    path honest."""
    if region is None:
        return []
    origin, length = region
    out = []
    if reserved and length < reserved:
        out.append(
            f"granted data region is {length} bytes, smaller than the "
            f"requested reserved_data_size {reserved}"
        )
    # A zero-length region covers no bytes, so it cannot sit anywhere illegal.
    if length:
        if origin < cfg["stack"]:
            out.append(f"data region at {origin} starts inside the stack")
        if origin + length > cfg["cap"]:
            out.append(f"data region {origin}+{length} runs past the tile")
        for name, (addr, size, _) in placed.items():
            if size and addr < origin + length and origin < addr + size:
                out.append(f"data region {origin}+{length} overlaps {name}")
                break
    start, best, _ = largest_free_run(cfg, placed)
    if (origin, length) != (start, best):
        out.append(
            f"data region {origin}+{length} is not the largest free run "
            f"{start}+{best}; the linker script's fallback would disagree"
        )
    return out


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


def quality(cfg, blocks, placed, region):
    bank = cfg["cap"] // cfg["banks"]
    by_name = {b["name"]: b for b in blocks}
    needless, per_bank = 0, defaultdict(int)

    def charge_to_banks(addr, size):
        first, last = addr // bank, (addr + size - 1) // bank
        for bk in range(first, last + 1):
            lo, hi = max(addr, bk * bank), min(addr + size, (bk + 1) * bank)
            per_bank[bk] += hi - lo
        return first, last

    for name, (addr, size, _) in placed.items():
        if size == 0:
            continue
        first, last = charge_to_banks(addr, size)
        # A buffer no larger than a bank fits inside one bank.
        if by_name[name]["role"] == "free" and size <= bank and last != first:
            needless += 1
    # The core's data region occupies banks the way a buffer does. Omitting it
    # scores the banks it covers as empty and reports imbalance for a balanced
    # layout.
    if region and region[1]:
        charge_to_banks(region[0], region[1])
    used = [per_bank.get(i, 0) for i in range(cfg["banks"])]
    return needless, (max(used) - min(used)) / bank


def main():
    with tempfile.TemporaryDirectory(prefix="aie-alloc-props-") as workdir_str:
        workdir = Path(workdir_str)
        solved = total = needless = nondet = 0
        imbalances, contiguity, illegal = [], [], []
        for seed in range(SEEDS):
            cfg = DEVICES[seed % len(DEVICES)]
            mlir, blocks, reserved = build_design(random.Random(seed), cfg)
            if mlir is None:
                continue
            total += 1
            result = allocate(mlir, workdir)
            if result is None:
                continue
            placed, region = result
            solved += 1
            bad = legality_violations(cfg, blocks, placed)
            # The allocator reported success, so the region it recorded is the
            # region the core receives. Check that region directly.
            bad += region_violations(cfg, placed, region, reserved)
            if bad:
                illegal.append(f"seed {seed} ({cfg['name']}): {bad[0]}")
            _, run, free = largest_free_run(cfg, placed)
            n, imb = quality(cfg, blocks, placed, region)
            needless += n
            imbalances.append(imb)
            # Core tiles only: the region holds a core's own sections, so
            # placement optimizes for it on a core tile and spreads across banks
            # on a memtile. One average over both would mix two objectives.
            if free and cfg["stack"]:
                contiguity.append(run / free)
            if allocate(mlir, workdir) != result:
                nondet += 1

        # A metric with no samples across solved designs means the property was
        # never exercised. Report that instead of passing it by default.
        imb_has_data = bool(imbalances) or solved == 0
        contig_has_data = bool(contiguity) or solved == 0
        mean_imb = sum(imbalances) / len(imbalances) if imbalances else 0.0
        mean_contig = sum(contiguity) / len(contiguity) if contiguity else 0.0
        for line in illegal[:10]:
            print("ILLEGAL:", line)

        def report(label, ok, detail):
            print(f"{label}: {detail} : {'OK' if ok else 'REGRESSION'}")

        report("legality", not illegal, f"{len(illegal)} illegal of {solved} placed")
        report("determinism", nondet == 0, f"{nondet} unstable")
        report("completeness", solved >= MIN_SOLVED, f"{solved}/{total} solved")
        report(
            "bank-splitting",
            needless <= MAX_NEEDLESS_CROSSINGS,
            f"{needless} needless (max {MAX_NEEDLESS_CROSSINGS})",
        )
        report(
            "bank-balance",
            imb_has_data and mean_imb <= MAX_BANK_IMBALANCE,
            (
                f"mean imbalance {mean_imb:.2f} (max {MAX_BANK_IMBALANCE})"
                if imb_has_data
                else "0 samples despite solved designs"
            ),
        )
        report(
            "contiguity",
            contig_has_data and mean_contig >= MIN_MEAN_CONTIGUITY,
            (
                f"mean largest-run/free {mean_contig:.2f} (min {MIN_MEAN_CONTIGUITY})"
                if contig_has_data
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
        stress_result = allocate(stress_mlir, workdir)
        stress_elapsed = time.monotonic() - t0
        stress_placed = stress_result[0] if stress_result else None
        report(
            "stress",
            stress_placed is not None and stress_elapsed <= STRESS_MAX_SECONDS,
            f"{len(stress_placed) if stress_placed else 0}/{STRESS_BUFFERS} placed "
            f"in {stress_elapsed:.1f}s (max {STRESS_MAX_SECONDS}s)",
        )
        return 0


# Every property must report OK; any REGRESSION fails the test.
# CHECK: legality: {{.*}} : OK
# CHECK: determinism: {{.*}} : OK
# CHECK: completeness: {{.*}} : OK
# CHECK: bank-splitting: {{.*}} : OK
# CHECK: bank-balance: {{.*}} : OK
# CHECK: contiguity: {{.*}} : OK
# CHECK: forced-regressions: {{.*}} : OK
# CHECK: stress: {{.*}} : OK
# CHECK-NOT: REGRESSION

if __name__ == "__main__":
    sys.exit(main())
