#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# RUN: %python %s | FileCheck %s

# Property-based regression test for --aie-assign-buffer-addresses.
#
# The individual .mlir tests in this directory pin exact addresses, which pins
# down behaviour but says nothing about whether the allocator is any *good*.
# This checks the properties that matter across many generated designs:
#
#   legality      -- placements never overlap, escape the tile, land in the
#                    stack, break alignment, move a pinned address, leave the
#                    bank a design asked for, or (when reserved_data_size was
#                    set) leave too little contiguous room for it despite the
#                    allocator reporting success
#   completeness  -- a design that provably fits is actually placed
#   determinism   -- the same input twice gives the same addresses
#   quality       -- buffers spread over banks, and a buffer that fits inside
#                    one bank is not needlessly split across two
#
# Feasibility comes from a construct-then-hide oracle: a valid layout is built
# first, then a random subset of it is hidden behind `address` / `mem_bank` /
# nothing. The constructed layout is by definition a solution, so a failure to
# allocate is unambiguously an allocator bug and not an impossible input.
#
# Completeness and quality are ratchets rather than absolutes: packing around
# fixed obstacles is NP-hard, so the allocator is a heuristic and a handful of
# adversarial layouts are expected to defeat it. Tighten the bounds when it
# improves; a drop below them is a regression.

import random
import re
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

SEEDS = 200
MIN_SOLVED = 200  # of SEEDS; seeds are fixed, so this is exact
MAX_NEEDLESS_CROSSINGS = 2
MAX_BANK_IMBALANCE = 0.70  # mean (max-min)/bankSize over banks in use
# Fragmentation: the fraction of the free bytes that survive as ONE run. The
# generated linker script hands the core compiler a single region -- the
# largest gap left between the stack and the buffers -- so this, not the total
# free space, is what decides whether a core's own .data/.bss fit. 1.0 would
# mean perfectly contiguous.
MIN_MEAN_CONTIGUITY = 0.55

DEVICES = [
    dict(name="core", dev="npu2", tile=(0, 2), cap=65536, banks=4, bus=32, stack=1024),
    dict(name="memtile", dev="npu2", tile=(0, 1), cap=524288, banks=8, bus=4, stack=0),
]

BUF_RE = re.compile(r"aie\.buffer.*?\{(.*?)\}\s*:\s*memref<(\d+)xi8>")


def align_up(v, a):
    return ((v + a - 1) // a) * a


def oracle_largest_free_run(cap, stack, blocks):
    """Largest contiguous free run above the stack in the as-constructed
    (pre-hide) oracle layout -- used to cap a randomly chosen
    reserved_data_size to something the oracle's own placement can support."""
    taken = sorted((b["addr"], b["addr"] + b["size"]) for b in blocks if b["size"])
    best, cursor = 0, stack
    for lo, hi in taken:
        if lo > cursor:
            best = max(best, lo - cursor)
        cursor = max(cursor, hi)
    return max(best, cap - cursor)


def build_design(rng, cfg):
    """A random valid layout, then hide part of it.

    Returns (mlir, blocks, reserved): `reserved` is the reserved_data_size
    (bytes) requested on the core, or 0 if none (only ever set for a "core"
    config -- reserved_data_size is a CoreOp attribute, and the memtile config
    has no aie.core at all).
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
            # mem_bank is a hard constraint, so only request it when the block
            # really does lie inside one bank, else the design is infeasible.
            b["role"] = "bank"
        else:
            b["role"] = "free"

    # reserved_data_size: exercise the allocator's tight-packing strategy and
    # its reservation acceptance test, only feasible on a config with a core.
    # Capped to a fraction of the *oracle's own* largest contiguous gap (not
    # total free space) so the construct-then-hide guarantee keeps holding:
    # the real allocator reorders blocks (largest-first, then by strategy),
    # so it is not guaranteed to reproduce the oracle's exact packing, but it
    # should get close -- leaving margin below the oracle's own number keeps
    # this a legality question (is the allocator's own bookkeeping honoured?)
    # rather than a fresh feasibility question this generator cannot answer.
    reserved = 0
    if cfg["stack"] and rng.random() < 0.35:
        oracle_run = oracle_largest_free_run(cap, stack, blocks)
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


def allocate(mlir, workdir):
    """Run the pass; returns {name: (addr, size, bank)} or None if it failed."""
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
    """(largest contiguous free run, total free bytes) above the stack."""
    cap, stack = cfg["cap"], cfg["stack"]
    taken = sorted((a, a + s) for a, s, _ in placed.values() if s)
    best = total = 0
    cursor = stack
    for lo, hi in taken:
        if lo > cursor:
            best = max(best, lo - cursor)
            total += lo - cursor
        cursor = max(cursor, hi)
    if cursor < cap:
        best = max(best, cap - cursor)
        total += cap - cursor
    return best, total


def quality(cfg, blocks, placed):
    bank = cfg["cap"] // cfg["banks"]
    by_name = {b["name"]: b for b in blocks}
    needless, per_bank = 0, defaultdict(int)
    for name, (addr, size, _) in placed.items():
        if size == 0:
            continue
        first, last = addr // bank, (addr + size - 1) // bank
        # A buffer no larger than a bank never has to be split across two.
        if by_name[name]["role"] == "free" and size <= bank and last != first:
            needless += 1
        for bk in range(first, last + 1):
            lo, hi = max(addr, bk * bank), min(addr + size, (bk + 1) * bank)
            per_bank[bk] += hi - lo
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
            placed = allocate(mlir, workdir)
            if placed is None:
                continue
            solved += 1
            bad = legality_violations(cfg, blocks, placed)
            run, free = largest_free_run(cfg, placed)
            # The allocator reported success, so its own reservation
            # acceptance test claims this run is big enough; if it isn't,
            # that check is broken, not just this one design.
            if reserved and run < reserved:
                bad.append(
                    f"reserved_data_size {reserved} not honored, largest "
                    f"contiguous run is only {run}"
                )
            if bad:
                illegal.append(f"seed {seed} ({cfg['name']}): {bad[0]}")
            n, imb = quality(cfg, blocks, placed)
            needless += n
            imbalances.append(imb)
            if free:
                contiguity.append(run / free)
            if allocate(mlir, workdir) != placed:
                nondet += 1

        # A metric that never accumulated any samples despite solving designs
        # is suspicious on its own -- silently defaulting it to "pass" would
        # hide that the property was never actually exercised.
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
        return 0


# Every property must report OK; any REGRESSION fails the test.
# CHECK: legality: {{.*}} : OK
# CHECK: determinism: {{.*}} : OK
# CHECK: completeness: {{.*}} : OK
# CHECK: bank-splitting: {{.*}} : OK
# CHECK: bank-balance: {{.*}} : OK
# CHECK: contiguity: {{.*}} : OK
# CHECK-NOT: REGRESSION

if __name__ == "__main__":
    sys.exit(main())
