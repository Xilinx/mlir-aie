#!/usr/bin/env python3
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""
Migrate aie.dma_bd from legacy positional syntax to declarative DynamicIndexList.

Old: aie.dma_bd(%buf : memref<256xi32>, 0, 256, [<size = 2, stride = 1>])
New: aie.dma_bd(%buf : memref<256xi32> offset = %c0_i32 len = %c256_i32
                sizes = [2] strides = [1])

Algorithm:
  1. Parse the file into regions (balanced-brace tree).
  2. For each region body, find every aie.dma_bd call (balanced-paren scan).
  3. Collect the integer constants needed for offset/len in that region.
  4. Inject `%c{v}_i32 = arith.constant {v} : i32` once per region, right
     after the opening `{` (before any ^bb labels or ops). Constants are
     deduplicated per region.
  5. Rewrite each dma_bd call in-place.
  6. Leave CHECK lines untouched — the caller handles those separately.
"""

import re
import sys
from collections import defaultdict


# ---------------------------------------------------------------------------
# Balanced-paren scan: find all aie.dma_bd(...)  in a string
# ---------------------------------------------------------------------------
def find_dma_bd_calls(text, start=0, end=None):
    """Return list of (call_start, call_end) for every aie.dma_bd(...) span."""
    if end is None:
        end = len(text)
    results = []
    needle = "aie.dma_bd("
    i = start
    while i < end:
        pos = text.find(needle, i, end)
        if pos == -1:
            break
        depth = 0
        j = pos + len(needle) - 1
        while j < end:
            if text[j] == "(":
                depth += 1
            elif text[j] == ")":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        results.append((pos, j + 1))
        i = j + 1
    return results


def split_top_level(s, sep=","):
    """Split string on sep at bracket/angle/paren depth 0."""
    parts, cur, depth = [], [], 0
    for c in s:
        if c in "<[(":
            depth += 1
        elif c in ">])":
            depth -= 1
        if c == sep and depth == 0:
            parts.append("".join(cur))
            cur = []
        else:
            cur.append(c)
    parts.append("".join(cur))
    return parts


def on_check_line(text, pos):
    """True if pos falls on a line that is a FileCheck directive."""
    sol = text.rfind("\n", 0, pos) + 1
    line = text[sol:pos]
    return "CHECK" in line or (line.lstrip().startswith("//") and "CHECK" in line)


# ---------------------------------------------------------------------------
# Parse one dma_bd call body → new text + set of integer constants needed
# ---------------------------------------------------------------------------
def parse_and_convert(body):
    """
    body: text inside aie.dma_bd(...), exclusive of the parens.
    Returns (new_call_text, {int_values_needed}) or (None, set()) if unrecognized.
    """
    # Already new-style?
    if re.search(r"\b(sizes|offset)\s*=", body):
        return None, set()

    parts = split_top_level(body)
    # First part: "%buf : memref<...>"
    m = re.match(r"\s*(%[^\s:]+)\s*:\s*(.+?)\s*$", parts[0], re.S)
    if not m:
        return None, set()
    buf, ty = m.group(1).strip(), m.group(2).strip()

    rest = [p.strip() for p in parts[1:]]
    offset = length = dims_text = pad_text = pad_value_text = None
    idx = 0

    # offset: integer literal only (not %val — those are already new-style)
    if idx < len(rest) and re.fullmatch(r"-?\d+", rest[idx]):
        offset = int(rest[idx])
        idx += 1
    # len
    if idx < len(rest) and re.fullmatch(r"-?\d+", rest[idx]):
        length = int(rest[idx])
        idx += 1
    # dimensions: [...] attribute form
    if idx < len(rest) and rest[idx].startswith("["):
        dims_text = rest[idx]
        idx += 1
    # pad_dimensions: [...] attribute form
    if idx < len(rest) and rest[idx].startswith("["):
        pad_text = rest[idx]
        idx += 1
    # pad_value = N
    if idx < len(rest) and rest[idx].startswith("pad_value"):
        pad_value_text = rest[idx]
        idx += 1
    if idx != len(rest):
        return None, set()

    # Decompose dimensions into sizes/strides lists
    sizes, strides = [], []
    if dims_text:
        inner = dims_text.strip()[1:-1].strip()
        if inner:
            for tup in split_top_level(inner):
                ms = re.search(r"size\s*=\s*(-?\d+)", tup)
                mt = re.search(r"stride\s*=\s*(-?\d+)", tup)
                if not (ms and mt):
                    return None, set()
                sizes.append(int(ms.group(1)))
                strides.append(int(mt.group(1)))

    needed = set()
    out = f"{buf} : {ty}"
    if offset is not None:
        needed.add(offset)
        out += f" offset = %c{offset}_i32"
    if length is not None:
        needed.add(length)
        out += f" len = %c{length}_i32"
    out += f" sizes = [{', '.join(map(str, sizes))}]"
    out += f" strides = [{', '.join(map(str, strides))}]"
    if pad_text:
        out += f" pad {pad_text}"
    if pad_value_text:
        out += f" {pad_value_text}"

    return f"aie.dma_bd({out})", needed


# ---------------------------------------------------------------------------
# Find all region bodies in the file (balanced-brace segments)
# Each region is (open_brace_pos, close_brace_pos).
# We find every '{' that follows an aie/aiex op opener, then scan for the
# matching '}'. We use a simple scan rather than full parsing.
# ---------------------------------------------------------------------------
REGION_OPEN_PAT = re.compile(
    r"(?:aie|aiex)\.\w+(?:\s+@\w+)?"  # op keyword
    r"(?:\s*<[^>]*>)?"  # optional generic params
    r"\s*\([^)]*\)"  # argument list
    r"\s*\{"  # opening brace
)


def find_region_bodies(text):
    """
    Return list of (open_brace_pos, close_brace_pos) for every aie/aiex region.
    open_brace_pos is the index of '{'; close_brace_pos is the index of '}'.
    """
    regions = []
    for m in REGION_OPEN_PAT.finditer(text):
        brace = m.end() - 1  # index of '{'
        # scan for matching '}'
        depth = 0
        j = brace
        while j < len(text):
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        regions.append((brace, j))
    return regions


def indent_of_line(text, pos):
    """Return the leading whitespace of the line containing pos."""
    sol = text.rfind("\n", 0, pos) + 1
    m = re.match(r"[ \t]*", text[sol:])
    return m.group(0) if m else ""


# ---------------------------------------------------------------------------
# Main migration: two-pass (collect then inject)
# ---------------------------------------------------------------------------
def migrate_file(text):
    skipped = []

    # Pass 1: scan every region, collect rewrites and constants needed.
    # We process regions smallest-first so nested regions are handled before
    # their parents (inner regions' dma_bds won't be re-processed by parents).
    regions = find_region_bodies(text)
    # sort by size ascending so inner regions come first
    regions.sort(key=lambda r: r[1] - r[0])

    # Map: region open-brace pos -> list of (call_start, call_end, new_text)
    region_rewrites = defaultdict(list)
    region_consts = defaultdict(set)

    # Track which call spans have been claimed by an inner region already.
    claimed = set()

    for rbrace, rclose in regions:
        rbody_start = rbrace + 1
        rbody_end = rclose

        calls = find_dma_bd_calls(text, rbody_start, rbody_end)
        for cstart, cend in calls:
            if (cstart, cend) in claimed:
                continue
            if on_check_line(text, cstart):
                continue
            body = text[cstart + len("aie.dma_bd(") : cend - 1]
            new_text, needed = parse_and_convert(body)
            if new_text is None:
                if not re.search(r"\b(sizes|offset)\s*=", body):
                    skipped.append(text[cstart:cend][:80])
                continue
            region_rewrites[rbrace].append((cstart, cend, new_text))
            region_consts[rbrace] |= needed
            claimed.add((cstart, cend))

    # Pass 2: apply rewrites right-to-left across ALL regions together,
    # and inject constants right after each region's opening brace.
    # Collect all (pos, replacement) pairs and apply right-to-left.

    # Build insertion list: for each region that needs constants, insert
    # constant definitions right after the '{' (on a new line).
    insertions = []  # (pos, text_to_insert)
    for rbrace, consts in region_consts.items():
        if not consts:
            continue
        # Determine indentation from the line containing '{'
        ind = indent_of_line(text, rbrace) + "  "
        snippet = "".join(
            f"\n{ind}%c{v}_i32 = arith.constant {v} : i32" for v in sorted(consts)
        )
        insertions.append((rbrace + 1, snippet))  # insert after '{'

    # Combine all rewrites and insertions, sort right-to-left
    ops = []
    for rbrace, calls in region_rewrites.items():
        for cstart, cend, new_text in calls:
            ops.append((cstart, cend, new_text))
    for pos, snippet in insertions:
        ops.append((pos, pos, snippet))  # zero-width insert

    ops.sort(key=lambda x: x[0], reverse=True)

    for start, end, replacement in ops:
        text = text[:start] + replacement + text[end:]

    return text, skipped


def main():
    args = sys.argv[1:]
    in_place = "--in-place" in args
    paths = [a for a in args if not a.startswith("-")]
    if not paths:
        print("Usage: migrate_dma_bd_full.py [--in-place] <file.mlir>", file=sys.stderr)
        sys.exit(1)
    path = paths[0]
    text = open(path).read()
    new_text, skipped = migrate_file(text)
    if skipped:
        for s in skipped:
            sys.stderr.write(f"SKIP: {s}\n")
    if in_place:
        open(path, "w").write(new_text)
    else:
        print(new_text, end="")


if __name__ == "__main__":
    main()
