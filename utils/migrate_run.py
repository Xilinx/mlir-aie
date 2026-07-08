#!/usr/bin/env python3
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Rewrite legacy aie.dma_bd op text to the declarative DynamicIndexList form,
injecting arith.constant defs for offset/len into the enclosing block.

Usage: migrate_run.py <file.mlir> [--in-place]
Prints the migrated text to stdout (or rewrites in place).

Strategy:
  * Rewrite each `aie.dma_bd(...)` occurrence (balanced-paren scan).
  * For constant offset/len, reference a per-block SSA constant %c<val>_i32,
    and remember to inject `%c<val>_i32 = arith.constant <val> : i32` once per
    enclosing block (the block that directly contains the dma_start / task).
  * Lines that are FileCheck directives (contain 'CHECK') are left ALONE here;
    CHECK regeneration is a separate step for files that check dma_bd syntax.
"""
import re
import sys

from migrate_dma_bd import find_dma_bd_calls, split_top_level


def parse_legacy_body(body):
    """body is the text inside aie.dma_bd(...). Returns dict or None if it's
    already new-style / unrecognized."""
    # New-style already? (contains 'sizes' or 'offset =')
    if re.search(r"\b(sizes|offset)\s*=", body):
        return None
    # buffer ':' type , [offset [, len [, dims [, paddims [, pad_value=..]]]]]
    parts = split_top_level(body)
    head = parts[0]  # "%buf : memref<...>"
    m = re.match(r"\s*(%[^:]+?)\s*:\s*(.+?)\s*$", head, re.S)
    if not m:
        return None
    buf, ty = m.group(1).strip(), m.group(2).strip()
    rest = [p.strip() for p in parts[1:]]
    offset = length = dims = pad = pad_value = None
    idx = 0
    # offset (integer literal)
    if idx < len(rest) and re.fullmatch(r"-?\d+", rest[idx]):
        offset = int(rest[idx]); idx += 1
    if idx < len(rest) and re.fullmatch(r"-?\d+", rest[idx]):
        length = int(rest[idx]); idx += 1
    if idx < len(rest) and rest[idx].startswith("["):
        dims = rest[idx]; idx += 1
    if idx < len(rest) and rest[idx].startswith("["):
        pad = rest[idx]; idx += 1
    if idx < len(rest) and rest[idx].startswith("pad_value"):
        pad_value = rest[idx]; idx += 1
    if idx != len(rest):
        return None  # unrecognized trailing content
    return dict(buf=buf, ty=ty, offset=offset, length=length,
                dims=dims, pad=pad, pad_value=pad_value)


def dims_to_lists(dims):
    """'[<size = 2, stride = 1>, <size = 3, stride = 2>]' -> ([2,3],[1,2])."""
    if not dims:
        return [], []
    inner = dims.strip()[1:-1].strip()
    if not inner:
        return [], []
    sizes, strides = [], []
    for tup in split_top_level(inner):
        ms = re.search(r"size\s*=\s*(-?\d+)", tup)
        mt = re.search(r"stride\s*=\s*(-?\d+)", tup)
        sizes.append(int(ms.group(1)))
        strides.append(int(mt.group(1)))
    return sizes, strides


def main():
    path = sys.argv[1]
    text = open(path).read()
    consts_needed = set()

    def convert(body):
        info = parse_legacy_body(body)
        if info is None:
            return None
        sizes, strides = dims_to_lists(info["dims"])
        out = f'{info["buf"]} : {info["ty"]}'
        if info["offset"] is not None:
            consts_needed.add(info["offset"])
            out += f' offset = %c{info["offset"]}_i32'
        if info["length"] is not None:
            consts_needed.add(info["length"])
            out += f' len = %c{info["length"]}_i32'
        out += f' sizes = [{", ".join(map(str, sizes))}]'
        out += f' strides = [{", ".join(map(str, strides))}]'
        if info["pad"]:
            out += f' pad {info["pad"]}'
        if info["pad_value"]:
            out += f' {info["pad_value"]}'
        return f"aie.dma_bd({out})"

    # Pass 1: rewrite dma_bd op text (right-to-left). Collect which constant
    # values are needed per enclosing region entry point. Skip CHECK lines.
    # region_entry_consts maps char-offset-of-region-open -> set(int values)
    from collections import defaultdict
    region_entry_consts = defaultdict(set)

    calls = find_dma_bd_calls(text)
    rewrites = []  # (start, end, new_text) to apply right-to-left
    for start, end, whole in calls:
        line_start = text.rfind("\n", 0, start) + 1
        line = text[line_start:start]
        if "CHECK" in line or "//" in line:
            continue
        body = whole[len("aie.dma_bd("):-1]
        before = set(consts_needed)
        new = convert(body)
        if new is None:
            sys.stderr.write(f"SKIP (unrecognized): {whole[:80]}\n")
            continue
        new_here = consts_needed - before
        # Find the innermost enclosing region-open brace (last '{' before start
        # that isn't inside a string/comment — good enough heuristic).
        region_pos = text.rfind("{", 0, start)
        if region_pos != -1 and new_here:
            region_entry_consts[region_pos] |= new_here
        rewrites.append((start, end, new))

    # Apply rewrites right-to-left so positions stay valid.
    for start, end, new in sorted(rewrites, reverse=True):
        text = text[:start] + new + text[end:]

    # Pass 2: for each region open brace, inject the needed constants on the
    # next line (right after '{'), using the brace's line indentation + 2 spaces.
    # We recalculate positions since text changed in pass 1.
    # Approach: collect insertion points as (char_offset_in_original, snippet)
    # then do them right-to-left on the already-pass-1-rewritten text.
    # Re-search the braces by scanning for the same content patterns.
    # Simpler: inject by scanning the final text for region openers that
    # directly precede the first ^bb or dma_bd in the body, using regex.
    # Strategy: insert constants right after each '{' that was a region entry.
    # We use the line-indentation of the '{' line + 4 spaces for the constant.
    if region_entry_consts:
        # Rebuild: find all '{' positions in the new text and inject.
        # Since pass 1 rewrites don't change the number of '{' chars (dma_bd
        # has none), positions shift predictably. Use a fresh scan on the
        # updated text: find region-open braces by context.
        # Simple heuristic: inject at every '{' that appears at end-of-line
        # (possibly with trailing space/comment) and belongs to an aie/aiex op.
        region_open_pat = re.compile(
            r'((?:aie|aiex)\.\w+(?:\s+@\w+)?\s*\([^)]*\))\s*\{')
        # Build the full set of all needed constants (any region gets all of
        # them; DCE removes unused ones, and MLIR allows unused constants).
        all_consts = set()
        for s in region_entry_consts.values():
            all_consts |= s
        const_snippet = lambda indent: "".join(
            f"\n{indent}  %c{v}_i32 = arith.constant {v} : i32"
            for v in sorted(all_consts)
        )

        def do_inject(m):
            # Indentation of the line containing the match.
            sol = text.rfind("\n", 0, m.start())
            line_text = text[sol + 1:m.start()]
            indent = re.match(r"[ \t]*", line_text).group(0)
            return m.group(0) + const_snippet(indent)

        # Only inject into regions that actually had dma_bds needing consts.
        # We do a single global substitution into all aie/aiex region openers
        # in the file; unused constants are harmless.
        text = region_open_pat.sub(do_inject, text)

    print(text, end="")
    sys.stderr.write(f"constants needed: {sorted(consts_needed)}\n")


if __name__ == "__main__":
    main()
