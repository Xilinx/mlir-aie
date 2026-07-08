#!/usr/bin/env python3
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Migrate aie.dma_bd from the legacy positional syntax to the declarative
DynamicIndexList form.

Legacy:  aie.dma_bd(%b : memref<256xi32>, 0, 256, [<size = 16, stride = 1>])
New:     aie.dma_bd(%b : memref<256xi32> offset = %c0_i32 len = %c256_i32
                    sizes = [16] strides = [1])

offset/len are now SSA operands (arith.constant for constants); sizes/strides
are a mixed static/dynamic index list. This script rewrites the op text and,
for constant offset/len, records which constants must exist so the caller can
inject them. It does NOT touch CHECK lines (those are regenerated from real
aie-opt output by a separate step).

Only the op-text rewrite lives here; it is deliberately conservative and skips
any dma_bd whose offset/len are already SSA values (%...) or whose form it does
not recognize, reporting them for manual handling.
"""
import re
import sys


# A balanced-paren scan from the '(' after 'aie.dma_bd', so memref types that
# themselves contain commas/brackets (e.g. memref<64xi32, 2>) don't confuse a
# naive split.
def find_dma_bd_calls(text):
    calls = []
    i = 0
    needle = "aie.dma_bd("
    while True:
        start = text.find(needle, i)
        if start == -1:
            break
        # scan balanced () from the opening paren
        depth = 0
        j = start + len(needle) - 1
        while j < len(text):
            c = text[j]
            if c == "(":
                depth += 1
            elif c == ")":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        calls.append((start, j + 1, text[start : j + 1]))
        i = j + 1
    return calls


def split_top_level(s, sep=","):
    """Split on `sep` at bracket/angle/paren depth 0."""
    parts = []
    depth = 0
    cur = []
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


if __name__ == "__main__":
    print("migration helper module; invoked by migrate_run.py", file=sys.stderr)
