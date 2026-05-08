#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2026, Advanced Micro Devices, Inc.
"""Emit a graphviz DOT diagram for an IRON design.

Walks a list of `Worker` objects (the design tree), collects ObjectFifos via
their fn_args, and renders:

  - one node per Worker (labeled with tile coords + first kernel name)
  - one edge per ObjectFifo (labeled with element shape and depth)
  - subgraphs grouping workers by tile column (visual spatial layout)
  - colors: shim (row 0) blue, memtile (row 1) green, compute orange

Run via the --dot flag on aie2_mobilenet_iron.py:
  python3 aie2_mobilenet_iron.py --dot | dot -Tpng -o /tmp/m.png
"""

import sys
from collections import defaultdict


# ---------------------------------------------------------------------------
# Style helpers
# ---------------------------------------------------------------------------
def _tile_color(tile):
    """Color a node by its row (shim row 0 = blue, mem row 1 = green, compute = orange)."""
    if tile.row == 0:
        return "lightblue"
    if tile.row == 1:
        return "palegreen"
    return "navajowhite"


def _kernel_name(arg):
    """Return the user-facing kernel name if `arg` looks like a Kernel, else None.

    Kernel exposes both `_name` and `_arg_types`; Buffer only has `_name` (not
    `_arg_types`) — that distinguishes the two without hard-importing aie.iron.
    """
    if not hasattr(arg, "_arg_types"):
        return None
    n = getattr(arg, "_name", None)
    return n if isinstance(n, str) else None


def _worker_label(worker):
    """Short label: tile coords + first kernel name + while_true marker."""
    t = worker._tile
    coord = f"({t.col},{t.row})"
    kernels = [_kernel_name(a) for a in worker.fn_args if _kernel_name(a) is not None]
    if kernels:
        # Strip leading bn-prefix to keep the label compact (already grouped visually).
        label = kernels[0]
        for prefix in ("bn0_", "bn1_", "bn10_", "bn11_", "bn12_", "bn13_", "bn14_"):
            if label.startswith(prefix):
                label = label[len(prefix) :]
                break
        return f"{coord}\\n{label}"
    return coord


def _fifo_handles(worker):
    """Yield (handle, fifo) for each ObjectFifoHandle in worker.fn_args."""
    for arg in worker.fn_args:
        of = getattr(arg, "_object_fifo", None)
        if of is not None:
            yield arg, of


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def emit_dot(workers, out=sys.stdout):
    """Render the design as graphviz DOT to `out`.

    `workers` is a list of IRON Worker objects (e.g. what aie2_mobilenet_iron's
    `mobilenet_iron(collect_only=True)` returns).
    """
    # 1. Map each fifo to (producer_workers, consumer_workers) by walking handles.
    fifo_endpoints = defaultdict(lambda: {"prod": [], "cons": []})
    seen_fifos = {}  # id(fifo) -> fifo
    for w in workers:
        for handle, fifo in _fifo_handles(w):
            seen_fifos[id(fifo)] = fifo
            side = "prod" if handle._is_prod else "cons"
            fifo_endpoints[id(fifo)][side].append(w)

    # 2. Group workers by tile column (visual layout).
    by_col = defaultdict(list)
    for w in workers:
        by_col[w._tile.col].append(w)

    # 3. Emit DOT.
    print("digraph mobilenet {", file=out)
    print("  rankdir=TB;", file=out)
    print("  graph [splines=ortho, nodesep=0.4, ranksep=0.5];", file=out)
    print(
        '  node  [shape=box, style="filled,rounded", fontname="Helvetica"];', file=out
    )
    print('  edge  [fontname="Helvetica", fontsize=9];', file=out)
    print(file=out)

    for col in sorted(by_col):
        print(f"  subgraph cluster_col{col} {{", file=out)
        print(f'    label="col {col}"; style=dotted; fontsize=10;', file=out)
        # Sort within column by row (top to bottom = row 0 → row N).
        for w in sorted(by_col[col], key=lambda w: w._tile.row):
            label = _worker_label(w)
            color = _tile_color(w._tile)
            print(f'    "w{id(w)}" [label="{label}", fillcolor={color}];', file=out)
        print("  }", file=out)

    print(file=out)

    # 4. Emit edges. One edge per (producer, consumer) pair per fifo.
    for fid, ends in fifo_endpoints.items():
        fifo = seen_fifos[fid]
        shape = "x".join(str(d) for d in fifo.shape)
        depth = fifo.depth if isinstance(fifo.depth, int) else fifo.depth[0]
        edge_label = f'"{shape}\\nd={depth}"'
        for prod in ends["prod"]:
            for cons in ends["cons"]:
                if prod is cons:
                    continue  # self-loop fifo (e.g. bn12_dw_tmp); skip noise
                print(
                    f'  "w{id(prod)}" -> "w{id(cons)}" [label={edge_label}];',
                    file=out,
                )

    print("}", file=out)


# ---------------------------------------------------------------------------
# Standalone test entry: import aie2_mobilenet_iron and emit
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import aie2_mobilenet_iron as m

    workers = m.mobilenet_iron(collect_only=True)
    emit_dot(workers)
