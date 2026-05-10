#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2026, Advanced Micro Devices, Inc.
"""Emit a graphviz DOT diagram for an IRON design.

Walks a list of `Worker` objects (the design tree) and collects three kinds
of inter-tile connection:

  - ObjectFifos via worker.fn_args (._object_fifo on each handle)
  - CascadeFlows via worker._outgoing_cascades
  - StaticWeightStream-like objects via worker.fn_args
    (duck-typed by the presence of _memtile + _compute attrs)

Renders:

  - an underlay grid of every tile in the device (faint, white-filled,
    dashed-gray border) — only when a `device` is passed to `emit_dot`
  - one overlay node per AIE tile that the design uses, pinned to its
    (col, row) chip coordinates (shim row 0 at the bottom, memtile row 1
    above, compute on top)
  - ObjectFifo edges: solid black, label = element shape and depth
  - CascadeFlow edges: dashed red, label = "cascade"
  - StaticWeightStream edges: dotted blue, label = "static W: <name>"
    (with an extra hop through the ping-pong memtile if one is configured)
  - colors: shim (row 0) blue, memtile (row 1) green, compute orange

Layout uses the `neato` engine with hard-pinned positions so the diagram
matches the chip floorplan. The DOT file sets `layout=neato`, so either
`neato -Tpng` or `dot -Tpng` will pick up the right engine.

Run via the --dot flag on aie2_mobilenet_iron.py:
  python3 aie2_mobilenet_iron.py --dot | neato -Tpng -o /tmp/m.png
"""

import sys
from collections import defaultdict

# Spatial layout scales (graphviz inches per tile column/row).
SCALE_X = 1.5
SCALE_Y = 1.2


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


def _worker_kernel(worker):
    """Compact kernel-name string for `worker`, or None if it has no kernel arg."""
    kernels = [_kernel_name(a) for a in worker.fn_args if _kernel_name(a) is not None]
    if not kernels:
        return None
    label = kernels[0]
    for prefix in ("bn0_", "bn1_", "bn10_", "bn11_", "bn12_", "bn13_", "bn14_"):
        if label.startswith(prefix):
            label = label[len(prefix) :]
            break
    return label


def _node_id(tile):
    return f"t{tile.col}_{tile.row}"


def _tile_pos(tile):
    # AIE row 0 (shim) is at the bottom of the chip; graphviz Y also grows up,
    # so row → y is a direct mapping (no flip needed).
    return f"{tile.col * SCALE_X},{tile.row * SCALE_Y}!"


def _fifo_handles(worker):
    """Yield (handle, fifo) for each ObjectFifoHandle in worker.fn_args."""
    for arg in worker.fn_args:
        of = getattr(arg, "_object_fifo", None)
        if of is not None:
            yield arg, of


def _static_weight_streams(worker):
    """Yield StaticWeightStream-like objects in worker.fn_args (duck-typed)."""
    for arg in worker.fn_args:
        if hasattr(arg, "_memtile") and hasattr(arg, "_compute"):
            yield arg


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def emit_dot(workers, out=sys.stdout, device=None):
    """Render the design as graphviz DOT to `out`.

    `workers` is a list of IRON Worker objects (e.g. what aie2_mobilenet_iron's
    `mobilenet_iron(collect_only=True)` returns).

    `device`, if provided, is an IRON Device (e.g. NPU2()). When set, every
    tile in the device's (cols x rows) grid is emitted as a faint underlay
    node, with the design's worker tiles drawn on top. Without `device`, only
    the design's tiles appear.
    """
    # 1. Map each fifo to (producer_workers, consumer_workers) by walking handles.
    fifo_endpoints = defaultdict(lambda: {"prod": [], "cons": []})
    seen_fifos = {}  # id(fifo) -> fifo
    for w in workers:
        for handle, fifo in _fifo_handles(w):
            seen_fifos[id(fifo)] = fifo
            side = "prod" if handle._is_prod else "cons"
            fifo_endpoints[id(fifo)][side].append(w)

    # 2. Dedup StaticWeightStreams by identity (one stream may be passed to
    #    several workers' fn_args).
    static_streams = {}
    for w in workers:
        for s in _static_weight_streams(w):
            static_streams[id(s)] = s

    # 3. Collect every tile we need to render: workers' tiles, plus any
    #    memtiles referenced only by static-weight streams.
    tiles_by_coord = {}  # (col,row) -> tile object
    workers_by_coord = defaultdict(list)
    for w in workers:
        coord = (w._tile.col, w._tile.row)
        tiles_by_coord[coord] = w._tile
        workers_by_coord[coord].append(w)
    for s in static_streams.values():
        for t in (s._memtile, s._compute, s._ping_pong_memtile):
            if t is None:
                continue
            tiles_by_coord[(t.col, t.row)] = t

    # 4. Emit DOT header.
    print("digraph mobilenet {", file=out)
    print("  layout=neato;", file=out)
    print("  graph [overlap=false, splines=true];", file=out)
    print(
        '  node  [shape=box, style="filled,rounded", fontname="Helvetica"];', file=out
    )
    print('  edge  [fontname="Helvetica", fontsize=9];', file=out)
    print(file=out)

    # 5. Underlay: faint placeholder for every tile in the device grid that
    #    the design doesn't use. Skipped entirely when no device is provided.
    if device is not None:
        for col in range(device.cols):
            for row in range(device.rows):
                if (col, row) in tiles_by_coord:
                    continue
                pos = f"{col * SCALE_X},{row * SCALE_Y}!"
                print(
                    f'  "t{col}_{row}" [label="({col},{row})", '
                    f"fillcolor=white, color=gray70, fontcolor=gray60, "
                    f'fontsize=8, style="dashed,filled,rounded", pos="{pos}"];',
                    file=out,
                )

    # 6. Overlay: one node per design tile, pinned to chip coordinates.
    for coord, tile in sorted(tiles_by_coord.items()):
        kernels = [
            k for k in (_worker_kernel(w) for w in workers_by_coord.get(coord, [])) if k
        ]
        coord_str = f"({tile.col},{tile.row})"
        label = coord_str + ("\\n" + "\\n".join(kernels) if kernels else "")
        color = _tile_color(tile)
        pos = _tile_pos(tile)
        print(
            f'  "{_node_id(tile)}" [label="{label}", fillcolor={color}, pos="{pos}"];',
            file=out,
        )

    print(file=out)

    # 7. ObjectFifo edges (one per producer/consumer pair per fifo).
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
                    f'  "{_node_id(prod._tile)}" -> "{_node_id(cons._tile)}" '
                    f"[label={edge_label}];",
                    file=out,
                )

    # 8. CascadeFlow edges (dashed red).
    for w in workers:
        for cf in getattr(w, "_outgoing_cascades", []):
            print(
                f'  "{_node_id(cf._src._tile)}" -> "{_node_id(cf._dst._tile)}" '
                f'[label="cascade", style=dashed, color=red];',
                file=out,
            )

    # 9. StaticWeightStream edges (dotted blue). Ping-pong memtile, if present,
    #    is rendered as an extra hop (memtile -> pp_memtile -> compute).
    for s in static_streams.values():
        name = getattr(s, "_name", "") or ""
        edge_label = f'"static W: {name}"'
        edge_attrs = f"[label={edge_label}, style=dotted, color=blue]"
        if s._ping_pong_memtile is not None:
            print(
                f'  "{_node_id(s._memtile)}" -> '
                f'"{_node_id(s._ping_pong_memtile)}" {edge_attrs};',
                file=out,
            )
            print(
                f'  "{_node_id(s._ping_pong_memtile)}" -> '
                f'"{_node_id(s._compute)}" {edge_attrs};',
                file=out,
            )
        else:
            print(
                f'  "{_node_id(s._memtile)}" -> '
                f'"{_node_id(s._compute)}" {edge_attrs};',
                file=out,
            )

    print("}", file=out)


# ---------------------------------------------------------------------------
# Standalone test entry: import aie2_mobilenet_iron and emit
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import aie2_mobilenet_iron as m

    workers, device = m.mobilenet_iron(collect_only=True)
    emit_dot(workers, device=device)
