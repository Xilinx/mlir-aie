# tiletrace.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc.
"""TileTrace: declarative trace configuration attached to a single tile.

A ``TileTrace`` describes *what to trace on one tile*. It is attached to
whatever owns the tile:

* to a :class:`Worker` via ``Worker(..., trace=TileTrace(...))`` -- the tile is
  the worker's compute core, and the ``tile`` field is ignored;
* to a Program via ``Program(..., trace_tiles=[TileTrace(tile=..., ...)])`` for
  non-worker (mem tile / shim tile) tracing, where ``tile`` is required.

A compute (core) tile exposes **two independent trace units** with disjoint
event vocabularies: the *core* unit (``CoreEvent.*``) and the *core-memory*
unit (``MemEvent.*``). A single ``TileTrace`` may target either or both: the
unit each event belongs to is inferred from the event's type, so a list mixing
``CoreEvent`` and ``MemEvent`` values is automatically split across the two
units. Mem tiles and shim tiles each have a single unit (``MemTileEvent`` /
``ShimTileEvent``).
"""

from __future__ import annotations

from .events import GenericEvent

# Map a trace event to the hardware trace unit it belongs to. The event enum
# classes are arch-specific (e.g. CoreEventAIE2, CoreEventAIE2P) and share no
# common per-unit base, so the unit is identified by the class-name prefix.
# The four prefixes are mutually non-overlapping (e.g. "MemTileEvent" does not
# start with "MemEvent"), so a single startswith match is unambiguous.
# Returns one of: "core", "mem", "memtile", "shim".
_UNIT_BY_PREFIX = (
    ("CoreEvent", "core"),
    ("MemTileEvent", "memtile"),
    ("ShimTileEvent", "shim"),
    ("MemEvent", "mem"),
)


def _event_unit(event) -> str:
    """Return the trace unit ('core'/'mem'/'memtile'/'shim') for an event.

    Accepts a bare event enum value or a GenericEvent/PortEvent wrapper (whose
    ``.code`` carries the underlying enum value).
    """
    code = event.code if isinstance(event, GenericEvent) else event
    cls_name = type(code).__name__
    for prefix, unit in _UNIT_BY_PREFIX:
        if cls_name.startswith(prefix):
            return unit
    raise ValueError(
        f"Cannot determine trace unit for event {event!r} (type "
        f"{cls_name}); expected a CoreEvent, MemEvent, MemTileEvent, or "
        f"ShimTileEvent value."
    )


# The trace units physically available on each tile class.
_UNITS_FOR_TILE = {
    "core": ("core", "mem"),
    "mem": ("memtile",),
    "shim": ("shim",),
}


def _tile_class(tile_op) -> str:
    """Return the tile class key ('core'/'mem'/'shim') for a resolved tile op."""
    if tile_op.is_core_tile():
        return "core"
    if tile_op.is_mem_tile():
        return "mem"
    if tile_op.is_shim_tile():
        return "shim"
    raise ValueError(f"Cannot trace tile {tile_op}: unknown tile class")


class TileTrace:
    """Trace configuration for a single tile.

    Args:
        events: Trace events to capture. Each event's type selects the hardware
            unit it is programmed into (``CoreEvent`` -> core, ``MemEvent`` ->
            core-memory, ``MemTileEvent`` -> mem tile, ``ShimTileEvent`` -> shim
            tile). ``None`` (the default) captures a sensible default event set
            for every unit the tile supports.
        tile: The tile to trace. Required only for non-worker tracing via
            ``Program(trace_tiles=...)``; ignored when the ``TileTrace`` is
            attached to a :class:`Worker` (the worker supplies its core tile).
    """

    def __init__(self, events: list | None = None, tile=None):
        self.events = events
        self.tile = tile

    def units(self) -> dict[str, list]:
        """Group this trace's explicit events by hardware unit.

        Returns a mapping of unit name -> list of events. Empty when ``events``
        is ``None`` (defaults are filled in later, against the resolved tile).
        """
        grouped: dict[str, list] = {}
        if self.events is None:
            return grouped
        for event in self.events:
            grouped.setdefault(_event_unit(event), []).append(event)
        return grouped

    def specs_for(self, tile_op) -> list[tuple]:
        """Resolve to ``(tile_op, unit, events)`` specs for a placed tile.

        With explicit events, emits one spec per hardware unit the events name
        (validated against the tile's available units). With ``events=None``,
        emits one spec per unit the tile supports, each with ``None`` events so
        the lowering fills in that unit's defaults.
        """
        available = _UNITS_FOR_TILE[_tile_class(tile_op)]
        if self.events is None:
            return [(tile_op, unit, None) for unit in available]

        specs = []
        for unit, events in self.units().items():
            if unit not in available:
                raise ValueError(
                    f"Trace events for the '{unit}' unit were given for tile "
                    f"{tile_op}, which only has units {available}. "
                    f"(Event type determines the unit; check the event classes.)"
                )
            specs.append((tile_op, unit, events))
        return specs


def build_trace_specs(worker_traces, trace_tiles) -> list[tuple]:
    """Build the flat ``(tile_op, unit, events)`` spec list for a program.

    Args:
        worker_traces: Iterable of ``(tile_op, TileTrace)`` for each traced
            worker (the tile is the worker's resolved compute tile op).
        trace_tiles: Iterable of :class:`TileTrace`, each carrying its own
            resolved ``tile`` op, for non-worker (mem/shim) tracing.

    Returns:
        A list of ``(tile_op, unit, events)`` specs suitable for
        :func:`configure_trace_specs`.
    """
    specs: list[tuple] = []
    for tile_op, tiletrace in worker_traces:
        specs.extend(tiletrace.specs_for(tile_op))
    for tiletrace in trace_tiles:
        if tiletrace.tile is None:
            raise ValueError(
                "A TileTrace in Program(trace_tiles=...) must specify a tile."
            )
        specs.extend(tiletrace.specs_for(tiletrace.tile.op))
    return specs
