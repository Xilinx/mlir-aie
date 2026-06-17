# test_tiletrace.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc.

# RUN: %python -m pytest %s -v

"""Tests for TileTrace event-unit inference, validation, and spec building.

These cover the pure-Python logic of the IRON trace API redesign: a single
TileTrace infers the hardware trace unit of each event from its type, validates
that unit against the tile it is attached to, and expands to per-unit specs.
"""

import pytest

from aie.utils.trace.tiletrace import (
    TileTrace,
    build_trace_specs,
    _event_unit,
    _UNITS_FOR_TILE,
    _tile_class,
)
from aie.utils.trace.events import (
    CoreEvent,
    MemEvent,
    MemTileEvent,
    ShimTileEvent,
    PortEvent,
    MemTilePortEvent,
    WireBundle,
)


class FakeTileOp:
    """Minimal stand-in for an MLIR tile op, exposing the is_*_tile() predicates
    and col/row that TileTrace.specs_for / build_trace_specs rely on."""

    def __init__(self, kind, col=0, row=0):
        self._kind = kind  # "core" | "mem" | "shim"
        self.col = col
        self.row = row

    def is_core_tile(self):
        return self._kind == "core"

    def is_mem_tile(self):
        return self._kind == "mem"

    def is_shim_tile(self):
        return self._kind == "shim"


class FakeTile:
    """Stand-in for an iron Tile: just carries a resolved .op."""

    def __init__(self, op):
        self.op = op


# -- event -> unit inference ---------------------------------------------------


class TestEventUnit:
    def test_bare_enum_events(self):
        assert _event_unit(CoreEvent.INSTR_VECTOR) == "core"
        assert _event_unit(MemEvent.DMA_S2MM_0_START_TASK) == "mem"
        assert _event_unit(MemTileEvent.NONE) == "memtile"
        assert _event_unit(ShimTileEvent.DMA_S2MM_0_START_TASK) == "shim"

    def test_wrapped_port_events_unwrap_to_code(self):
        pe = PortEvent(CoreEvent.PORT_RUNNING_0, WireBundle.DMA, 0, True)
        assert _event_unit(pe) == "core"
        mpe = MemTilePortEvent(MemTileEvent.PORT_RUNNING_0, WireBundle.DMA, 0, True)
        assert _event_unit(mpe) == "memtile"

    def test_overlapping_int_values_distinguished_by_class(self):
        # CoreEvent.NONE and MemEvent.NONE share int value 0 but different units.
        assert int(CoreEvent.NONE) == int(MemEvent.NONE)
        assert _event_unit(CoreEvent.NONE) == "core"
        assert _event_unit(MemEvent.NONE) == "mem"

    def test_unknown_event_type_raises(self):
        with pytest.raises(ValueError, match="Cannot determine trace unit"):
            _event_unit(123)


# -- TileTrace.units() grouping ------------------------------------------------


class TestUnitsGrouping:
    def test_none_events_yields_empty(self):
        assert TileTrace().units() == {}

    def test_mixed_core_and_mem_split(self):
        tt = TileTrace(
            events=[
                CoreEvent.INSTR_VECTOR,
                MemEvent.DMA_S2MM_0_START_TASK,
                CoreEvent.MEMORY_STALL,
            ]
        )
        units = tt.units()
        assert set(units) == {"core", "mem"}
        assert len(units["core"]) == 2
        assert len(units["mem"]) == 1


# -- specs_for: defaults + validation ------------------------------------------


class TestSpecsFor:
    def test_default_core_tile_emits_both_units(self):
        op = FakeTileOp("core")
        specs = TileTrace().specs_for(op)
        units = {unit for (_, unit, _) in specs}
        assert units == {"core", "mem"}
        # default events are deferred (None) for the lowering to fill in
        assert all(events is None for (_, _, events) in specs)

    def test_default_mem_tile_single_unit(self):
        specs = TileTrace().specs_for(FakeTileOp("mem"))
        assert [unit for (_, unit, _) in specs] == ["memtile"]

    def test_default_shim_tile_single_unit(self):
        specs = TileTrace().specs_for(FakeTileOp("shim"))
        assert [unit for (_, unit, _) in specs] == ["shim"]

    def test_explicit_events_only_named_units(self):
        op = FakeTileOp("core")
        specs = TileTrace(events=[CoreEvent.INSTR_VECTOR]).specs_for(op)
        assert [unit for (_, unit, _) in specs] == ["core"]

    def test_wrong_unit_for_tile_raises(self):
        # CoreEvent on a mem tile: the 'core' unit doesn't exist there.
        with pytest.raises(ValueError, match="only has units"):
            TileTrace(events=[CoreEvent.INSTR_VECTOR]).specs_for(FakeTileOp("mem"))

    def test_mem_event_on_shim_raises(self):
        with pytest.raises(ValueError, match="only has units"):
            TileTrace(events=[MemEvent.NONE]).specs_for(FakeTileOp("shim"))


# -- _tile_class / _UNITS_FOR_TILE ---------------------------------------------


class TestTileClass:
    def test_tile_class_mapping(self):
        assert _tile_class(FakeTileOp("core")) == "core"
        assert _tile_class(FakeTileOp("mem")) == "mem"
        assert _tile_class(FakeTileOp("shim")) == "shim"

    def test_units_for_tile(self):
        assert _UNITS_FOR_TILE["core"] == ("core", "mem")
        assert _UNITS_FOR_TILE["mem"] == ("memtile",)
        assert _UNITS_FOR_TILE["shim"] == ("shim",)


# -- build_trace_specs ---------------------------------------------------------


class TestBuildTraceSpecs:
    def test_worker_and_trace_tiles_combined(self):
        core_op = FakeTileOp("core", col=0, row=2)
        mem_op = FakeTileOp("mem", col=0, row=1)
        worker_traces = [(core_op, TileTrace())]
        trace_tiles = [TileTrace(tile=FakeTile(mem_op))]
        specs = build_trace_specs(worker_traces, trace_tiles)
        units = sorted(unit for (_, unit, _) in specs)
        # core+mem from the worker, memtile from the trace tile
        assert units == ["core", "mem", "memtile"]

    def test_trace_tile_without_tile_raises(self):
        with pytest.raises(ValueError, match="must specify a tile"):
            build_trace_specs([], [TileTrace(events=[MemTileEvent.NONE])])

    def test_empty_inputs_yield_no_specs(self):
        assert build_trace_specs([], []) == []
