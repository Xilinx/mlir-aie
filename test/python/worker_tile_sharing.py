# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %python %s

"""Tile identity and type-stamping contract for Worker.

Rule: a user-supplied Tile that is already typed CoreTile is stored by identity
(w.tile is t). Any other input — untyped Tile, AnyComputeTile singleton, None —
produces a fresh copy. This lets a Buffer and a Worker sharing the same Tile
object resolve to a single aie.logical_tile op."""

from aie.iron import Worker
from aie.iron.device import AnyComputeTile, Tile
from aie.dialects._aie_enum_gen import AIETileType


def test_default_tile_produces_fresh_copy():
    """Two Workers using the default produce distinct tile objects."""
    w1 = Worker(None)
    w2 = Worker(None)
    assert w1.tile is not w2.tile, "Default tiles should be distinct copies"


def test_anycomputetile_singleton_is_copied():
    """Passing AnyComputeTile directly must not bind the singleton."""
    w = Worker(None, tile=AnyComputeTile)
    assert w.tile is not AnyComputeTile, "AnyComputeTile singleton must be copied"


def test_none_treated_as_anycomputetile():
    """None is treated as AnyComputeTile — produces a fresh copy."""
    w = Worker(None, tile=None)
    assert w.tile is not AnyComputeTile


def test_pre_typed_tile_preserved_by_identity():
    """A CoreTile-typed Tile is stored directly — no copy, no mutation."""
    t = Tile(0, 2, tile_type=AIETileType.CoreTile)
    w = Worker(None, tile=t)
    assert w.tile is t, "Pre-typed CoreTile must be stored by identity"


def test_untyped_placed_tile_is_copied():
    """An untyped placed Tile gets a fresh CoreTile copy (type needs stamping)."""
    t = Tile(0, 2)
    w = Worker(None, tile=t)
    assert w.tile is not t, "Untyped Tile must not be mutated; worker gets a copy"
    assert w.tile.tile_type == AIETileType.CoreTile
    assert w.tile.col == 0 and w.tile.row == 2


def test_wrong_type_raises():
    """Non-CoreTile typed Tile raises ValueError immediately."""
    try:
        Worker(None, tile=Tile(0, 1, tile_type=AIETileType.ShimNOCTile))
        assert False, "Expected ValueError"
    except ValueError:
        pass


test_default_tile_produces_fresh_copy()
test_anycomputetile_singleton_is_copied()
test_none_treated_as_anycomputetile()
test_pre_typed_tile_preserved_by_identity()
test_untyped_placed_tile_is_copied()
test_wrong_type_raises()
print("All tests passed")
