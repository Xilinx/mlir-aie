# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %python %s

"""Test that Worker shares tile identity when an explicit tile is passed,
but copies the tile when using the default AnyComputeTile singleton."""

from aie.iron import Worker
from aie.iron.device import AnyComputeTile, Tile


def test_default_tile_is_copied():
    """Two Workers using the default tile should get separate tile objects."""
    w1 = Worker(None)
    w2 = Worker(None)
    assert w1.tile is not w2.tile, "Default tiles should be distinct copies"


def test_explicit_tile_is_shared():
    """An explicitly-passed tile should be shared, not copied."""
    t = AnyComputeTile.copy()
    w = Worker(None, tile=t)
    assert w.tile is t, "Explicit tile should be shared with the Worker"


def test_default_instance_is_not_mutated():
    """Passing AnyComputeTile directly should copy it to avoid mutation."""
    w = Worker(None, tile=AnyComputeTile)
    assert w.tile is not AnyComputeTile, "Shared default should be copied"


def test_placed_tile_is_shared():
    """A placed Tile with coordinates should be shared."""
    t = Tile(0, 2)
    w = Worker(None, tile=t)
    assert w.tile is t, "Placed tile should be shared with the Worker"


test_default_tile_is_copied()
test_explicit_tile_is_shared()
test_default_instance_is_not_mutated()
test_placed_tile_is_shared()
print("All tests passed")
