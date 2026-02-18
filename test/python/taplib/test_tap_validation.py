import pytest
import numpy as np
from aie.helpers.taplib import TensorAccessPattern


def test_bounds_validation():
    # Valid pattern
    tap = TensorAccessPattern((10,), 0, (10,), (1,))

    # Invalid offset
    with pytest.raises(ValueError, match="Offset too large"):
        TensorAccessPattern((10,), 10, (1,), (1,))

    # Invalid bounds (access index 10)
    with pytest.raises(ValueError, match="Access pattern goes out of bounds"):
        TensorAccessPattern((10,), 0, (11,), (1,))

    # Invalid bounds with stride
    with pytest.raises(ValueError, match="Access pattern goes out of bounds"):
        TensorAccessPattern(
            (10,), 0, (6,), (2,)
        )  # max index = 0 + 5*2 = 10 (out of bounds)


def test_split_dim_validation():
    tap = TensorAccessPattern((10,))

    # Invalid outer size
    with pytest.raises(ValueError, match="Outer size must be positive"):
        tap.split_dim(0, 0)

    with pytest.raises(ValueError, match="Outer size must be positive"):
        tap.split_dim(0, -1)

    # Not divisible
    with pytest.raises(ValueError, match="not divisible"):
        tap.split_dim(0, 3)


def test_slice_validation():
    tap = TensorAccessPattern((10,))

    # Invalid length
    with pytest.raises(ValueError, match="Slice length must be positive"):
        tap.slice(0, 0, 0)

    with pytest.raises(ValueError, match="Slice length must be positive"):
        tap.slice(0, 0, -1)

    # Out of bounds slice
    with pytest.raises(ValueError, match="Invalid slice"):
        tap.slice(0, 5, 6)  # 5+6 = 11 > 10

    with pytest.raises(ValueError, match="Invalid slice"):
        tap.slice(0, -1, 5)


def test_tile_validation():
    tap = TensorAccessPattern((10,))

    # Tile size larger than dimension (outer size 0)
    with pytest.raises(ValueError, match="Outer size must be positive"):
        tap.tile((11,))


def test_resize_validation():
    tap = TensorAccessPattern((10,))

    # Resize larger than original
    with pytest.raises(ValueError, match="Invalid slice"):
        tap.resize((11,))

    # Resize to 0
    with pytest.raises(ValueError, match="Slice length must be positive"):
        tap.resize((0,))
