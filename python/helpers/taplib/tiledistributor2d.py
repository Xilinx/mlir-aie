import numpy as np
from typing import Sequence

from .tensortiler2d import TensorTiler2D
from .tas import TensorAccessSequence
from .utils import validate_tensor_dims


class TileDistributor2D:
    """
    This is a class that generates TensorAccessSequences to fulfill the common
    pattern of taking a (M, N) tensor tiled into (m, n) tiles and distributing an
    (ideally equal) number of tiles to w workers.

    This class will divide the tiles equally between worker if N // n // w or M // n // w
    but does not guarantee even split otherwise.
    """

    _NUM_DIMS = 2

    def __init__(self):
        raise Exception(
            f"{self.__class__} cannot be instantiated. \
            Use it as a factory/generator of TensorAccessSequences."
        )

    @classmethod
    def distribute(
        cls,
        tensor_dims: Sequence[int],
        tile_dims: Sequence[int],
        num_workers: int,
        allow_uneven_work: bool = False,
        allow_reduce_num_workers: bool = False,
    ) -> list[TensorAccessSequence]:
        # Validation of inputs
        tensor_dims = validate_tensor_dims(tensor_dims, expected_dims=cls._NUM_DIMS)
        tile_dims = validate_tensor_dims(tile_dims, expected_dims=cls._NUM_DIMS)
        if num_workers < 1:
            raise ValueError("num_workers must be >= 1")

        # Calculate number of tiles per dimension, and check tensor divides evenly into tiles
        num_tiles_per_dim = []
        for i, (tensor_dim, tile_dim) in enumerate(zip(tensor_dims, tile_dims)):
            if tensor_dim % tile_dim != 0:
                raise ValueError(
                    f"Tensor dimension {i} ({tensor_dim}) is not divisible by tile dim ({tile_dim})"
                )
            num_tiles_per_dim.append(tensor_dim // tile_dim)

        # Reduce number of workers if more workers than tiles, or error if not allowed
        total_tiles = np.prod(num_tiles_per_dim)
        if total_tiles > num_workers:
            if not allow_reduce_num_workers:
                raise ValueError(
                    f"More workers than number of tiles and allow_reduce_num_workers is False"
                )
            else:
                num_workers = total_tiles

        # Check for uneven work
        if (not allow_uneven_work) and total_tiles % num_workers != 0:
            raise ValueError(
                f"Number of tiles ({total_tiles}) does not divide evenly by number of workers ({num_workers}) \
                      and allow_uneven_work is False"
            )

        # Try to split over a dimension evenly
        distribute_over_dim = None
        max_mod = -1
        farthest_divide_dim = -1

        for i, count in enumerate(num_tiles_per_dim):
            if count % num_workers == 0:
                distribute_over_dim = i
                break
            elif count % num_workers > max_mod:
                max_mod = count % num_workers
                farthest_divide_dim = i

        if distribute_over_dim is None:
            if not allow_uneven_work:
                raise NotImplementedError(
                    "While the number of tiles could theoretically be split into even groups, \
                                          this function does not know how to tile it evenly (yet)."
                )
            dim_split_over = farthest_divide_dim
        else:
            dim_split_over = distribute_over_dim

        tile_group_repeats = []
        tile_group_steps = []
        for i, num_tiles in enumerate(num_tiles_per_dim):
            if i == dim_split_over:
                tile_group_repeats.append(num_tiles // num_workers)
                tile_group_steps.append(num_workers)
            else:
                tile_group_repeats.append(num_tiles)
                tile_group_steps.append(1)

        # TODO: using the 2D tiler is dependent on being 2-dimensional
        taps = TensorTiler2D.step_tiler(
            tensor_dims,
            tile_dims,
            tile_group_repeats=tile_group_repeats,
            tile_group_steps=tile_group_steps,
            allow_partial=True,
        )

        # Split taps across workers
        taps_per_worker = [[]] * num_workers
        for i, tap in enumerate(taps):
            taps_per_worker[i % num_workers].append(tap)

        return taps_per_worker
