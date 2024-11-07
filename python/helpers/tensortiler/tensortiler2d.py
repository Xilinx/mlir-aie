from copy import deepcopy
from functools import partial
import numpy as np
from typing import Sequence

from .tensortilesequence import TensorTileSequence
from .utils import ceildiv, validate_and_clean_sizes_strides, validate_tensor_dims


class TensorTiler2D:
    """
    This is a generator (similar to factory pattern) class which produces TensorTileSequence
    objects for common 2-dimensional tiling patterns.
    """

    _DTYPE = np.int32
    _NUM_DIMS = 2

    def __init__(self):
        raise Exception(
            f"{self.__class__} cannot be instantiated. Use it as a factory/generator of TensorTileSequences."
        )

    @classmethod
    def simple_tiler(
        cls,
        tensor_dims: Sequence[int],
        tile_dims: Sequence[int] | None,
        tile_col_major: bool = False,
        iter_col_major: bool = False,
        pattern_repeat: int = 1,
    ) -> TensorTileSequence:
        if tile_dims is None:
            tile_dims = deepcopy(tensor_dims)
        # Special case of group_tiler
        return cls.group_tiler(
            tensor_dims=tensor_dims,
            tile_dims=tile_dims,
            tile_col_major=tile_col_major,
            iter_col_major=iter_col_major,
            pattern_repeat=pattern_repeat,
        )

    @classmethod
    def group_tiler(
        cls,
        tensor_dims: Sequence[int],
        tile_dims: Sequence[int],
        tile_group_dims: Sequence[int] | None = None,
        tile_col_major: bool = False,
        tile_group_col_major: bool = False,
        iter_col_major: bool = False,
        pattern_repeat: int = 1,
        allow_partial: bool = False,
    ) -> TensorTileSequence:
        if tile_group_dims is None:
            tile_group_dims = (1,) * cls._NUM_DIMS
        # Special case of step_tiler
        return cls.step_tiler(
            tensor_dims=tensor_dims,
            tile_dims=tile_dims,
            tile_group_repeats=tile_group_dims,
            tile_col_major=tile_col_major,
            tile_group_col_major=tile_group_col_major,
            iter_col_major=iter_col_major,
            pattern_repeat=pattern_repeat,
            allow_partial=allow_partial,
        )

    @classmethod
    def step_tiler(
        cls,
        tensor_dims: Sequence[int],
        tile_dims: Sequence[int],
        tile_group_repeats: Sequence[int],
        tile_group_steps: Sequence[int] | None = None,
        tile_col_major: bool = False,
        tile_group_col_major: bool = False,
        iter_col_major: bool = False,
        allow_partial: bool = False,
        pattern_repeat: int = 1,
    ) -> TensorTileSequence:
        if tile_group_steps is None:
            tile_group_steps = (1,) * cls._NUM_DIMS
        tensor_dims = validate_tensor_dims(tensor_dims, expected_dims=cls._NUM_DIMS)
        tile_dims = validate_tensor_dims(tile_dims, expected_dims=cls._NUM_DIMS)
        tile_group_repeats = validate_tensor_dims(
            tile_group_repeats, expected_dims=cls._NUM_DIMS
        )
        if len(tile_group_steps) != cls._NUM_DIMS:
            raise ValueError(f"Expcted {cls._NUM_DIMS} dimensions of tile group steps")
        for i, s in enumerate(tile_group_steps):
            if s < 0:
                raise ValueError(
                    f"Tile group step dimension {i} must be >= 0, but got {s}"
                )

        for i, (tensor_dim, tile_dim) in enumerate(zip(tensor_dims, tile_dims)):
            if tensor_dim % tile_dim != 0:
                raise ValueError(
                    f"Tensor dimension {i} ({tensor_dim}) is not divisible by tile dim ({tile_dim})"
                )

        if not isinstance(pattern_repeat, int) or pattern_repeat < 1:
            raise ValueError(f"Pattern repeat must be >= 1 but is {pattern_repeat}")
        if not allow_partial:
            for i, (tensor_dim, tile_dim, repeat_dim, step_dim) in enumerate(
                zip(tensor_dims, tile_dims, tile_group_repeats, tile_group_steps)
            ):
                if tensor_dim % (tile_dim * repeat_dim * step_dim) != 0:
                    raise ValueError(
                        f"allow_partial={allow_partial} but tensor does not divide evenly into tile groups in dimension {i}"
                    )
                if tile_dim * repeat_dim * step_dim > tensor_dim:
                    raise ValueError(
                        f"Tile pattern exceeds tensor size in dimension {i} ({tile_dim}x{repeat_dim}x{step_dim} > {tensor_dim})"
                    )

        steps_per_dim = cls.__get_num_steps(
            tensor_dims=tensor_dims,
            tile_dims=tile_dims,
            step_dims=tile_group_steps,
            repeat_dims=tile_group_repeats,
        )
        print(steps_per_dim)
        num_steps = np.prod(steps_per_dim)
        print(num_steps)

        def offset_fn(step_num: int, _prev_offset: int) -> int:
            tile_offset_in_col, tile_offset_in_row = cls.__tile_offset_by_step_num(
                step_num,
                tile_group_steps,
                tile_group_repeats,
                steps_per_dim,
                iter_col_major,
            )
            # TODO: this code is specific to 2-dimensions
            offset = (
                tile_offset_in_row * tile_dims[1]
                + tile_offset_in_col * tile_dims[0] * tensor_dims[1]
            )
            return offset

        def sizes_or_strides_fn(step_num, _prev_sizes, is_sizes):
            tile_offsets = cls.__tile_offset_by_step_num(
                step_num,
                tile_group_steps,
                tile_group_repeats,
                steps_per_dim,
                iter_col_major,
            )

            iter_sizes, iter_strides = cls.__sizes_strides_for_step_tile_group(
                tensor_dims,
                tile_dims,
                tile_group_steps,
                tile_group_repeats,
                tile_offsets,
                tile_col_major,
                tile_group_col_major,
                pattern_repeat=pattern_repeat,
            )
            if is_sizes:
                return iter_sizes
            else:
                return iter_strides

        sizes_fn = partial(sizes_or_strides_fn, is_sizes=True)
        strides_fn = partial(sizes_or_strides_fn, is_sizes=False)

        return TensorTileSequence(
            tensor_dims,
            num_steps,
            sizes_fn=sizes_fn,
            strides_fn=strides_fn,
            offset_fn=offset_fn,
        )

    @classmethod
    def __get_num_steps(
        cls,
        tensor_dims: Sequence[int],
        tile_dims: Sequence[int],
        step_dims: Sequence[int],
        repeat_dims: Sequence[int],
    ) -> Sequence[int]:
        num_steps_dims = []
        for tensor_dim, tile_dim, step_dim, repeat_dim in zip(
            tensor_dims, tile_dims, step_dims, repeat_dims
        ):
            num_steps_per_dim = tensor_dim // (tile_dim * repeat_dim)
            tiles_in_tensor = (tensor_dim // tile_dim) - num_steps_per_dim * repeat_dim
            partial_height_steps = 0
            while tiles_in_tensor > 0:
                tiles_in_tensor -= min(repeat_dim, ceildiv(tiles_in_tensor, step_dim))
                partial_height_steps += 1
            num_steps_per_dim += partial_height_steps
            num_steps_dims.append(num_steps_per_dim)
        return num_steps_dims

    @classmethod
    def __tile_offset_by_step_num(
        cls,
        step_num: int,
        tile_group_steps: Sequence[int],
        tile_group_repeats: Sequence[int],
        num_steps: Sequence[int],
        iter_col_major: bool,
    ) -> Sequence[int]:
        # TODO: this code is still specific to two dimensions
        steps_per_col, steps_per_row = num_steps
        tile_step_height, tile_step_width = tile_group_steps
        tile_repeat_height, tile_repeat_width = tile_group_repeats

        if not iter_col_major:
            row_idx = step_num % steps_per_row
            col_idx = step_num // steps_per_row
        else:
            col_idx = step_num % steps_per_col
            row_idx = step_num // steps_per_col

        col_chunk_idx = col_idx // tile_step_height
        row_chunk_idx = row_idx // tile_step_width
        col_in_chunk_idx = col_idx % tile_step_height
        row_in_chunk_idx = row_idx % tile_step_width

        tile_offset_in_row = (
            row_chunk_idx * tile_step_width * tile_repeat_width + row_in_chunk_idx
        )
        tile_offset_in_col = (
            col_chunk_idx * tile_step_height * tile_repeat_height + col_in_chunk_idx
        )
        return (tile_offset_in_col, tile_offset_in_row)

    @classmethod
    def __sizes_strides_for_step_tile_group(
        cls,
        tensor_dims: Sequence[int],
        tile_dims: Sequence[int],
        tile_group_steps: Sequence[int],
        tile_group_repeats: Sequence[int],
        tile_offsets: Sequence[int],
        tile_col_major: bool,
        tile_group_col_major: bool,
        pattern_repeat: int,
    ) -> tuple[Sequence[int], Sequence[int]]:
        # TODO: this code is still specific to two dimensions
        # TODO: this code assumes sizes/strides of len 4

        # Interior method, assumes all validation already done
        tensor_height, tensor_width = tensor_dims
        tile_height, tile_width = tile_dims
        tile_step_height, tile_step_width = tile_group_steps
        tile_repeat_height, tile_repeat_width = tile_group_repeats
        tile_offset_height, tile_offset_width = tile_offsets

        tiles_remaining_height = tensor_height // tile_height - tile_offset_height
        tiles_remaining_width = tensor_width // tile_width - tile_offset_width

        # use tile offsets to prune step
        if tile_step_height > tiles_remaining_height:
            tile_step_height = 1
        if tile_step_width > tiles_remaining_width:
            tile_step_width = 1

        # use tile offsets to prune repeat count
        tile_repeat_width = min(
            tile_repeat_width,
            ceildiv(tiles_remaining_width, tile_step_width),
        )
        tile_repeat_height = min(
            tile_repeat_height,
            ceildiv(tiles_remaining_height, tile_step_height),
        )
        tile_group_repeats = (tile_repeat_height, tile_repeat_width)

        if (
            tile_group_col_major
            and tile_step_height == 1
            and tile_repeat_height > 1
            and not tile_col_major
        ):
            # Can combine into one big tile vertically
            tile_height *= tile_repeat_height
            tile_repeat_height = 1
        elif (
            not tile_group_col_major
            and tile_step_width == 1
            and tile_repeat_width > 1
            and tile_col_major
        ):
            # Can combine into one big tile horizontally
            tile_width *= tile_repeat_width
            tile_repeat_width = 1

        if not tile_col_major:
            iter_sizes = [1, 1, tile_height, tile_width]
            iter_strides = [0, 0, tensor_width, 1]
        else:
            iter_sizes = [1, 1, tile_width, tile_height]
            iter_strides = [0, 0, 1, tensor_width]

        if tile_repeat_width > 1 and tile_repeat_height > 1:
            idx_order = [0, 1]
            if tile_group_col_major:
                idx_order = [1, 0]
            iter_sizes[idx_order[0]] = tile_repeat_height
            iter_sizes[idx_order[1]] = tile_repeat_width
            iter_strides[idx_order[0]] = tensor_width * tile_height * tile_step_height
            iter_strides[idx_order[1]] = tile_width * tile_step_width
        elif tile_repeat_height > 1:
            iter_sizes[1], iter_strides[1] = (
                tile_repeat_height,
                tensor_width * tile_height * tile_step_height,
            )
        elif tile_repeat_width > 1:
            iter_sizes[1], iter_strides[1] = (
                tile_repeat_width,
                tile_width * tile_step_width,
            )

        iter_sizes, iter_strides = validate_and_clean_sizes_strides(
            iter_sizes, iter_strides
        )

        if pattern_repeat != 1:
            if iter_sizes[1] == 1 and iter_strides[0] == 0:
                iter_sizes[1] = pattern_repeat
            else:
                if iter_sizes[0] != 1 or iter_strides[0] != 0:
                    raise ValueError(
                        f"Ran out of dimensions for repeat (sizes={iter_sizes}, strides={iter_strides})"
                    )
                iter_sizes[0] = pattern_repeat

        return iter_sizes, iter_strides
