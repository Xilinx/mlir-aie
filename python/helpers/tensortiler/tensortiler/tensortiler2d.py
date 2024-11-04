from enum import Enum
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

    def __init__(self):
        raise Exception(
            f"{self.__class__} cannot be instantiated. Use it as a factory/generator of TensorTileSequences."
        )

    @classmethod
    def simple_tiler(
        cls,
        tensor_dims: Sequence[int],
        tile_dims: Sequence[int],
        tile_col_major: bool = False,
        iter_col_major: bool = False,
        pattern_repeat: int = 1,
    ) -> TensorTileSequence:
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
        tile_group_dims: Sequence[int] = (1, 1),
        tile_col_major: bool = False,
        tile_group_col_major: bool = False,
        iter_col_major: bool = False,
        pattern_repeat: int = 1,
        allow_partial: bool = False,
    ) -> TensorTileSequence:
        tensor_height, tensor_width = validate_tensor_dims(tensor_dims, expected_dims=2)
        tile_height, tile_width = validate_tensor_dims(tile_dims, expected_dims=2)
        tile_group_height, tile_group_width = validate_tensor_dims(
            tile_group_dims, expected_dims=2
        )

        if tensor_width % tile_width != 0:
            raise ValueError(
                f"Tensor width ({tensor_width}) is not divisible by tile width ({tile_width})"
            )
        if tensor_height % tile_height != 0:
            raise ValueError(
                f"Tensor height ({tensor_height}) is not divisible by tile height ({tile_height})"
            )
        if pattern_repeat < 1:
            raise ValueError(f"Pattern repeat must be >= 1 but is {pattern_repeat}")
        if not allow_partial:
            if tensor_width % (tile_width * tile_group_width) != 0:
                raise ValueError(
                    f"allow_partial={allow_partial} but tensor does not divide evenly into tile groups in width"
                )
            if tensor_height % (tile_height * tile_group_height) != 0:
                raise ValueError(
                    f"allow_partial={allow_partial} but tensor does not divide evenly into tile groups in height"
                )
        partial_tile_group_width = (
            tensor_width % (tile_width * tile_group_width)
        ) // tile_width
        partial_tile_group_height = (
            tensor_height % (tile_height * tile_group_height)
        ) // tile_height
        steps_per_row = ceildiv(tensor_width, (tile_width * tile_group_width))
        steps_per_col = ceildiv(tensor_height, (tile_height * tile_group_height))

        num_steps = steps_per_row * steps_per_col

        def calc_offset(iter_num, _prev_offset):
            if not iter_col_major:
                row_idx = iter_num % steps_per_row
                col_idx = iter_num // steps_per_row
            else:
                col_idx = iter_num % steps_per_col
                row_idx = iter_num // steps_per_col

            offset = row_idx * tile_group_width * tile_width
            offset += col_idx * tile_group_height * tensor_width * tile_height
            return offset

        calc_sizes = None
        calc_strides = None
        iter_sizes, iter_strides = cls.__sizes_strides_for_tile_group_with_repeat(
            tensor_dims,
            tile_dims,
            tile_group_dims,
            tile_col_major,
            tile_group_col_major,
            pattern_repeat,
        )

        if partial_tile_group_width:
            partial_row_sizes, partial_row_strides = (
                cls.__sizes_strides_for_tile_group_with_repeat(
                    tensor_dims,
                    tile_dims,
                    (tile_group_height, partial_tile_group_width),
                    tile_col_major,
                    tile_group_col_major,
                    pattern_repeat,
                )
            )
            partial_both_sizes, partial_both_strides = (
                partial_row_sizes,
                partial_row_strides,
            )
        else:
            partial_row_sizes, partial_row_strides = iter_sizes, iter_strides

        if partial_tile_group_height:
            partial_col_sizes, partial_col_strides = (
                cls.__sizes_strides_for_tile_group_with_repeat(
                    tensor_dims,
                    tile_dims,
                    (partial_tile_group_height, tile_group_width),
                    tile_col_major,
                    tile_group_col_major,
                    pattern_repeat,
                )
            )
            partial_both_sizes, partial_both_strides = (
                partial_col_sizes,
                partial_col_strides,
            )
        else:
            partial_col_sizes, partial_col_strides = iter_sizes, iter_strides

        if partial_tile_group_width and partial_tile_group_height:
            partial_both_sizes, partial_both_strides = (
                cls.__sizes_strides_for_tile_group_with_repeat(
                    tensor_dims,
                    tile_dims,
                    (partial_tile_group_height, partial_tile_group_width),
                    tile_col_major,
                    tile_group_col_major,
                    pattern_repeat,
                )
            )

        if partial_tile_group_height or partial_tile_group_width:

            def sizes_or_strides_fn(step_num, _prev_sizes, strides_or_sizes):
                normal, partial_both, partial_col, partial_row = strides_or_sizes
                if not iter_col_major:
                    row_idx = step_num % steps_per_row
                    col_idx = step_num // steps_per_row
                else:
                    col_idx = step_num % steps_per_col
                    row_idx = step_num // steps_per_col

                if row_idx == steps_per_row - 1 and col_idx == steps_per_col - 1:
                    return partial_both
                elif row_idx == steps_per_row - 1:
                    return partial_row
                elif col_idx == steps_per_col - 1:
                    return partial_col
                return normal

            sizes_fn = partial(
                sizes_or_strides_fn,
                strides_or_sizes=[
                    iter_sizes,
                    partial_both_sizes,
                    partial_col_sizes,
                    partial_row_sizes,
                ],
            )
            strides_fn = partial(
                sizes_or_strides_fn,
                strides_or_sizes=[
                    iter_strides,
                    partial_both_strides,
                    partial_col_strides,
                    partial_row_strides,
                ],
            )

            calc_sizes = sizes_fn
            calc_strides = strides_fn

        return TensorTileSequence(
            (tensor_height, tensor_width),
            num_steps,
            sizes=iter_sizes,
            strides=iter_strides,
            offset_fn=calc_offset,
            sizes_fn=calc_sizes,
            strides_fn=calc_strides,
        )

    @classmethod
    def step_tiler(
        cls,
        tensor_dims: Sequence[int],
        tile_dims: Sequence[int],
        tile_group_steps: Sequence[int],
        tile_group_repeats: Sequence[int],
        tile_col_major: bool = False,
        tile_group_col_major: bool = False,
        iter_col_major: bool = False,
        allow_partial: bool = False,
        pattern_repeat: int = 1,
    ) -> TensorTileSequence:
        tensor_height, tensor_width = validate_tensor_dims(tensor_dims, expected_dims=2)
        tile_height, tile_width = validate_tensor_dims(tile_dims, expected_dims=2)
        tile_repeat_height, tile_repeat_width = validate_tensor_dims(
            tile_group_repeats, expected_dims=2
        )
        if len(tile_group_steps) != 2:
            raise ValueError("Expcted two dimensions of tile group steps")
        tile_step_height, tile_step_width = tile_group_steps
        if tile_step_height < 0 or tile_step_width < 0:
            raise ValueError(
                f"Tile group steps must be >= 0, but got ({tile_group_steps})"
            )

        if tensor_width % tile_width != 0:
            raise ValueError(
                f"Tensor width ({tensor_width}) is not divisible by tile width ({tile_width})"
            )
        if tensor_height % tile_height != 0:
            raise ValueError(
                f"Tensor height ({tensor_height}) is not divisible by tile height ({tile_height})"
            )

        if not allow_partial:
            if tensor_width % (tile_width * tile_repeat_width * tile_step_width) != 0:
                raise ValueError(
                    f"allow_partial={allow_partial} but tensor does not divide evenly into tile groups in width"
                )
            if (
                tensor_height % (tile_height * tile_repeat_height * tile_step_height)
                != 0
            ):
                raise ValueError(
                    f"allow_partial={allow_partial} but tensor does not divide evenly into tile groups in height"
                )
            if tile_width * tile_repeat_width * tile_step_width > tensor_width:
                raise ValueError(
                    f"Tile pattern exceeds tensor width ({tile_width}x{tile_repeat_width}x{tile_step_width} > {tensor_width})"
                )
            if tile_height * tile_repeat_height * tile_step_height > tensor_height:
                raise ValueError(
                    f"Tile pattern exceeds tensor height ({tile_height}x{tile_repeat_height}x{tile_step_height} > {tensor_height})"
                )

        # prune repeat counts if not enough space in tensor for single pattern
        tile_repeat_width = min(
            tile_repeat_width,
            ceildiv((tensor_width // tile_width), tile_step_width * tile_repeat_width),
        )
        tile_repeat_height = min(
            tile_repeat_height,
            ceildiv(
                (tensor_height // tile_height), tile_step_height * tile_repeat_height
            ),
        )

        steps_per_row = (tensor_width // (tile_width * tile_repeat_width)) + (
            tensor_width % (tile_width * tile_repeat_width)
        ) // tile_width
        steps_per_col = (tensor_height // (tile_height * tile_repeat_height)) + (
            tensor_height % (tile_height * tile_repeat_height)
        ) // tile_height

        num_steps = steps_per_row * steps_per_col

        def calc_offset(iter_num, _prev_offset):
            if not iter_col_major:
                row_idx = iter_num % steps_per_row
                col_idx = iter_num // steps_per_row
            else:
                col_idx = iter_num % steps_per_col
                row_idx = iter_num // steps_per_col

            col_chunk_idx = col_idx // tile_step_height
            row_chunk_idx = row_idx // tile_step_width
            col_in_chunk_idx = col_idx % tile_step_height
            row_in_chunk_idx = row_idx % tile_step_width

            offset = (
                row_chunk_idx * tile_width * tile_step_width * tile_repeat_width
                + row_in_chunk_idx * tile_width
            )
            offset += (
                col_chunk_idx * tile_height * tile_step_height * tile_repeat_height
                + col_in_chunk_idx * tile_height
            ) * tensor_width
            return offset

        iter_sizes, iter_strides = cls.__sizes_strides_for_step_tile_group(
            tensor_dims,
            tile_dims,
            tile_group_steps,
            tile_group_repeats,
            tile_col_major,
            tile_group_col_major,
            pattern_repeat=pattern_repeat,
        )

        return TensorTileSequence(
            (tensor_height, tensor_width),
            num_steps,
            sizes=iter_sizes,
            strides=iter_strides,
            offset_fn=calc_offset,
        )

    @classmethod
    def __sizes_strides_for_step_tile_group(
        cls,
        tensor_dims: Sequence[int],
        tile_dims: Sequence[int],
        tile_group_steps: Sequence[int],
        tile_group_repeats: Sequence[int],
        tile_col_major: bool,
        tile_group_col_major: bool,
        pattern_repeat: int,
    ) -> tuple[Sequence[int], Sequence[int]]:

        # Interior method, assumes all validation already done
        _tensor_height, tensor_width = tensor_dims
        tile_height, tile_width = tile_dims
        tile_step_height, tile_step_width = tile_group_steps
        tile_repeat_height, tile_repeat_width = tile_group_repeats

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

    @classmethod
    def __sizes_strides_for_tile_group_with_repeat(
        cls,
        tensor_dims: Sequence[int],
        tile_dims: Sequence[int],
        tile_group_dims: Sequence[int],
        tile_col_major: bool,
        tile_group_col_major: bool,
        pattern_repeat: int,
    ) -> tuple[Sequence[int], Sequence[int]]:

        # Interior method, assumes all validation already done
        _tensor_height, tensor_width = tensor_dims
        tile_height, tile_width = tile_dims
        tile_group_height, tile_group_width = tile_group_dims

        if not tile_col_major:
            iter_sizes = [1, 1, tile_height, tile_width]
            iter_strides = [0, 0, tensor_width, 1]
        else:
            iter_sizes = [1, 1, tile_width, tile_height]
            iter_strides = [0, 0, 1, tensor_width]

        if tile_group_height != 1 or tile_group_width != 1:
            if tile_col_major and not tile_group_col_major:
                # This is a special case where we can combine a tile group into one logical tile (horizontally)
                iter_sizes[1], iter_sizes[2] = (
                    tile_group_height,
                    tile_group_width * tile_width,
                )
                iter_strides[2] = 1
                iter_strides[1] = tensor_width * tile_height
            elif not tile_col_major and tile_group_col_major:
                # This is a special case where we can combine a tile group into one logical tile (vertically)
                iter_sizes[1], iter_sizes[2] = (
                    tile_group_width,
                    tile_group_height * tile_height,
                )
                iter_strides[1] = tile_width
                iter_strides[2] = tensor_width
            elif tile_group_width == 1:
                # These are two more special cases; we can combine tiles here too to get a simpler transform
                if tile_col_major:
                    iter_sizes[1] = tile_group_height
                    iter_strides[1] = tile_height * tensor_width
                else:
                    iter_sizes[2] = tile_height * tile_group_height
                    iter_strides[1], iter_strides[2] = tile_width, tensor_width
            elif tile_group_height == 1:
                # These are two more special cases; we can combine tiles here too to get a simpler transform
                if tile_col_major:
                    iter_sizes[2] = tile_width * tile_group_width
                else:
                    iter_sizes[1] = tile_group_width
                    iter_strides[1], iter_strides[2] = tile_width, tensor_width
            else:
                # This should always be the case that creates a correct transfrom;
                # however, it may be needlessly complex (large values in out dimensions)
                idx_order = [0, 1]
                if tile_group_col_major:
                    idx_order = [1, 0]
                iter_sizes[idx_order[0]] = tile_group_height
                iter_sizes[idx_order[1]] = tile_group_width
                iter_strides[idx_order[0]] = tensor_width * tile_height
                iter_strides[idx_order[1]] = tile_width

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
