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
        tile_repeat: int = 1,
    ) -> TensorTileSequence:
        # Special case of group_tiler
        return cls.group_tiler(
            tensor_dims=tensor_dims,
            tile_dims=tile_dims,
            tile_col_major=tile_col_major,
            iter_col_major=iter_col_major,
            tile_repeat=tile_repeat,
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
        tile_repeat: int = 1,
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
        if tile_repeat < 1:
            raise ValueError(f"Tile repeat must be >= 1 but is {tile_repeat}")
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
            tile_repeat,
        )

        if partial_tile_group_width:
            partial_row_sizes, partial_row_strides = (
                cls.__sizes_strides_for_tile_group_with_repeat(
                    tensor_dims,
                    tile_dims,
                    (tile_group_height, partial_tile_group_width),
                    tile_col_major,
                    tile_group_col_major,
                    tile_repeat,
                )
            )

        if partial_tile_group_height:
            partial_col_sizes, partial_col_strides = (
                cls.__sizes_strides_for_tile_group_with_repeat(
                    tensor_dims,
                    tile_dims,
                    (partial_tile_group_height, tile_group_width),
                    tile_col_major,
                    tile_group_col_major,
                    tile_repeat,
                )
            )

        if partial_tile_group_width and partial_tile_group_height:
            partial_both_sizes, partial_both_strides = (
                cls.__sizes_strides_for_tile_group_with_repeat(
                    tensor_dims,
                    tile_dims,
                    (partial_tile_group_height, partial_tile_group_width),
                    tile_col_major,
                    tile_group_col_major,
                    tile_repeat,
                )
            )

            def sizes_or_strides_fn(step_num, _prev_sizes, strides_or_sizes):
                normal, partial_both, partial_col, partial_row = strides_or_sizes
                if not iter_col_major:
                    row_idx = step_num % steps_per_row
                    col_idx = step_num // steps_per_row
                else:
                    col_idx = step_num % steps_per_col
                    row_idx = step_num // steps_per_col

                if (
                    partial_tile_group_width
                    and partial_tile_group_height
                    and step_num == num_steps - 1
                ):
                    return partial_both
                elif row_idx == steps_per_row - 1:
                    return partial_col
                elif col_idx == steps_per_col - 1:
                    return partial_row
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
    def __sizes_strides_for_tile_group_with_repeat(
        cls,
        tensor_dims: Sequence[int],
        tile_dims: Sequence[int],
        tile_group_dims: Sequence[int],
        tile_col_major: bool,
        tile_group_col_major: bool,
        tile_repeat: int,
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

        if tile_repeat != 1:
            if iter_sizes[1] == 1 and iter_strides[0] == 0:
                iter_sizes[1] = tile_repeat
            else:
                if iter_sizes[0] != 1 or iter_strides[0] != 0:
                    raise ValueError(
                        f"Ran out of dimensions for repeat (sizes={iter_sizes}, strides={iter_strides})"
                    )
                iter_sizes[0] = tile_repeat

        return iter_sizes, iter_strides

    """
    # TileStepGroupVerticalTiler(VerticalTileStep, RepeatCount=All|N, HorizontalRepeat=N, TileColMajor=True|False, IterColMajor=True|False, IterScheme=TensorComplete|BlockComplete)
    # Copy #3
    def tile_iter(
        self,
        tile_group_height: int = 1,
        tile_group_width: int = 1,
        tile_repeat_step_horizontal: int | None = None,
        tile_repeat_step_vertical: int | None = None,
        tile_repeat: int = 1,
        col_major: bool = False,
        iter_step: int = 1,
    ) -> TensorTile2DIter:
        assert (
            tile_group_height >= 1 and tile_group_width >= 1 and tile_repeat >= 1
        ), f"Tile group height, Tile group width, tile repeat ({tile_group_height}, {tile_group_width}, {tile_repeat}) must be >0"
        assert (
            tile_repeat_step_horizontal is None or tile_repeat_step_horizontal > 0
        ) and (
            tile_repeat_step_vertical is None or tile_repeat_step_vertical > 0
        ), f"Tile repeat step horizontal and vertical ({tile_repeat_step_horizontal}, {tile_repeat_step_vertical}) must be None or >0"
        assert (tile_group_height == 1 or tile_repeat_step_vertical is None) and (
            tile_group_width == 1 or tile_repeat_step_horizontal is None
        ), f"Cannot specify both tile_step and tile_group for given dimension"
        assert (
            self._num_tiles_per_row % tile_group_width == 0
        ), f"Tiles per row ({self._num_tiles_per_row}) must be divisible by tile group width ({tile_group_width})"
        assert (
            self._num_tiles_per_col % tile_group_height == 0
        ), f"Tiles per row ({self._num_tiles_per_col}) must be divisible by tile group width ({tile_group_height})"

        steps_per_row = self._num_tiles_per_row // tile_group_width
        steps_per_col = self._num_tiles_per_col // tile_group_height

        if tile_repeat_step_horizontal:
            assert (
                self._num_tiles_per_row % tile_repeat_step_horizontal == 0
            ), f"Tiles per row ({self._num_tiles_per_row}) must be divisible by tile repeat step horizontal ({tile_repeat_step_horizontal})"
            steps_per_row = self._num_tiles_per_row // tile_repeat_step_horizontal
        if tile_repeat_step_vertical:
            assert (
                self._num_tiles_per_col % tile_repeat_step_vertical == 0
            ), f"Tiles per col ({self._num_tiles_per_col}) must be divisible by tile repeat step vertical ({tile_repeat_step_vertical})"
            steps_per_col = self._num_tiles_per_col // tile_repeat_step_vertical

        if tile_repeat_step_horizontal:
            steps_per_row = tile_repeat_step_horizontal
        if tile_repeat_step_vertical:
            steps_per_col = tile_repeat_step_vertical

        steps = steps_per_row * steps_per_col
        assert (
            iter_step == 1 or steps % iter_step == 0
        ), "Cannot iterate in steps of the given size, must be divisible by total steps"

        def calc_offset(iter_num):
            if iter_step != 1:
                iter_num *= iter_step
            if not col_major:
                row_idx = iter_num % steps_per_row
                col_idx = iter_num // steps_per_row
            else:
                col_idx = iter_num % steps_per_col
                row_idx = iter_num // steps_per_col

            offset = row_idx * tile_group_width * self._tile_width
            offset += (
                col_idx * tile_group_height * self._tensor_width * self._tile_height
            )
            return offset

        iter_sizes = self._sizes.copy()
        iter_strides = self._strides.copy()

        if self._tile_col_major and not self._tensor_col_major:
            # This is a special case where we can combine a tile group into one logical tile (horizontally)
            iter_sizes[1] = tile_group_height
            iter_sizes[2] = tile_group_width * self._tile_width
        elif not self._tile_col_major and self._tensor_col_major:
            # This is a special case where we can combine a tile group into one logical tile (vertically)
            iter_sizes[1] = tile_group_width
            iter_sizes[2] = tile_group_height * self._tile_height
        elif tile_group_width == 1:
            # These are two more special cases; we can combine tiles here too to get a simpler transform
            if self._tile_col_major:
                iter_sizes = [1, tile_group_height, self._tile_width, self._tile_height]
                iter_strides = [
                    1,
                    self._tile_height * self._tensor_width,
                    1,
                    self._tensor_width,
                ]
            else:
                iter_sizes = [
                    1,
                    1,
                    self._tile_height * tile_group_height,
                    self._tile_width,
                ]
                iter_strides = [1, self._tile_width, self._tensor_width, 1]
        elif tile_group_height == 1:
            # These are two more special cases; we can combine tiles here too to get a simpler transform
            if self._tile_col_major:
                iter_sizes = [
                    1,
                    1,
                    self._tile_width * tile_group_width,
                    self._tile_height,
                ]
                iter_strides = [1, 1, 1, self._tensor_width]
            else:
                iter_sizes = [1, tile_group_width, self._tile_height, self._tile_width]
                iter_strides = [1, self._tile_width, self._tensor_width, 1]
        else:
            # This should always be the case that creates a correct transfrom;
            # however, it may be needlessly complex (large values in out dims)
            size_idx = [0, 1]
            if self._tensor_col_major:
                size_idx = [1, 0]
            iter_sizes[size_idx[0]] = tile_group_height
            iter_sizes[size_idx[1]] = tile_group_width

        iter_sizes, iter_strides = self._validate_and_clean_sizes_strides(
            iter_sizes, iter_strides
        )

        if tile_repeat_step_horizontal:
            tile_row_repeats = ceildiv(
                (self._tensor_width // self._tile_width), tile_repeat_step_horizontal
            )
            assert (
                iter_sizes[1] == 1 and iter_strides[1] == 0
            ), f"Cannot do col tile repeat step horizontal, sizes={iter_sizes}, strides={iter_strides}"
            iter_sizes[1] = tile_row_repeats
            iter_strides[1] = self._tile_width * tile_repeat_step_horizontal

        if tile_repeat_step_vertical:
            tile_col_repeats = ceildiv(
                (self._tensor_height // self._tile_height), tile_repeat_step_vertical
            )
            assert (
                iter_sizes[1] == 1 and iter_strides[1] == 0
            ), f"Cannot do tile repeat step vertical, sizes={iter_sizes}, strides={iter_strides}"
            iter_sizes[1] = tile_col_repeats
            iter_strides[1] = (
                self._tensor_width * self._tile_height * tile_repeat_step_vertical
            )

        iter_sizes, iter_strides = self._validate_and_clean_sizes_strides(
            iter_sizes, iter_strides
        )

        if tile_repeat != 1:
            assert (
                iter_sizes[0] == 1 and iter_strides[0] == 0
            ), f"Highest (sizes, strides) dim must be (1, 0) for tile repeat but is ({iter_sizes}, {iter_strides})"
            iter_sizes[0] = tile_repeat

        if iter_step != 1:
            # TODO: unchecked with other features, including repeat
            assert (
                iter_sizes[0] == 1 and iter_strides[0] == 0
            ), f"Highest (sizes, strides) dim must be (1, 0) for iter step but is ({iter_sizes}, {iter_strides})"
            iter_sizes[0] = iter_step
            iter_strides[0] = tile_group_height * self._tile_height * self._tensor_width

        return TensorTile2DIter(
            self._tensor_height,
            self._tensor_width,
            iter_sizes,
            iter_strides,
            offset_fn=calc_offset,
            num_steps=steps,
        )
    # End Copy 3
    """

    """
    # TileStepGroupHorizontalTiler(HorizontalTileStep, RepeatCount=All|N, VerticalRepeat=N, TileColMajor=True|False, IterColMajor=True|False, IterScheme=TensorComplete|BlockComplete)
    # Copy #4
    def tile_iter(
        self,
        tile_group_height: int = 1,
        tile_group_width: int = 1,
        tile_repeat_step_horizontal: int | None = None,
        tile_repeat_step_vertical: int | None = None,
        tile_repeat: int = 1,
        col_major: bool = False,
        iter_step: int = 1,
    ) -> TensorTile2DIter:
        assert (
            tile_group_height >= 1 and tile_group_width >= 1 and tile_repeat >= 1
        ), f"Tile group height, Tile group width, tile repeat ({tile_group_height}, {tile_group_width}, {tile_repeat}) must be >0"
        assert (
            tile_repeat_step_horizontal is None or tile_repeat_step_horizontal > 0
        ) and (
            tile_repeat_step_vertical is None or tile_repeat_step_vertical > 0
        ), f"Tile repeat step horizontal and vertical ({tile_repeat_step_horizontal}, {tile_repeat_step_vertical}) must be None or >0"
        assert (tile_group_height == 1 or tile_repeat_step_vertical is None) and (
            tile_group_width == 1 or tile_repeat_step_horizontal is None
        ), f"Cannot specify both tile_step and tile_group for given dimension"
        assert (
            self._num_tiles_per_row % tile_group_width == 0
        ), f"Tiles per row ({self._num_tiles_per_row}) must be divisible by tile group width ({tile_group_width})"
        assert (
            self._num_tiles_per_col % tile_group_height == 0
        ), f"Tiles per row ({self._num_tiles_per_col}) must be divisible by tile group width ({tile_group_height})"

        steps_per_row = self._num_tiles_per_row // tile_group_width
        steps_per_col = self._num_tiles_per_col // tile_group_height

        if tile_repeat_step_horizontal:
            assert (
                self._num_tiles_per_row % tile_repeat_step_horizontal == 0
            ), f"Tiles per row ({self._num_tiles_per_row}) must be divisible by tile repeat step horizontal ({tile_repeat_step_horizontal})"
            steps_per_row = self._num_tiles_per_row // tile_repeat_step_horizontal
        if tile_repeat_step_vertical:
            assert (
                self._num_tiles_per_col % tile_repeat_step_vertical == 0
            ), f"Tiles per col ({self._num_tiles_per_col}) must be divisible by tile repeat step vertical ({tile_repeat_step_vertical})"
            steps_per_col = self._num_tiles_per_col // tile_repeat_step_vertical

        if tile_repeat_step_horizontal:
            steps_per_row = tile_repeat_step_horizontal
        if tile_repeat_step_vertical:
            steps_per_col = tile_repeat_step_vertical

        steps = steps_per_row * steps_per_col
        assert (
            iter_step == 1 or steps % iter_step == 0
        ), "Cannot iterate in steps of the given size, must be divisible by total steps"

        def calc_offset(iter_num):
            if iter_step != 1:
                iter_num *= iter_step
            if not col_major:
                row_idx = iter_num % steps_per_row
                col_idx = iter_num // steps_per_row
            else:
                col_idx = iter_num % steps_per_col
                row_idx = iter_num // steps_per_col

            offset = row_idx * tile_group_width * self._tile_width
            offset += (
                col_idx * tile_group_height * self._tensor_width * self._tile_height
            )
            return offset

        iter_sizes = self._sizes.copy()
        iter_strides = self._strides.copy()

        if self._tile_col_major and not self._tensor_col_major:
            # This is a special case where we can combine a tile group into one logical tile (horizontally)
            iter_sizes[1] = tile_group_height
            iter_sizes[2] = tile_group_width * self._tile_width
        elif not self._tile_col_major and self._tensor_col_major:
            # This is a special case where we can combine a tile group into one logical tile (vertically)
            iter_sizes[1] = tile_group_width
            iter_sizes[2] = tile_group_height * self._tile_height
        elif tile_group_width == 1:
            # These are two more special cases; we can combine tiles here too to get a simpler transform
            if self._tile_col_major:
                iter_sizes = [1, tile_group_height, self._tile_width, self._tile_height]
                iter_strides = [
                    1,
                    self._tile_height * self._tensor_width,
                    1,
                    self._tensor_width,
                ]
            else:
                iter_sizes = [
                    1,
                    1,
                    self._tile_height * tile_group_height,
                    self._tile_width,
                ]
                iter_strides = [1, self._tile_width, self._tensor_width, 1]
        elif tile_group_height == 1:
            # These are two more special cases; we can combine tiles here too to get a simpler transform
            if self._tile_col_major:
                iter_sizes = [
                    1,
                    1,
                    self._tile_width * tile_group_width,
                    self._tile_height,
                ]
                iter_strides = [1, 1, 1, self._tensor_width]
            else:
                iter_sizes = [1, tile_group_width, self._tile_height, self._tile_width]
                iter_strides = [1, self._tile_width, self._tensor_width, 1]
        else:
            # This should always be the case that creates a correct transfrom;
            # however, it may be needlessly complex (large values in out dims)
            size_idx = [0, 1]
            if self._tensor_col_major:
                size_idx = [1, 0]
            iter_sizes[size_idx[0]] = tile_group_height
            iter_sizes[size_idx[1]] = tile_group_width

        iter_sizes, iter_strides = self._validate_and_clean_sizes_strides(
            iter_sizes, iter_strides
        )

        if tile_repeat_step_horizontal:
            tile_row_repeats = ceildiv(
                (self._tensor_width // self._tile_width), tile_repeat_step_horizontal
            )
            assert (
                iter_sizes[1] == 1 and iter_strides[1] == 0
            ), f"Cannot do col tile repeat step horizontal, sizes={iter_sizes}, strides={iter_strides}"
            iter_sizes[1] = tile_row_repeats
            iter_strides[1] = self._tile_width * tile_repeat_step_horizontal

        if tile_repeat_step_vertical:
            tile_col_repeats = ceildiv(
                (self._tensor_height // self._tile_height), tile_repeat_step_vertical
            )
            assert (
                iter_sizes[1] == 1 and iter_strides[1] == 0
            ), f"Cannot do tile repeat step vertical, sizes={iter_sizes}, strides={iter_strides}"
            iter_sizes[1] = tile_col_repeats
            iter_strides[1] = (
                self._tensor_width * self._tile_height * tile_repeat_step_vertical
            )

        iter_sizes, iter_strides = self._validate_and_clean_sizes_strides(
            iter_sizes, iter_strides
        )

        if tile_repeat != 1:
            assert (
                iter_sizes[0] == 1 and iter_strides[0] == 0
            ), f"Highest (sizes, strides) dim must be (1, 0) for tile repeat but is ({iter_sizes}, {iter_strides})"
            iter_sizes[0] = tile_repeat

        if iter_step != 1:
            # TODO: unchecked with other features, including repeat
            assert (
                iter_sizes[0] == 1 and iter_strides[0] == 0
            ), f"Highest (sizes, strides) dim must be (1, 0) for iter step but is ({iter_sizes}, {iter_strides})"
            iter_sizes[0] = iter_step
            iter_strides[0] = tile_group_height * self._tile_height * self._tensor_width

        return TensorTile2DIter(
            self._tensor_height,
            self._tensor_width,
            iter_sizes,
            iter_strides,
            offset_fn=calc_offset,
            num_steps=steps,
        )
    # End Copy 4
    """
