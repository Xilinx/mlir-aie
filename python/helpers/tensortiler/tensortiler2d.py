import numpy as np
import os
import sys
from typing import Callable


class TensorTile:
    def __init__(
        self,
        tensor_height: int,
        tensor_width: int,
        offset: int,
        sizes: list[int],
        strides: list[int],
    ):
        self.tensor_height = tensor_height
        self.tensor_width = tensor_width
        self.offset = offset
        self.sizes, self.strides = TensorTiler2D._validate_and_clean_sizes_strides(
            sizes, strides
        )

    @property
    def dimensions(self) -> list[tuple[int, int]]:
        return list(zip(self.sizes, self.strides))

    def visualize(
        self,
        show_arrows: bool = True,
        show_numbers: bool = False,
        file_path: str | None = None,
        show_plot: bool = True,
        plot_access_count: bool = False,
    ) -> None:
        TensorTiler2D.generate_access_graphs(
            self.tensor_height,
            self.tensor_width,
            self.sizes,
            self.strides,
            offset=self.offset,
            show_arrows=show_arrows,
            show_numbers=show_numbers,
            file_path=file_path,
            show_plot=show_plot,
            plot_access_count=plot_access_count,
        )

    def access_tensors(self) -> tuple[np.ndarray, np.ndarray]:
        return TensorTiler2D.get_access_tensors(
            self.tensor_height,
            self.tensor_width,
            self.sizes,
            self.strides,
            offset=self.offset,
        )

    def access_order(self) -> np.ndarray:
        return TensorTiler2D.get_access_tensors(
            self.tensor_height,
            self.tensor_width,
            self.sizes,
            self.strides,
            offset=self.offset,
        )[0]

    def access_count(self) -> np.ndarray:
        return TensorTiler2D.get_access_tensors(
            self.tensor_height,
            self.tensor_width,
            self.sizes,
            self.strides,
            offset=self.offset,
        )[1]

    def __str__(self) -> str:
        return (
            f"TensorTile(tensor_height={self.tensor_height}, tensor_width={self.tensor_width}, "
            f"offset={self.offset}, sizes={self.sizes}, strides={self.strides}, "
            f"transfer_len={self.transfer_len})"
        )

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return (
                self.tensor_height == other.tensor_height
                and self.tensor_width == other.tensor_width
                and self.offset == other.offset
                and self.sizes == other.sizes
                and self.strides == other.strides
            )
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)


class TensorTile2DIter:
    def __init__(
        self,
        tensor_height: int,
        tensor_width: int,
        sizes: list[int],
        strides: list[int],
        offset_fn: Callable[[int], int],
        num_steps: int,
    ):
        self._num_steps = num_steps
        self._current_step = 0

        self._tensor_height = tensor_height
        self._tensor_width = tensor_width
        self._sizes, self._strides = TensorTiler2D._validate_and_clean_sizes_strides(
            sizes, strides
        )
        self._offset_fn = offset_fn

    def __iter__(self):
        return self

    def __next__(self) -> TensorTile:
        if self._current_step == self._num_steps:
            raise StopIteration
        step = self._current_step
        self._current_step += 1
        return TensorTile(
            self._tensor_height,
            self._tensor_width,
            self._offset_fn(step),
            self._sizes,
            self._strides,
        )


class TensorTiler2D:
    """
    This is an experimental class to help with defining data transformations.
    It is a work in progress.
    """

    DTYPE = np.int32

    def __init__(
        self,
        tensor_height: int,
        tensor_width: int,
        tile_height: int | None = None,
        tile_width: int | None = None,
        tensor_col_major: bool = False,
        tile_col_major: bool = False,
    ):
        self._tensor_height = tensor_height
        self._tensor_width = tensor_width
        self._tile_height = tile_height
        self._tile_width = tile_width
        if tile_height is None or tile_width is None:
            assert (
                tile_height is None and tile_width is None
            ), f"Must supply both or neither of tile_height and tile_width"
            self._tile_height = self._tensor_height
            self._tile_width = self._tensor_width
            tile_col_major = tensor_col_major

        assert (
            self._tensor_height % self._tile_height == 0
        ), f"Tensor height ({self._tensor_height}) must be divisible by tile height ({self._tile_height})"
        assert (
            self._tensor_width % self._tile_width == 0
        ), f"Tensor width ({self._tensor_width}), must be divisible by tile width ({self._tile_width})"

        self._num_tiles_per_row = self._tensor_width // self._tile_width
        self._num_tiles_per_col = self._tensor_height // self._tile_height

        self._tensor_col_major = tensor_col_major
        self._tile_col_major = tile_col_major

        if not self._tile_col_major and (
            self._tensor_col_major or self._tile_width == self._tensor_width
        ):
            # This is a special case where we can express the transformation with a less complicated transform
            self._sizes = [
                1,
                self._num_tiles_per_row,
                self._tensor_height,
                self._tile_width,
            ]
            self._strides = [0, self._tile_width, self._tensor_width, 1]
        elif self._tile_col_major and (
            not self._tensor_col_major or self._tile_height == self._tensor_height
        ):
            # This is a special case where we can express the transformation with a less complicated transform
            self._sizes = [
                1,
                self._num_tiles_per_col,
                self._tensor_width,
                self._tile_height,
            ]
            self._strides = [
                0,
                self._tensor_width * self._tile_height,
                1,
                self._tensor_width,
            ]
        else:
            """
            This is the case that *should* always represent a correct/valid
            transformation (according to my testing using visualization tools).

            It should work even with the special cases above.

            But in my experience, these transformations are not always valid to the NPU
            as stride dimension size may exceed allowable, hence the special cases above.
            """
            self._sizes = [
                self._num_tiles_per_col,
                self._num_tiles_per_row,
                self._tile_height,
                self._tile_width,
            ]
            self._strides = [
                self._tile_width * self._tile_height * self._num_tiles_per_row,
                self._tile_width,
                self._tensor_width,
                1,
            ]

            if self._tensor_col_major:
                self._sizes[0], self._sizes[1] = self._sizes[1], self._sizes[0]
                self._strides[0], self._strides[1] = (
                    self._strides[1],
                    self._strides[0],
                )
            if self._tile_col_major:
                self._sizes[2], self._sizes[3] = self._sizes[3], self._sizes[2]
                self._strides[2], self._strides[3] = (
                    self._strides[3],
                    self._strides[2],
                )
        self._sizes, self._strides = self._validate_and_clean_sizes_strides(
            self._sizes, self._strides
        )

    @property
    def sizes(self) -> list[int]:
        return self._sizes.copy()

    @property
    def strides(self) -> list[int]:
        return self._strides.copy()

    def as_tile(self) -> TensorTile:
        return TensorTile(
            tensor_height=self._tensor_height,
            tensor_width=self._tensor_width,
            offset=0,
            sizes=self._sizes.copy(),
            strides=self._strides.copy(),
        )

    def tile_iter(
        self,
        tile_group_height: int = 1,
        tile_group_width: int = 1,
        tile_repeat: int = 1,
        col_major: bool = False,
    ) -> TensorTile2DIter:
        assert (
            tile_group_height >= 1 and tile_group_width >= 1 and tile_repeat >= 1
        ), f"Tile group height, Tile group width, tile repeat ({tile_group_height}, {tile_group_width}, {tile_repeat}) must be >0"
        assert (
            self._num_tiles_per_row % tile_group_width == 0
        ), f"Tiles per row ({self._num_tiles_per_row}) must be divisible by Tile group width ({tile_group_width})"
        assert (
            self._num_tiles_per_col % tile_group_height == 0
        ), f"Tiles per row ({self._num_tiles_per_col}) must be divisible by Tile group width ({tile_group_height})"
        tile_groups_per_row = self._num_tiles_per_row // tile_group_width
        tile_groups_per_col = self._num_tiles_per_col // tile_group_height

        steps = tile_groups_per_row * tile_groups_per_col

        def calc_offset(iter_num):
            if not col_major:
                row_idx = iter_num % tile_groups_per_row
                col_idx = iter_num // tile_groups_per_row
            else:
                col_idx = iter_num % tile_groups_per_col
                row_idx = iter_num // tile_groups_per_col

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
            # however, it may be needlessly complex (large values in out dimensions)
            size_idx = [0, 1]
            if self._tensor_col_major:
                size_idx = [1, 0]
            iter_sizes[size_idx[0]] = tile_group_height
            iter_sizes[size_idx[1]] = tile_group_width

        iter_sizes, iter_strides = self._validate_and_clean_sizes_strides(
            iter_sizes, iter_strides
        )

        if tile_repeat != 1:
            assert (
                iter_sizes[0] == 1
            ), f"Highest (sizes, strides) dim must be (1, 0) for tile repeat but is ({iter_sizes[0]}, {iter_strides[0]})"
            iter_sizes[0] = tile_repeat

        return TensorTile2DIter(
            self._tensor_height,
            self._tensor_width,
            iter_sizes,
            iter_strides,
            offset_fn=calc_offset,
            num_steps=steps,
        )

    def __str__(self) -> str:
        return f"sizes={self._sizes}, strides={self._strides}"

    @classmethod
    def _generate_access_graphs_from_tensor(
        cls,
        access_order_tensor: np.ndarray,
        access_count_tensor: np.ndarray | None,
        title: str = "Access Order and Access Count",
        show_arrows: bool = True,
        show_numbers: bool = False,
        file_path: str | None = None,
        show_plot: bool = True,
    ):
        try:
            import matplotlib
            import matplotlib.pyplot as plt
            import matplotlib.patheffects as pe
        except:
            raise ImportError(
                "You must pip install matplotlib in order to render access graphs"
            )

        if not (access_count_tensor is None):
            fig, (ax_order, ax_count) = plt.subplots(1, 2)
        else:
            fig, ax_order = plt.subplots()
        _access_heatmap = ax_order.pcolor(access_order_tensor, cmap="gnuplot2")

        # Thanks to https://stackoverflow.com/questions/14406214/moving-x-axis-to-the-top-of-a-plot-in-matplotlib
        # put the major ticks at the middle of each cell, (0, 0) in upper left corner
        ax_order.set_xticks(np.arange(access_order_tensor.shape[1]) + 0.5, minor=False)
        ax_order.set_yticks(np.arange(access_order_tensor.shape[0]) + 0.5, minor=False)
        ax_order.invert_yaxis()
        ax_order.xaxis.tick_top()
        ax_order.set_xticklabels(
            np.arange(0, access_order_tensor.shape[1]), minor=False, rotation="vertical"
        )
        ax_order.set_yticklabels(
            np.arange(0, access_order_tensor.shape[0]), minor=False
        )

        # Add arrows to show access order
        if show_arrows:
            order_dict = {}
            for i in range(access_order_tensor.shape[0]):
                for j in range(access_order_tensor.shape[1]):
                    if access_order_tensor[i, j] != -1:
                        order_dict[access_order_tensor[i, j]] = (i, j)
            order_keys = list(order_dict.keys())
            order_keys.sort()
            for i in range(order_keys[0], order_keys[-1]):
                y1, x1 = order_dict[i]
                y2, x2 = order_dict[i + 1]
                ax_order.arrow(
                    x1 + 0.5,
                    y1 + 0.5,
                    x2 - x1,
                    y2 - y1,
                    length_includes_head=True,
                    head_width=0.1,
                    head_length=0.15,
                    overhang=0.2,
                    path_effects=[pe.withStroke(linewidth=3, foreground="white")],
                )
            ax_order.set_title("Access Order")

        if not (access_count_tensor is None):
            max_count = np.max(access_count_tensor)

            _count_heatmap = ax_count.pcolor(access_count_tensor, cmap="gnuplot2")
            # Thanks to https://stackoverflow.com/questions/14406214/moving-x-axis-to-the-top-of-a-plot-in-matplotlib
            # put the major ticks at the middle of each cell, (0, 0) in upper left corner
            ax_count.set_xticks(
                np.arange(access_count_tensor.shape[1]) + 0.5, minor=False
            )
            ax_count.set_yticks(
                np.arange(access_count_tensor.shape[0]) + 0.5, minor=False
            )
            ax_count.invert_yaxis()
            ax_count.xaxis.tick_top()
            ax_count.set_xticklabels(
                np.arange(0, access_count_tensor.shape[1]),
                minor=False,
                rotation="vertical",
            )
            ax_count.set_yticklabels(
                np.arange(0, access_count_tensor.shape[0]), minor=False
            )
            ax_count.set_title("Access Counts")

        # Add numbers to the plot
        if show_numbers:
            # Thanks to https://stackoverflow.com/questions/37719304/python-imshow-set-certain-value-to-defined-color
            # Thanks to tmdavison answer here https://stackoverflow.com/a/40890587/7871710
            for i in range(access_order_tensor.shape[0]):
                for j in range(access_order_tensor.shape[1]):
                    c = access_order_tensor[i, j]
                    if c != -1:
                        ax_order.text(
                            j + 0.45,
                            i + 0.45,
                            str(c),
                            path_effects=[
                                pe.withStroke(linewidth=3, foreground="white")
                            ],
                        )
                    if not (access_count_tensor is None):
                        c = access_count_tensor[i, j]
                        ax_count.text(
                            j + 0.45,
                            i + 0.45,
                            str(c),
                            path_effects=[
                                pe.withStroke(linewidth=3, foreground="white")
                            ],
                        )

        # plt.title(title)
        if show_plot:
            plt.show()
        if file_path:
            if os.path.exists(file_path):
                print(
                    f"Cannot save plot to {file_path}; file already exists",
                    file=sys.stderr,
                )
            plt.savefig(file_path)

    @classmethod
    def get_access_tensors(
        cls,
        tensor_height: int,
        tensor_width: int,
        sizes: list[int],
        strides: list[int],
        tile_height: int | None = None,
        tile_width: int | None = None,
        offset: int = 0,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        assert tensor_height > 0 and tensor_width > 0, "Tensor dimensions must be > 0"
        assert len(sizes) == 4, "Sizes should be a list of size 4"
        assert len(strides) == 4, "Strides should be a list of size 4"
        assert (tile_height is None and tile_width is None) or (
            (tile_height != None and tile_width != None)
            and (tile_height > 0 and tile_width > 0)
        ), "Tile Height and Tile Width should both be specified, or neither specified"

        # Create access order map
        access_order_tensor = np.full(
            (tensor_height * tensor_width,), -1, dtype=cls.DTYPE
        )

        # Create access count map (if repeat allowed)
        access_count_tensor = np.full(
            (tensor_height * tensor_width,), 0, dtype=cls.DTYPE
        )

        access_count = 0
        for i in range(sizes[0]):
            for j in range(sizes[1]):
                for k in range(sizes[2]):
                    for l in range(sizes[3]):
                        access_idx = (
                            offset
                            + i * strides[0]
                            + j * strides[1]
                            + k * strides[2]
                            + l * strides[3]
                        ) % np.prod(access_count_tensor.shape)
                        access_count_tensor[access_idx] += 1
                        access_order_tensor[access_idx] = access_count
                        access_count += 1

        access_order_tensor = access_order_tensor.reshape((tensor_height, tensor_width))
        access_count_tensor = access_count_tensor.reshape((tensor_height, tensor_width))
        return access_order_tensor, access_count_tensor

    @classmethod
    def generate_access_graphs(
        cls,
        tensor_height: int,
        tensor_width: int,
        sizes: list[int],
        strides: list[int],
        tile_height: int | None = None,
        tile_width: int | None = None,
        offset: int = 0,
        show_arrows: bool = True,
        show_numbers: bool = False,
        file_path: str | None = None,
        show_plot: bool = True,
        plot_access_count: bool = False,
    ):
        access_order_tensor, access_count_tensor = cls.get_access_tensors(
            tensor_height,
            tensor_width,
            sizes,
            strides,
            tile_height=tile_height,
            tile_width=tile_width,
            offset=offset,
        )

        # Show a graph for a single tile
        if tile_height != None and tile_width != None:
            if file_path:
                tile_file_path = file_path + ".tile.png"
            else:
                tile_file_path = None
            cls._generate_access_graphs_from_tensor(
                access_order_tensor[0:tile_height, 0:tile_width],
                (
                    None
                    if not plot_access_count
                    else access_count_tensor[0:tile_height, 0:tile_width]
                ),
                title="Per-Tile Access Order",
                show_arrows=show_arrows,
                show_numbers=show_numbers,
                file_path=tile_file_path,
                show_plot=show_plot,
            )

        cls._generate_access_graphs_from_tensor(
            access_order_tensor,
            (None if not plot_access_count else access_count_tensor),
            show_arrows=show_arrows,
            show_numbers=show_numbers,
            file_path=file_path,
            show_plot=show_plot,
        )

    @classmethod
    def _validate_and_clean_sizes_strides(
        cls, sizes: list[int], strides: list[int]
    ) -> tuple[list[int], list[int]]:
        sizes = sizes.copy()
        strides = strides.copy()

        # Validate sizes/strides
        assert len(sizes) == 4
        assert len(strides) == 4
        for s in sizes:
            assert s >= 1, "Size must be positive"
        for s in strides:
            assert s >= 0, "Stride must be >= 0"

        # Clean (set size=1, stride=0 for as many dims as possible)
        for i in range(3):
            if sizes[i] == 1:
                if i != 3:
                    strides[i] = 0
                else:
                    # smallest stride dim should always be 1
                    strides[i] = 1
            else:
                break
        return sizes, strides

    def visualize(
        self,
        show_tile: bool = True,
        show_arrows: bool = True,
        show_numbers: bool = False,
        file_path: str | None = None,
        show_plot: bool = True,
        plot_access_count: bool = False,
    ) -> None:
        tile_height = self._tile_height if show_tile else None
        tile_width = self._tile_width if show_tile else None
        self.generate_access_graphs(
            self._tensor_height,
            self._tensor_width,
            self._sizes,
            self._strides,
            tile_height=tile_height,
            tile_width=tile_width,
            show_arrows=show_arrows,
            show_numbers=show_numbers,
            file_path=file_path,
            show_plot=show_plot,
            plot_access_count=plot_access_count,
        )

    def access_order(self) -> np.ndarray:
        return self.get_access_tensors(
            self._tensor_height,
            self._tensor_width,
            self._sizes,
            self._strides,
            self._tile_height,
            self._tile_width,
        )[0]

    def access_count(self) -> np.ndarray:
        return self.get_access_tensors(
            self._tensor_height,
            self._tensor_width,
            self._sizes,
            self._strides,
            self._tile_height,
            self._tile_width,
        )[1]
