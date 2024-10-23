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
        transfer_len: int | None = None,
        repeats: bool = False,
    ):
        self.tensor_height = tensor_height
        self.tensor_width = tensor_width
        self.offset = offset
        self.sizes = sizes
        self.strides = strides
        self.repeats = repeats
        self.transfer_len = transfer_len

    @property
    def dimensions(self) -> list[tuple[int, int]]:
        return list(zip(self.sizes, self.strides))

    def visualize(
        self,
        show_arrows: bool = True,
        show_numbers: bool = False,
        file_path: str | None = None,
        show_plot: bool = True,
    ) -> None:
        TensorTiler2D.generate_access_graph(
            self.tensor_height,
            self.tensor_width,
            self.sizes,
            self.strides,
            offset=self.offset,
            show_arrows=show_arrows,
            show_numbers=show_numbers,
            file_path=file_path,
            show_plot=show_plot,
        )

    def access_order(self) -> np.ndarray:
        return TensorTiler2D.get_access_order_tensor(
            self.tensor_height,
            self.tensor_width,
            self.sizes,
            self.strides,
            offset=self.offset,
            allow_repeat=self.repeats,
        )


class TensorTile2DIter:
    def __init__(
        self,
        tensor_height: int,
        tensor_width: int,
        sizes: list[int],
        strides: list[int],
        offset_fn: Callable[[int], int],
        num_steps: int,
        transfer_len: int | None = None,
        repeats: bool = False,
    ):
        self.__num_steps = num_steps
        self.__current_step = 0
        self.__transfer_len = transfer_len
        self.__repeats = repeats

        self.__tensor_height = tensor_height
        self.__tensor_width = tensor_width
        self.__sizes = sizes
        self.__strides = strides
        self.__offset_fn = offset_fn

    def __iter__(self):
        return self

    def __next__(self) -> TensorTile:
        if self.__current_step == self.__num_steps:
            raise StopIteration
        step = self.__current_step
        self.__current_step += 1
        return TensorTile(
            self.__tensor_height,
            self.__tensor_width,
            self.__offset_fn(step),
            self.__sizes,
            self.__strides,
            self.__transfer_len,
            self.__repeats,
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
        self.__tensor_height = tensor_height
        self.__tensor_width = tensor_width
        self.__tile_height = tile_height
        self.__tile_width = tile_width
        if tile_height is None or tile_width is None:
            assert (
                tile_height is None and tile_width is None
            ), f"Must supply both or neither of tile_height and tile_width"
            self.__tile_height = self.__tensor_height
            self.__tile_width = self.__tensor_width
            tile_col_major = tensor_col_major

        assert (
            self.__tensor_height % self.__tile_height == 0
        ), f"Tensor height ({self.__tensor_height}) must be divisible by tile height ({self.__tile_height})"
        assert (
            self.__tensor_width % self.__tile_width == 0
        ), f"Tensor width ({self.__tensor_width}), must be divisible by tile width ({self.__tile_width})"

        self.__num_tiles_per_row = self.__tensor_width // self.__tile_width
        self.__num_tiles_per_col = self.__tensor_height // self.__tile_height

        self.__tensor_col_major = tensor_col_major
        self.__tile_col_major = tile_col_major

        if not self.__tile_col_major and (
            self.__tensor_col_major or self.__tile_width == self.__tensor_width
        ):
            # This is a special case where we can express the transformation with a less complicated transform
            self.__sizes = [
                1,
                self.__num_tiles_per_row,
                self.__tensor_height,
                self.__tile_width,
            ]
            self.__strides = [1, self.__tile_width, self.__tensor_width, 1]
        elif self.__tile_col_major and (
            not self.__tensor_col_major or self.__tile_height == self.__tensor_height
        ):
            # This is a special case where we can express the transformation with a less complicated transform
            self.__sizes = [
                1,
                self.__num_tiles_per_col,
                self.__tensor_width,
                self.__tile_height,
            ]
            self.__strides = [
                1,
                self.__tensor_width * self.__tile_height,
                1,
                self.__tensor_width,
            ]
        else:
            """
            This is the case that *should* always represent a correct/valid
            transformation (according to my modelling using visualization tools).

            It should work even with the special cases above.

            But in my experience, these transformations are not always valid to the NPU
            as stride dimension size may exceed allowable, hence the special cases above.
            """
            self.__sizes = [
                self.__num_tiles_per_col,
                self.__num_tiles_per_row,
                self.__tile_height,
                self.__tile_width,
            ]
            self.__strides = [
                self.__tile_width * self.__tile_height * self.__num_tiles_per_row,
                self.__tile_width,
                self.__tensor_width,
                1,
            ]

            if self.__tensor_col_major:
                self.__sizes[0], self.__sizes[1] = self.__sizes[1], self.__sizes[0]
                self.__strides[0], self.__strides[1] = (
                    self.__strides[1],
                    self.__strides[0],
                )
            if self.__tile_col_major:
                self.__sizes[2], self.__sizes[3] = self.__sizes[3], self.__sizes[2]
                self.__strides[2], self.__strides[3] = (
                    self.__strides[3],
                    self.__strides[2],
                )

    @property
    def sizes(self) -> list[int]:
        return self.__sizes.copy()

    @property
    def strides(self) -> list[int]:
        return self.__strides.copy()

    def tile_iter(
        self,
        chunk_height: int = 1,
        chunk_width: int = 1,
        col_major: bool = False,
    ) -> TensorTile2DIter:
        assert (
            self.__num_tiles_per_row % chunk_width == 0
        ), f"Tiles per row ({self.__num_tiles_per_row}) must be divisible by chunk width ({chunk_width})"
        assert (
            self.__num_tiles_per_col % chunk_height == 0
        ), f"Tiles per row ({self.__num_tiles_per_col}) must be divisible by chunk width ({chunk_height})"

        chunks_per_row = self.__num_tiles_per_row // chunk_width
        chunks_per_col = self.__num_tiles_per_col // chunk_height

        steps = chunks_per_row * chunks_per_col

        def calc_offset(iter_num):
            if not col_major:
                row_idx = iter_num % chunks_per_row
                col_idx = iter_num // chunks_per_row
            else:
                col_idx = iter_num % chunks_per_col
                row_idx = iter_num // chunks_per_col

            offset = row_idx * chunk_width * self.__tile_width
            offset += col_idx * chunk_height * self.__tensor_width * self.__tile_height
            return offset

        iter_sizes = self.__sizes.copy()
        iter_strides = self.__strides.copy()

        if self.__tile_col_major and not self.__tensor_col_major:
            # This is a special case where we can combine a chunk into one logical tile (horizontally)
            iter_sizes[1] = chunk_height
            iter_sizes[2] = chunk_width * self.__tile_width
        elif not self.__tile_col_major and self.__tensor_col_major:
            # This is a special case where we can combine a chunk into one logical tile (vertically)
            iter_sizes[1] = chunk_width
            iter_sizes[2] = chunk_height * self.__tile_height
        elif chunk_width == 1:
            # These are two more special cases; we can combine tiles here too to get a simpler transform
            if self.__tile_col_major:
                iter_sizes = [1, chunk_height, self.__tile_width, self.__tile_height]
                iter_strides = [
                    1,
                    self.__tile_height * self.__tensor_width,
                    1,
                    self.__tensor_width,
                ]
            else:
                iter_sizes = [
                    1,
                    1,
                    self.__tile_height * chunk_height,
                    self.__tile_width,
                ]
                iter_strides = [1, self.__tile_width, self.__tensor_width, 1]
        elif chunk_height == 1:
            # These are two more special cases; we can combine tiles here too to get a simpler transform
            if self.__tile_col_major:
                iter_sizes = [1, 1, self.__tile_width * chunk_width, self.__tile_height]
                iter_strides = [1, 1, 1, self.__tensor_width]
            else:
                iter_sizes = [1, chunk_width, self.__tile_height, self.__tile_width]
                iter_strides = [1, self.__tile_width, self.__tensor_width, 1]
        else:
            # This should always be the case that creates a correct transfrom;
            # however, it may be needlessly complex (large values in out dimensions)
            size_idx = [0, 1]
            if self.__tensor_col_major:
                size_idx = [1, 0]
            iter_sizes[size_idx[0]] = chunk_height
            iter_sizes[size_idx[1]] = chunk_width

        # TODO: for contemplation later
        """
        if chunk_width == 1 and chunk_height == 1:
            iter_strides = [0, 0, 0, 1]
            iter_sizes = [1, 1, self.__tile_width, self.__tile_height]
        """

        if iter_strides[3] != 1:
            print(
                f"WARNING: innermost strides dimension in {iter_strides[3]}, but current hardware requires it to be 1.",
                file=sys.stderr,
            )

        return TensorTile2DIter(
            self.__tensor_height,
            self.__tensor_width,
            iter_sizes,
            iter_strides,
            offset_fn=calc_offset,
            num_steps=steps,
            transfer_len=chunk_width
            * chunk_height
            * self.__tile_height
            * self.__tile_width,
        )

    def __str__(self) -> str:
        return f"sizes={self.__sizes}, strides={self.__strides}"

    @classmethod
    def _generate_access_graph_from_tensor(
        cls,
        access_order_tensor: np.ndarray,
        title: str = "Access Order",
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

        # In inches, this is a little hacky
        # should maybe be defined by the size of the tensor e.g., how many elem per inch
        matplotlib.rcParams["figure.figsize"] = [10, 7]

        _fig, ax = plt.subplots()
        _heatmap = ax.pcolor(access_order_tensor, cmap="gnuplot2")

        # Thanks to https://stackoverflow.com/questions/14406214/moving-x-axis-to-the-top-of-a-plot-in-matplotlib
        # put the major ticks at the middle of each cell, (0, 0) in upper left corner
        ax.set_xticks(np.arange(access_order_tensor.shape[1]) + 0.5, minor=False)
        ax.set_yticks(np.arange(access_order_tensor.shape[0]) + 0.5, minor=False)
        ax.invert_yaxis()
        ax.xaxis.tick_top()
        ax.set_xticklabels(
            np.arange(0, access_order_tensor.shape[1]), minor=False, rotation="vertical"
        )
        ax.set_yticklabels(np.arange(0, access_order_tensor.shape[0]), minor=False)
        plt.title(title)

        # Add numbers to the plot
        if show_numbers:
            # Thanks to https://stackoverflow.com/questions/37719304/python-imshow-set-certain-value-to-defined-color
            # Thanks to tmdavison answer here https://stackoverflow.com/a/40890587/7871710
            for i in range(access_order_tensor.shape[0]):
                for j in range(access_order_tensor.shape[1]):
                    c = access_order_tensor[i, j]
                    if c != -1:
                        ax.text(
                            j + 0.45,
                            i + 0.45,
                            str(c),
                            path_effects=[
                                pe.withStroke(linewidth=3, foreground="white")
                            ],
                        )

        # Add arrows to show access order
        if show_arrows:
            order_dict = {}
            for i in range(access_order_tensor.shape[0]):
                for j in range(access_order_tensor.shape[1]):
                    if access_order_tensor[i, j] != -1:
                        order_dict[access_order_tensor[i, j]] = (i, j)
            for i in range(len(order_dict) - 1):
                y1, x1 = order_dict[i]
                y2, x2 = order_dict[i + 1]
                ax.arrow(
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
    def get_access_order_tensor(
        cls,
        tensor_height: int,
        tensor_width: int,
        sizes: list[int],
        strides: list[int],
        tile_height: int | None = None,
        tile_width: int | None = None,
        offset: int = 0,
        allow_repeat: bool = False,
    ) -> np.ndarray:
        assert tensor_height > 0 and tensor_width > 0, "Tensor dimensions must be > 0"
        assert len(sizes) == 4, "Sizes should be a list of size 4"
        assert len(strides) == 4, "Strides should be a list of size 4"
        assert (tile_height is None and tile_width is None) or (
            (tile_height != None and tile_width != None)
            and (tile_height > 0 and tile_width > 0)
        ), "Tile Height and Tile Width should both be specified, or neither specified"

        # Generate access order map
        access_order_tensor = np.full(
            (tensor_height * tensor_width,), -1, dtype=cls.DTYPE
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
                        )
                        if not allow_repeat:
                            assert (
                                access_order_tensor[access_idx] == -1
                            ), f"Attempted to access index={access_idx} twice."
                        access_order_tensor[access_idx] = access_count
                        access_count += 1
        assert access_count <= np.prod(
            access_order_tensor.shape
        ), f"Access pattern has too many elements (expected max {np.prod(access_order_tensor.shape)}, got {access_count})"
        access_order_tensor = access_order_tensor.reshape((tensor_height, tensor_width))
        return access_order_tensor

    @classmethod
    def generate_access_graph(
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
    ):
        access_order_tensor = cls.get_access_order_tensor(
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
            cls._generate_access_graph_from_tensor(
                access_order_tensor[0:tile_height, 0:tile_width],
                title="Per-Tile Access Order",
                show_arrows=show_arrows,
                show_numbers=show_numbers,
                file_path=tile_file_path,
                show_plot=show_plot,
            )

        cls._generate_access_graph_from_tensor(
            access_order_tensor,
            show_arrows=show_arrows,
            show_numbers=show_numbers,
            file_path=file_path,
            show_plot=show_plot,
        )

    def visualize(
        self,
        show_tile: bool = True,
        show_arrows: bool = True,
        show_numbers: bool = False,
        file_path: str | None = None,
        show_plot: bool = True,
    ) -> None:
        tile_height = self.__tile_height if show_tile else None
        tile_width = self.__tile_width if show_tile else None
        self.generate_access_graph(
            self.__tensor_height,
            self.__tensor_width,
            self.__sizes,
            self.__strides,
            tile_height=tile_height,
            tile_width=tile_width,
            show_arrows=show_arrows,
            show_numbers=show_numbers,
            file_path=file_path,
            show_plot=show_plot,
        )

    def access_order(self) -> np.ndarray:
        # Call class method
        return self.get_access_order_tensor(
            self.__tensor_height,
            self.__tensor_width,
            self.__sizes,
            self.__strides,
            self.__tile_height,
            self.__tile_width,
        )
