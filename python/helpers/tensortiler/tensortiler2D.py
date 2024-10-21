import numpy as np


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
        self.sizes = sizes
        self.strides = strides

    def visualize(self, show_arrows: bool = True, show_numbers: bool = False):
        TensorTiler2D.access_heatmap(
            self.tensor_height,
            self.tensor_width,
            self.sizes,
            self.strides,
            offset=self.offset,
            show_arrows=show_arrows,
            show_numbers=show_numbers,
        )

    def access_map(self):
        return TensorTiler2D.access_order_map(
            self.tensor_height,
            self.tensor_width,
            self.sizes,
            self.strides,
            offset=self.offset,
        )


class TensorTile2DIter:
    def __init__(
        self,
        tensor_height: int,
        tensor_width: int,
        sizes: list[int],
        strides: list[int],
        offset_fn,
        num_steps: int,
    ):
        self.__num_steps = num_steps
        self.__current_step = 0

        self.__tensor_height = tensor_height
        self.__tensor_width = tensor_width
        self.__sizes = sizes
        self.__strides = strides
        self.__offset_fn = offset_fn

    def __iter__(self):
        return self

    def __next__(self):
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
        )


class TensorTiler2D:
    """
    This class tries really hard to keep the innermost stride dimension as 1,
    but is not always successful.
    """

    def __init__(
        self,
        tensor_height: int,
        tensor_width: int,
        tile_height: int,
        tile_width: int,
        tensor_col_major=False,
        tile_col_major=False,
    ):
        assert tensor_height % tile_height == 0
        assert tensor_width % tile_width == 0

        self.__tensor_height = tensor_height
        self.__tensor_width = tensor_width
        self.__tile_height = tile_height
        self.__tile_width = tile_width

        self.__num_tiles_per_row = self.__tensor_width // self.__tile_width
        self.__num_tiles_per_column = self.__tensor_height // self.__tile_height

        self.__tensor_col_major = tensor_col_major
        self.__tile_col_major = tile_col_major

        if not self.__tile_col_major and (
            self.__tensor_col_major or self.__tile_width == self.__tensor_width
        ):
            # This will work in one piece
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
            # This will also work in one piece
            self.__sizes = [
                1,
                self.__num_tiles_per_column,
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
            # These cases need to be done either column by column or row by row
            self.__sizes = [
                self.__num_tiles_per_column,
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
    def sizes(self):
        return self.__sizes.copy()

    @property
    def strides(self):
        return self.__strides.copy()

    def tile_iter(self, chunk_height: int = 1, chunk_width: int = 1, col_major=False):
        assert self.__num_tiles_per_row % chunk_width == 0
        assert self.__num_tiles_per_column % chunk_height == 0

        chunks_per_row = self.__num_tiles_per_row // chunk_width
        chunks_per_column = self.__num_tiles_per_column // chunk_height

        steps = chunks_per_row * chunks_per_column

        def calc_offset(iter_num):
            if not col_major:
                row_idx = iter_num % chunks_per_row
                col_idx = iter_num // chunks_per_row
            else:
                col_idx = iter_num % chunks_per_column
                row_idx = iter_num // chunks_per_column

            offset = row_idx * chunk_width * self.__tile_width
            offset += col_idx * chunk_height * self.__tensor_width * self.__tile_height
            return offset

        iter_sizes = self.__sizes.copy()
        iter_strides = self.__strides.copy()

        if self.__tile_col_major and not self.__tensor_col_major:
            iter_sizes[1] = chunk_height
            iter_sizes[2] = chunk_width * self.__tile_width
        elif not self.__tile_col_major and self.__tensor_col_major:
            iter_sizes[1] = chunk_width
            iter_sizes[2] = chunk_height * self.__tile_height
        elif chunk_width == 1 and not self.__tile_col_major:
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
            if self.__tile_col_major:
                iter_sizes = [1, 1, self.__tile_width * chunk_width, self.__tile_height]
                iter_strides = [1, 1, 1, self.__tensor_width]
            else:
                iter_sizes = [1, chunk_width, self.__tile_height, self.__tile_width]
                iter_strides = [1, self.__tile_width, self.__tensor_width, 1]
        else:
            size_idx = [0, 1]
            if self.__tensor_col_major:
                size_idx = [1, 0]
            iter_sizes[size_idx[0]] = chunk_height
            iter_sizes[size_idx[1]] = chunk_width

        return TensorTile2DIter(
            self.__tensor_height,
            self.__tensor_width,
            iter_sizes,
            iter_strides,
            offset_fn=calc_offset,
            num_steps=steps,
        )

    def __str__(self):
        return f"sizes={self.__sizes}, strides={self.__strides}"

    @classmethod
    def _generate_access_map(
        cls,
        access_order_map: type[np.ndarray],
        title="Access Order",
        show_arrows=True,
        show_numbers=False,
    ):
        try:
            import matplotlib
            import matplotlib.pyplot as plt
            import matplotlib.patheffects as pe
        except:
            raise ImportError(
                "You must pip install matplotlib in order to render visual access maps"
            )

        # In inches
        matplotlib.rcParams["figure.figsize"] = [10, 7]

        _fig, ax = plt.subplots()
        _heatmap = ax.pcolor(access_order_map, cmap="gnuplot2")

        # Thanks to https://stackoverflow.com/questions/14406214/moving-x-axis-to-the-top-of-a-plot-in-matplotlib
        # put the major ticks at the middle of each cell, (0, 0) in upper left corner
        ax.set_xticks(np.arange(access_order_map.shape[1]) + 0.5, minor=False)
        ax.set_yticks(np.arange(access_order_map.shape[0]) + 0.5, minor=False)
        ax.invert_yaxis()
        ax.xaxis.tick_top()
        ax.set_xticklabels(
            np.arange(0, access_order_map.shape[1]), minor=False, rotation="vertical"
        )
        ax.set_yticklabels(np.arange(0, access_order_map.shape[0]), minor=False)
        plt.title(title)

        # add numbers to the plot
        if show_numbers:
            # thanks to https://stackoverflow.com/questions/37719304/python-imshow-set-certain-value-to-defined-color
            # thanks to tmdavison answer here https://stackoverflow.com/a/40890587/7871710
            for i in range(access_order_map.shape[0]):
                for j in range(access_order_map.shape[1]):
                    c = access_order_map[i, j]
                    if c != -1:
                        ax.text(
                            j + 0.45,
                            i + 0.45,
                            str(c),
                            path_effects=[
                                pe.withStroke(linewidth=3, foreground="white")
                            ],
                        )

        # add arrows to show access order
        if show_arrows:
            order_dict = {}
            for i in range(access_order_map.shape[0]):
                for j in range(access_order_map.shape[1]):
                    if access_order_map[i, j] != -1:
                        order_dict[access_order_map[i, j]] = (i, j)
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
        plt.show()

    @classmethod
    def access_order_map(
        cls,
        tensor_height: int,
        tensor_width: int,
        sizes: list[int],
        strides: list[int],
        tile_height: int | None = None,
        tile_width: int | None = None,
        offset: int = 0,
    ):
        assert tensor_height > 0 and tensor_width > 0
        assert len(sizes) == 4
        assert len(strides) == 4
        assert (tile_height is None and tile_width is None) or (
            (tile_height != None and tile_width != None)
            and (tile_height > 0 and tile_width > 0)
        )

        # Generate access order map
        access_order_tensor = np.full(
            (tensor_height * tensor_width,), -1, dtype=np.int32
        )
        access_count = 0
        for i in range(sizes[0]):
            for j in range(sizes[1]):
                for k in range(sizes[2]):
                    for l in range(sizes[3]):
                        access_order_tensor[
                            offset
                            + i * strides[0]
                            + j * strides[1]
                            + k * strides[2]
                            + l * strides[3]
                        ] = access_count
                        access_count += 1
        access_order_tensor = access_order_tensor.reshape((tensor_height, tensor_width))
        return access_order_tensor

    @classmethod
    def access_heatmap(
        cls,
        tensor_height: int,
        tensor_width: int,
        sizes: list[int],
        strides: list[int],
        tile_height: int | None = None,
        tile_width: int | None = None,
        offset: int = 0,
        show_arrows=True,
        show_numbers=False,
    ):
        access_order_tensor = cls.access_order_map(
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
            cls._generate_access_map(
                access_order_tensor[0:tile_height, 0:tile_width],
                title="Per-Tile Access Order",
                show_arrows=show_arrows,
                show_numbers=show_numbers,
            )

        cls._generate_access_map(
            access_order_tensor, show_arrows=show_arrows, show_numbers=show_numbers
        )

    def visualize(
        self,
        show_tile: bool = True,
        show_arrows: bool = True,
        show_numbers: bool = False,
    ):
        tile_height = self.__tile_height if show_tile else None
        tile_width = self.__tile_width if show_tile else None
        self.access_heatmap(
            self.__tensor_height,
            self.__tensor_width,
            self.__sizes,
            self.__strides,
            tile_height=tile_height,
            tile_width=tile_width,
            show_arrows=show_arrows,
            show_numbers=show_numbers,
        )

    def access_map(self):
        return self.access_order_map(
            self.__tensor_height,
            self.__tensor_width,
            self.__sizes,
            self.__strides,
            self.__tile_height,
            self.__tile_width,
        )
