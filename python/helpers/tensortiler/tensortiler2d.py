import numpy as np


class TensorTile2DIter:
    def __init__(self, sizes, strides, offset_fn, num_steps):
        self.__num_steps = num_steps
        self.__current_step = 0

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
        return (self.__offset_fn(step), self.__sizes, self.__strides)


class TensorTiler2D:
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

        # For row-major tiling for row-major tiles
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

        self.__tensor_col_major = tensor_col_major
        if tensor_col_major:
            self.__sizes[0], self.__sizes[1] = self.__sizes[1], self.__sizes[0]
            self.__strides[0], self.__strides[1] = self.__strides[1], self.__strides[0]
        if tile_col_major:
            self.__sizes[2], self.__sizes[3] = self.__sizes[3], self.__sizes[2]
            self.__strides[2], self.__strides[3] = self.__strides[3], self.__strides[2]

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

        size_idx = [0, 1]
        if self.__tensor_col_major:
            size_idx = [1, 0]
        iter_sizes[size_idx[0]] = chunk_height
        iter_sizes[size_idx[1]] = chunk_width
        # TODO: handle row/col major

        return TensorTile2DIter(
            iter_sizes, self.__strides, offset_fn=calc_offset, num_steps=steps
        )

    def __str__(self):
        return f"sizes={self.__sizes}, strides={self.__strides}"

    @classmethod
    def access_heatmap(
        cls,
        tensor_height: int,
        tensor_width: int,
        tile_height: int,
        tile_width: int,
        sizes: list[int],
        strides: list[int],
        offset: int = 0,
        show_tile: bool = True,
    ):
        try:
            import matplotlib.pyplot as plt
        except:
            raise ImportError(
                "You must pip install matplotlib in order to use the access_heatmap() method"
            )

        access_order_tensor = np.zeros(tensor_height * tensor_width, dtype=np.int32)
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
        if show_tile:
            print("Per Tile:")
            per_tile_access_order = access_order_tensor[0:tile_height, 0:tile_width]
            plt.imshow(per_tile_access_order, cmap="gist_heat", interpolation="nearest")
            plt.show()
        print("Per Tensor:")
        plt.imshow(access_order_tensor, cmap="turbo", interpolation="nearest")
        plt.show()
