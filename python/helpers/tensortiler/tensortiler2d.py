import numpy as np
import matplotlib as plt


class TensorTiler2D:
    def __init__(
        self,
        tensor_height: int,
        tensor_width: int,
        tile_height: int,
        tile_width: int,
        tensor_row_major=True,
        tile_row_major=True,
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

    def __str__(self):
        return f"sizes={self.__sizes}, strides={self.__strides}"

    def access_heatmap(self):
        access_order_tensor = np.zeros(
            self.__tensor_height * self.__tensor_width, dtype=np.int32
        )
        access_count = 0
        for i in range(self.__sizes[0]):
            for j in range(self.__sizes[1]):
                for k in range(self.__sizes[2]):
                    for l in range(self.__sizes[3]):
                        access_order_tensor[
                            i * self.__strides[0]
                            + j * self.__strides[1]
                            + k * self.__strides[2]
                            + l * self.__strides[3]
                        ] = access_count
                        access_count += 1
        access_order_tensor = access_order_tensor.reshape(
            (self.__tensor_height, self.__tensor_width)
        )
        print("Per Tile:")
        per_tile_access_order = access_order_tensor[
            0 : self.__tile_height, 0 : self.__tile_width
        ]
        plt.imshow(per_tile_access_order, cmap="hot", interpolation="nearest")
        plt.show()
        print("Per Tensor:")
        plt.imshow(access_order_tensor, cmap="hot", interpolation="nearest")
        plt.show()
