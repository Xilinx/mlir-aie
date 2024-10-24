import numpy as np

from aie.helpers.tensortiler.tensortiler2D import TensorTiler2D
from util import construct_test

# RUN: %python %s | FileCheck %s


# CHECK-LABEL: matrix_vector_tiling_sweep
@construct_test
def matrix_vector_tiling_sweep():
    """
    TODO: only porting A and C right now, B requires a tile repeat count which TensorTiler2D does not currently support
    """
    cores_sweep = [1, 2, 4]
    M_sweep = range(512, 4096, 512)
    K_sweep = range(512, 4096, 512)
    m = 32
    k = 32

    for n_cores in cores_sweep:
        for M in M_sweep:
            for K in K_sweep:
                C_sz = M
                M_div_m_div_n_cores = M // (m * n_cores)
                K_div_k = K // k
                C_sz_div_n_cores = C_sz // n_cores
                m_x_K = m * K

                A_tiler = TensorTiler2D(
                    tensor_height=M, tensor_width=K, tile_height=m, tile_width=k
                )
                A_tile_iter = A_tiler.tile_iter(
                    chunk_height=M_div_m_div_n_cores, chunk_width=K // k
                )

                C_tiler = TensorTiler2D(
                    tensor_height=1,
                    tensor_width=C_sz,
                    tile_height=1,
                    tile_width=C_sz_div_n_cores,
                )
                C_tile_iter = C_tiler.tile_iter()

                for i in range(n_cores):
                    # Current way of calculting sizes/strides/offsets
                    A_offset = i * M_div_m_div_n_cores * m * K
                    A_sizes = [M_div_m_div_n_cores, K_div_k, m, k]
                    A_strides = [m_x_K, k, K, 1]

                    # Tile iter way to calculating sizes/strides/offsets
                    A_tile = next(A_tile_iter)
                    if (
                        A_sizes != A_tile.sizes
                        or A_offset != A_tile.offset
                        or A_strides != A_tile.strides
                    ):
                        # There may be different but equivalent transformations
                        reference_access = TensorTiler2D.get_access_order_tensor(
                            M, K, sizes=A_sizes, strides=A_strides, offset=A_offset
                        )
                        new_access = A_tile.access_order()
                        assert (
                            reference_access == new_access
                        ).all(), f"Expected (sizes={A_sizes}, strides={A_strides}, offset={A_offset}), got (sizes={A_tile.sizes}, strides={A_tile.strides}, offset={A_tile.offset})"

                    # Current way of calculting sizes/strides/offsets
                    C_offset = i * M_div_m_div_n_cores * m
                    C_sizes = [1, 1, 1, C_sz_div_n_cores]
                    C_strides = [0, 0, 0, 1]

                    # Tile iter way to calculating sizes/strides/offsets
                    C_tile = next(C_tile_iter)
                    if (
                        C_sizes != C_tile.sizes
                        or C_offset != C_tile.offset
                        or C_strides != C_tile.strides
                    ):
                        # There may be different but equivalent transformations
                        reference_access = TensorTiler2D.get_access_order_tensor(
                            1, C_sz, sizes=C_sizes, strides=C_strides, offset=C_offset
                        )
                        new_access = C_tile.access_order()
                        assert (
                            reference_access == new_access
                        ).all(), f"Expected (sizes={C_sizes}, strides={C_strides}, offset={C_offset}), got (sizes={C_tile.sizes}, strides={C_tile.strides}, offset={C_tile.offset})"
    # CHECK: Pass!
    print("Pass!")
