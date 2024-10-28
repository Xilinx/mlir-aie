import numpy as np

from aie.helpers.tensortiler.tensortiler2d import TensorTiler2D
from util import construct_test

# RUN: %python %s | FileCheck %s


# CHECK-LABEL: matrix_vector_tiling_sweep
@construct_test
def matrix_vector_tiling_sweep():
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
                    tile_group_height=M_div_m_div_n_cores, tile_group_width=K // k
                )

                B_tiler = TensorTiler2D(
                    tensor_height=1, tensor_width=K, tile_height=1, tile_width=K
                )
                B_tile = next(B_tiler.tile_iter(tile_repeat=M_div_m_div_n_cores))

                C_tiler = TensorTiler2D(
                    tensor_height=1,
                    tensor_width=C_sz,
                    tile_height=1,
                    tile_width=C_sz_div_n_cores,
                )
                C_tile_iter = C_tiler.tile_iter()

                B_sizes = [M_div_m_div_n_cores, 1, 1, K]
                B_strides = [0, 0, 0, 1]
                if B_sizes != B_tile.sizes or B_strides != B_tile.strides:
                    reference_access, reference_count = (
                        TensorTiler2D.get_access_tensors(
                            1, K, sizes=B_sizes, strides=B_strides
                        )
                    )
                    new_access, new_count = B_tile.access_tensors()
                    assert (reference_access == new_access).all(), (
                        f"B access orders do not match. "
                        "Expected (sizes={B_sizes}, strides={B_strides}), got (sizes={B_tile.sizes}, strides={B_tile.strides})"
                    )
                    assert (reference_count == new_count).all(), (
                        f"B access counts do not match. "
                        "Expected (sizes={B_sizes}, strides={B_strides}), got (sizes={B_tile.sizes}, strides={B_tile.strides})"
                    )

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
                        reference_access, reference_count = (
                            TensorTiler2D.get_access_tensors(
                                M, K, sizes=A_sizes, strides=A_strides, offset=A_offset
                            )
                        )
                        new_access, new_count = A_tile.access_tensors()
                        assert (reference_access == new_access).all(), (
                            f"A access orders do not match. "
                            "Expected (sizes={A_sizes}, strides={A_strides}, offset={A_offset}), got (sizes={A_tile.sizes}, strides={A_tile.strides}, offset={A_tile.offset})"
                        )
                        assert (reference_count == new_count).all(), (
                            f"A access counts do not match. "
                            "Expected (sizes={A_sizes}, strides={A_strides}, offset={A_offset}), got (sizes={A_tile.sizes}, strides={A_tile.strides}, offset={A_tile.offset})"
                        )

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
                        reference_access, reference_count = (
                            TensorTiler2D.get_access_tensors(
                                1,
                                C_sz,
                                sizes=C_sizes,
                                strides=C_strides,
                                offset=C_offset,
                            )
                        )
                        new_access, new_count = C_tile.access_tensors()
                        assert (reference_access == new_access).all(), (
                            f"C access orders do not match. "
                            "Expected (sizes={C_sizes}, strides={C_strides}, offset={C_offset}), got (sizes={C_tile.sizes}, strides={C_tile.strides}, offset={C_tile.offset})"
                        )
                        assert (reference_access == new_count).all(), (
                            f"C access counts do not match. "
                            "Expected (sizes={C_sizes}, strides={C_strides}, offset={C_offset}), got (sizes={C_tile.sizes}, strides={C_tile.strides}, offset={C_tile.offset})"
                        )

    # CHECK: Pass!
    print("Pass!")
