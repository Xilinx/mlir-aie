from aie.helpers.taplib import TensorAccessPattern, TensorTiler2D
from util import construct_test

# RUN: %python %s | FileCheck %s


# CHECK-LABEL: matrix_vector_tiling_sweep
@construct_test
def matrix_vector_tiling_sweep():
    # Note: reduced number of tests to reduce time to run tests
    cores_sweep = [1, 2, 4]
    M_sweep = [512, 1024, 1536]  # range(512, 4096, 512)
    K_sweep = [512, 1024, 1536]  # range(512, 4096, 512)
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

                A_iter = iter(
                    TensorTiler2D.group_tiler(
                        (M, K), (m, k), (M_div_m_div_n_cores, K // k)
                    )
                )
                B_tap = TensorTiler2D.simple_tiler(
                    (1, K), (1, K), pattern_repeat=M_div_m_div_n_cores
                )[0]
                C_iter = iter(
                    TensorTiler2D.simple_tiler((1, C_sz), (1, C_sz_div_n_cores))
                )

                B_sizes = [M_div_m_div_n_cores, 1, 1, K]
                B_strides = [0, 0, 0, 1]
                if B_sizes != B_tap.sizes or B_strides != B_tap.strides:
                    B_tap_ref = TensorAccessPattern(
                        (1, K), offset=0, sizes=B_sizes, strides=B_strides
                    )
                    assert B_tap.compare_access_orders(
                        B_tap_ref
                    ), f"B tile {B_tap} and ref tile {B_tap_ref} are not functionally equivalent."

                for i in range(n_cores):
                    # Current way of calculting sizes/strides/offsets
                    A_offset = i * M_div_m_div_n_cores * m * K
                    A_sizes = [M_div_m_div_n_cores, K_div_k, m, k]
                    A_strides = [m_x_K, k, K, 1]

                    # Tile iter way to calculating sizes/strides/offsets
                    A_tap = next(A_iter)
                    if (
                        A_sizes != A_tap.sizes
                        or A_offset != A_tap.offset
                        or A_strides != A_tap.strides
                    ):
                        # There may be different but equivalent transformations
                        A_tap_ref = TensorAccessPattern(
                            (M, K), offset=A_offset, sizes=A_sizes, strides=A_strides
                        )
                        assert A_tap.compare_access_orders(
                            A_tap_ref
                        ), f"A tile {A_tap} and ref tile {A_tap_ref} are not functionally equivalent."

                    # Current way of calculting sizes/strides/offsets
                    C_offset = i * M_div_m_div_n_cores * m
                    C_sizes = [1, 1, 1, C_sz_div_n_cores]
                    C_strides = [0, 0, 0, 1]

                    # Tile iter way to calculating sizes/strides/offsets
                    C_tap = next(C_iter)
                    if (
                        C_sizes != C_tap.sizes
                        or C_offset != C_tap.offset
                        or C_strides != C_tap.strides
                    ):
                        # There may be different but equivalent transformations
                        C_tap_ref = TensorAccessPattern(
                            (1, C_sz), offset=C_offset, sizes=C_sizes, strides=C_strides
                        )
                        assert C_tap.compare_access_orders(
                            C_tap_ref
                        ), f"C tile {C_tap} and ref tile {C_tap_ref} are not functionally equivalent."

    # CHECK: Pass!
    print("Pass!")
