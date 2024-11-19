#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2023 AMD Inc.

import sys
import argparse

from aie.extras.context import mlir_mod_ctx

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.dialects.scf import *
import aie.dialects.index as index_dialect
import aie.dialects.arith as arith_dialect
import aie.dialects.memref as memref_dialect

from util import *


def get_memref_len_elems(memref):
    out = 1
    for s in memref.shape:
        out *= s
    return out


def main():
    argparser = argparse.ArgumentParser(
        prog="AIE Matrix Multiplication MLIR Design (Whole Array)",
        description="Emits MLIR code for a matrix multiplication design of the given input size",
    )
    argparser.add_argument("-M", type=int, default=512)
    argparser.add_argument("-K", type=int, default=512)
    argparser.add_argument("-N", type=int, default=512)
    argparser.add_argument("-m", type=int, default=64)
    argparser.add_argument("-k", type=int, default=64)
    argparser.add_argument("-n", type=int, default=32)
    argparser.add_argument("--n-aie-cols", type=int, choices=[1, 2, 4], default=4)
    argparser.add_argument(
        "--dtype_in", type=str, choices=["bf16", "i16"], default="i16"
    )
    argparser.add_argument(
        "--dtype_out", type=str, choices=["bf16", "i16", "f32", "i32"], default="i16"
    )
    args = argparser.parse_args()
    with mlir_mod_ctx() as ctx:
        my_matmul(
            args.M,
            args.K,
            args.N,
            args.m,
            args.k,
            args.n,
            args.n_aie_cols,
            args.dtype_in,
            args.dtype_out,
        )
        # print(ctx.module.operation.verify())
        print(ctx.module)


def ceildiv(a, b):
    return (a + b - 1) // b


def my_matmul(M, K, N, m, k, n, n_aie_cols, dtype_in_str, dtype_out_str):

    n_aie_rows = 4
    n_aie_cores = n_aie_rows * n_aie_cols

    dtype_in = None
    if dtype_in_str == "bf16":
        dtype_in = T.bf16
    elif dtype_in_str == "i16":
        dtype_in = T.i16
    dtype_out = None
    if dtype_out_str == "bf16":
        dtype_out = T.bf16
    elif dtype_out_str == "i16":
        dtype_out = T.i16
    elif dtype_out_str == "f32":
        dtype_out = T.f32
    elif dtype_out_str == "i32":
        dtype_out = T.i32

    assert dtype_in == T.bf16
    assert dtype_out == T.f32

    if dtype_in_str == "bf16":
        r = 4
        s = 8
        t = 4
    elif dtype_in_str == "i16":
        r = 4
        s = 4
        t = 4

    # Input matrix A:
    # Conceptually, we divide input A into (m * n_rows, k)-sized blocks. These
    # blocks are _broadcast_ across AIE core columns, then _distributed_ across
    # rows, s.t. each of the n_rows compute cores in a column receives a
    # contiguous (m, k)-sized block of A.
    assert (
        M % (m * n_aie_rows) == 0
    ), """A must be tileable into (m * n_aie_rows, k)-sized blocks"""

    # Both A and B are tiled in the K dimension into size k.
    assert K % k == 0

    # Input matrix B:
    # Conceptually, we do the same as with A, but instead of broadcasting
    # across columns we broadcast across rows and distribute across columns.
    assert (
        N % (n * n_aie_cols) == 0
    ), """B must be tileable into (k, n * n_aie_cols)-sized blocks"""

    # r, s, t are the dimensions required by the microkernel MAC instructions.
    assert m % r == 0
    assert k % s == 0
    assert n % t == 0

    # If you get errors during CDO generation due to running out of program
    # memory, it may be because too much code is generated due to ObjectFIFO
    # loop unrollings. Reducing the depth to 1 here will work around that at
    # a big performance cost.
    fifo_depth = 2

    n_tiles_per_core = (M // m) * (N // n) // n_aie_cores

    n_A_tiles_per_shim = n_aie_rows // n_aie_cols

    dev = None
    if n_aie_cols == 1:
        dev = AIEDevice.npu1_1col
    elif n_aie_cols == 2:
        dev = AIEDevice.npu1_2col
    elif n_aie_cols == 4:
        dev = AIEDevice.npu1_4col

    @device(dev)
    def device_body():
        A_l3_memref_ty = T.memref(M * K, dtype_in())
        B_l3_memref_ty = T.memref(K * N, dtype_in())
        C_l3_memref_ty = T.memref(M * N, dtype_out())
        A_l2_memref_ty = T.memref(m * k * n_A_tiles_per_shim, dtype_in())
        B_l2_memref_ty = T.memref(k * n, dtype_in())
        C_l2_memref_ty = T.memref(m * n * n_aie_rows, dtype_out())
        A_l1_memref_ty = T.memref(m, k, dtype_in())
        B_l1_memref_ty = T.memref(k, n, dtype_in())
        C_l1_memref_ty = T.memref(m, n, dtype_out())
        rtp_ty = T.memref(3, T.i32())

        # AIE Core Function declarations
        zero_scalar = external_func(
            f"zero_scalar_{dtype_out_str}", inputs=[C_l1_memref_ty]
        )
        zero = external_func(f"zero_{dtype_out_str}", inputs=[C_l1_memref_ty])
        matmul_scalar = external_func(
            f"matmul_scalar_{dtype_in_str}_{dtype_out_str}",
            inputs=[A_l1_memref_ty, B_l1_memref_ty, C_l1_memref_ty],
        )
        matmul = external_func(
            f"matmul_{dtype_in_str}_{dtype_out_str}",
            inputs=[A_l1_memref_ty, B_l1_memref_ty, C_l1_memref_ty],
        )
        await_rtp = external_func(f"await_rtp", inputs=[rtp_ty])
        get_volatile_rtp = external_func(
            f"get_volatile_rtp", inputs=[rtp_ty, T.i32()], outputs=[T.i32()]
        )

        # Tile declarations as tile[row][col]
        tiles = [
            [tile(col, row) for col in range(0, n_aie_cols)] for row in range(0, 6)
        ]
        shim_tiles = tiles[0]
        mem_tiles = tiles[1]
        core_tiles = tiles[2:]

        # Run time parameter K//k
        rtp_bufs = [[None] * n_aie_cols for _ in range(4)]
        for col in range(n_aie_cols):
            for row in range(n_aie_rows):
                # RTP index 0: "ready" signal
                # RTP index 1: K // k // 2
                rtp_bufs[row][col] = buffer(
                    core_tiles[row][col],
                    datatype=T.memref(3, T.i32()),
                    name=f"rtp_{row}_{col}",
                )

        # AIE-array data movement with object fifos
        A_l3l2_fifos = [None] * n_aie_cols
        A_l2l1_fifos = [None] * n_aie_rows

        B_l3l2_fifos = [None] * n_aie_cols
        B_l2l1_fifos = [None] * n_aie_cols

        C_l1l2_fifos = [[None] * n_aie_cols for _ in range(n_aie_rows)]
        C_l2l3_fifos = [None] * n_aie_cols

        # Input A, L2 -> L1
        for row in range(n_aie_rows):
            mem_tile = mem_tiles[row // n_A_tiles_per_shim]
            A_l2l1_fifos[row] = {
                "prod": {
                    "endpoint": (mem_tile, WireBundle.DMA, 0),
                    "ping_buf": buffer(
                        mem_tile,
                        datatype=A_l2_memref_ty,
                        name=f"A_L3L2_{row}_cons_buff_0",
                    ),
                    "pong_buf": buffer(
                        mem_tile,
                        datatype=A_l2_memref_ty,
                        name=f"A_L3L2_{row}_cons_buff_1",
                    ),
                    "put_lock": lock(
                        mem_tile,
                        init=2,
                        sym_name=f"A_L3L2_{row}_cons_prod_lock",
                        lock_id=0,
                    ),
                    "get_lock": lock(
                        mem_tile,
                        init=0,
                        sym_name=f"A_L3L2_{row}_cons_cons_lock",
                        lock_id=1,
                    ),
                },
                "cons": [
                    {
                        "endpoint": (core_tiles[row][col], WireBundle.DMA, 0),
                        "ping_buf": buffer(
                            core_tiles[row][col],
                            datatype=A_l1_memref_ty,
                            name=f"A_L2L1_{row}_{col}_cons_buff_0",
                        ),
                        "pong_buf": buffer(
                            core_tiles[row][col],
                            datatype=A_l1_memref_ty,
                            name=f"A_L2L1_{row}_{col}_cons_buff_1",
                        ),
                        "put_lock": lock(
                            core_tiles[row][col],
                            init=2,
                            sym_name=f"A_L2L1_{row}_{col}_cons_prod_lock",
                            lock_id=0,
                        ),
                        "get_lock": lock(
                            core_tiles[row][col],
                            init=0,
                            sym_name=f"A_L2L1_{row}_{col}_cons_cons_lock",
                            lock_id=1,
                        ),
                    }
                    for col in range(n_aie_cols)
                ],  # broadcast along one row
            }
            for col in range(n_aie_cols):
                src_tile, src_bundle, src_channel = A_l2l1_fifos[row]["prod"][
                    "endpoint"
                ]
                dst_tile, dst_bundle, dst_channel = A_l2l1_fifos[row]["cons"][col][
                    "endpoint"
                ]
                flow(
                    src_tile, src_bundle, src_channel, dst_tile, dst_bundle, dst_channel
                )

        # Input A, L3 -> L2
        for col in range(n_aie_cols):
            shim_tile = shim_tiles[col]
            mem_tile = mem_tiles[col]
            A_l3l2_fifos[col] = {
                "prod": {
                    "endpoint": (shim_tile, WireBundle.DMA, 0),
                    "shim_memref": memref_dialect.global_(
                        sym_name=f"A_L3L2_{col}",
                        sym_visibility="public",
                        type_=A_l3_memref_ty,
                    ),
                    "shim_dma_alloc": ShimDMAAllocationOp(
                        f"A_L3L2_{col}", DMAChannelDir.MM2S, 0, col=col
                    ),
                },
                "cons": {
                    "endpoint": (mem_tile, WireBundle.DMA, 0),
                },
            }
            src_tile, src_bundle, src_channel = A_l3l2_fifos[col]["prod"]["endpoint"]
            dst_tile, dst_bundle, dst_channel = A_l3l2_fifos[col]["cons"]["endpoint"]
            flow(src_tile, src_bundle, src_channel, dst_tile, dst_bundle, dst_channel)

        # Input B, L2 -> L1
        for col in range(n_aie_cols):
            mem_tile = mem_tiles[col]
            B_l2l1_fifos[col] = {
                "prod": {
                    "endpoint": (mem_tile, WireBundle.DMA, 1),
                    "ping_buf": buffer(
                        mem_tile,
                        datatype=B_l2_memref_ty,
                        name=f"B_L3L2_{col}_cons_buff_0",
                    ),
                    "pong_buf": buffer(
                        mem_tile,
                        datatype=B_l2_memref_ty,
                        name=f"B_L3L2_{col}_cons_buff_1",
                    ),
                    "put_lock": lock(
                        mem_tile,
                        init=2,
                        sym_name=f"B_L3L2_{col}_cons_prod_lock",
                        lock_id=2,
                    ),
                    "get_lock": lock(
                        mem_tile,
                        init=0,
                        sym_name=f"B_L3L2_{col}_cons_cons_lock",
                        lock_id=3,
                    ),
                },
                "cons": [
                    {
                        "endpoint": (core_tiles[row][col], WireBundle.DMA, 1),
                        "ping_buf": buffer(
                            core_tiles[row][col],
                            datatype=B_l1_memref_ty,
                            name=f"B_L2L1_{col}_{row}_cons_buff_0",
                        ),
                        "pong_buf": buffer(
                            core_tiles[row][col],
                            datatype=B_l1_memref_ty,
                            name=f"B_L2L1_{col}_{row}_cons_buff_1",
                        ),
                        "put_lock": lock(
                            core_tiles[row][col],
                            init=2,
                            sym_name=f"B_L2L1_{col}_{row}_cons_prod_lock",
                            lock_id=2,
                        ),
                        "get_lock": lock(
                            core_tiles[row][col],
                            init=0,
                            sym_name=f"B_L2L1_{col}_{row}_cons_cons_lock",
                            lock_id=3,
                        ),
                    }
                    for row in range(n_aie_rows)
                ],  # broadcast along one column
            }
            for row in range(n_aie_rows):
                src_tile, src_bundle, src_channel = B_l2l1_fifos[col]["prod"][
                    "endpoint"
                ]
                dst_tile, dst_bundle, dst_channel = B_l2l1_fifos[col]["cons"][row][
                    "endpoint"
                ]
                flow(
                    src_tile, src_bundle, src_channel, dst_tile, dst_bundle, dst_channel
                )

        # Input B, L3 -> L2
        for col in range(n_aie_cols):
            mem_tile = mem_tiles[col]
            shim_tile = shim_tiles[col]
            B_l3l2_fifos[col] = {
                "prod": {
                    "endpoint": (shim_tile, WireBundle.DMA, 1),
                    "shim_memref_dialect": memref_dialect.global_(
                        sym_name=f"B_L3L2_{col}",
                        sym_visibility="public",
                        type_=B_l3_memref_ty,
                    ),
                    "shim_dma_alloc": ShimDMAAllocationOp(
                        f"B_L3L2_{col}", DMAChannelDir.MM2S, 1, col=col
                    ),
                },
                "cons": {"endpoint": (mem_tile, WireBundle.DMA, 1)},
            }
            src_tile, src_bundle, src_channel = B_l3l2_fifos[col]["prod"]["endpoint"]
            dst_tile, dst_bundle, dst_channel = B_l3l2_fifos[col]["cons"]["endpoint"]
            flow(src_tile, src_bundle, src_channel, dst_tile, dst_bundle, dst_channel)

        # Output C, L1 -> L2
        for col in range(n_aie_cols):
            for row in range(n_aie_rows):
                C_l1l2_fifos[row][col] = {
                    "prod": {
                        "endpoint": (core_tiles[row][col], WireBundle.DMA, 0),
                        "ping_buf": buffer(
                            core_tiles[row][col],
                            datatype=C_l1_memref_ty,
                            name=f"C_L1L2_{col}_{row}_buff_0",
                        ),
                        "pong_buf": buffer(
                            core_tiles[row][col],
                            datatype=C_l1_memref_ty,
                            name=f"C_L1L2_{col}_{row}_buff_1",
                        ),
                        "put_lock": lock(
                            core_tiles[row][col],
                            init=2,
                            sym_name=f"C_L1L2_{col}_{row}_prod_lock",
                            lock_id=4,
                        ),
                        "get_lock": lock(
                            core_tiles[row][col],
                            init=0,
                            sym_name=f"C_L1L2_{col}_{row}_cons_lock",
                            lock_id=5,
                        ),
                    },
                    "cons": {
                        "endpoint": (
                            mem_tiles[col],
                            WireBundle.DMA,
                            row
                            + 2,  # S2MM channels 0, 1 on memtile are used for A, B coming in from shim
                        ),
                    },
                }
                src_tile, src_bundle, src_channel = C_l1l2_fifos[row][col]["prod"][
                    "endpoint"
                ]
                dst_tile, dst_bundle, dst_channel = C_l1l2_fifos[row][col]["cons"][
                    "endpoint"
                ]
                flow(
                    src_tile, src_bundle, src_channel, dst_tile, dst_bundle, dst_channel
                )

        # Output C, L2 -> L3
        for col in range(n_aie_cols):
            C_l2l3_fifos[col] = {
                "prod": {
                    "endpoint": (mem_tiles[col], WireBundle.DMA, 2),
                    "ping_buf": buffer(
                        mem_tiles[col],
                        datatype=C_l2_memref_ty,
                        name=f"C_L2L3_{col}_buff_0",
                    ),
                    "pong_buf": buffer(
                        mem_tiles[col],
                        datatype=C_l2_memref_ty,
                        name=f"C_L2L3_{col}_buff_1",
                    ),
                    "put_lock": lock(
                        mem_tiles[col],
                        init=4 * 2,
                        sym_name=f"C_L2L3_{col}_prod_lock",
                        lock_id=4,
                    ),
                    "get_lock": lock(
                        mem_tiles[col],
                        init=0,
                        sym_name=f"C_L2L3_{col}_cons_lock",
                        lock_id=5,
                    ),
                },
                "cons": {
                    "endpoint": (shim_tiles[col], WireBundle.DMA, 0),
                    "shim_memref": memref_dialect.global_(
                        sym_name=f"C_L2L3_{col}",
                        sym_visibility="public",
                        type_=C_l3_memref_ty,
                    ),
                    "shim_dma_alloc": ShimDMAAllocationOp(
                        f"C_L2L3_{col}", DMAChannelDir.S2MM, 0, col=col
                    ),
                },
            }
            src_tile, src_bundle, src_channel = C_l2l3_fifos[col]["prod"]["endpoint"]
            dst_tile, dst_bundle, dst_channel = C_l2l3_fifos[col]["cons"]["endpoint"]
            flow(src_tile, src_bundle, src_channel, dst_tile, dst_bundle, dst_channel)

        # Set up the data movement

        # Mem tiles
        for col in range(n_aie_cols):

            @memtile_dma(mem_tiles[col])
            def memtile_body(block):

                # A input
                A_l3l2_fifo = A_l3l2_fifos[col]["cons"]
                A_l2l1_fifo = A_l2l1_fifos[col]["prod"]
                _, _, a_in_channel = A_l3l2_fifo["endpoint"]
                _ = block["a_in_ping"], block["a_in_pong"]
                dma_start(
                    DMAChannelDir.S2MM,
                    a_in_channel,
                    dest=block["a_in_ping"],
                    chain=block["a_out"],
                )
                for pp in ["ping", "pong"]:
                    with block[f"a_in_{pp}"]:
                        use_lock(
                            A_l2l1_fifo["put_lock"],
                            LockAction.AcquireGreaterEqual,
                            value=1,
                        )
                        dma_bd(
                            A_l2l1_fifo[f"{pp}_buf"],
                            offset=0,
                            len=get_memref_len_elems(A_l2_memref_ty),
                        )
                        use_lock(A_l2l1_fifo["get_lock"], LockAction.Release, value=1)
                        next_bd(block[f"a_in_{'pong' if pp == 'ping' else 'ping'}"])

                # A output
                with block["a_out"]:
                    A_l2l1_fifo = A_l2l1_fifos[col]["prod"]
                    _, _, a_out_channel = A_l2l1_fifo["endpoint"]
                    _ = block["a_out_ping"], block["a_out_pong"]
                    dma_start(
                        DMAChannelDir.MM2S,
                        a_out_channel,
                        dest=block["a_out_ping"],
                        chain=block["b_in"],
                    )
                    for pp in ["ping", "pong"]:
                        with block[f"a_out_{pp}"]:
                            use_lock(
                                A_l2l1_fifo["get_lock"],
                                LockAction.AcquireGreaterEqual,
                                value=1,
                            )
                            assert get_memref_len_elems(
                                A_l1_memref_ty
                            ) == get_memref_len_elems(A_l2_memref_ty)
                            dma_bd(
                                A_l2l1_fifo[f"{pp}_buf"],
                                offset=0,
                                len=get_memref_len_elems(A_l1_memref_ty),
                                dimensions=[
                                    (m // r, r * k),
                                    (k // s, s),
                                    (r, k),
                                    (s, 1),
                                ],
                            )
                            use_lock(
                                A_l2l1_fifo["put_lock"], LockAction.Release, value=1
                            )
                            next_bd(
                                block[f"a_out_{'pong' if pp == 'ping' else 'ping'}"]
                            )

                # B input
                with block["b_in"]:
                    B_l3l2_fifo = B_l3l2_fifos[col]["cons"]
                    B_l2l1_fifo = B_l2l1_fifos[col]["prod"]
                    _, _, b_in_channel = B_l3l2_fifo["endpoint"]
                    _ = block["b_in_ping"], block["b_in_pong"]
                    dma_start(
                        DMAChannelDir.S2MM,
                        b_in_channel,
                        dest=block["b_in_ping"],
                        chain=block["b_out"],
                    )
                    for pp in ["ping", "pong"]:
                        with block[f"b_in_{pp}"]:
                            use_lock(
                                B_l2l1_fifo["put_lock"],
                                LockAction.AcquireGreaterEqual,
                                value=1,
                            )
                            dma_bd(
                                B_l2l1_fifo[f"{pp}_buf"],
                                offset=0,
                                len=get_memref_len_elems(B_l2_memref_ty),
                            )
                            use_lock(
                                B_l2l1_fifo["get_lock"], LockAction.Release, value=1
                            )
                            next_bd(block[f"b_in_{'pong' if pp == 'ping' else 'ping'}"])

                # B output
                with block["b_out"]:
                    B_l2l1_fifo = B_l2l1_fifos[col]["prod"]
                    _, _, b_out_channel = B_l2l1_fifo["endpoint"]
                    _ = block["b_out_ping"], block["b_out_pong"]
                    dma_start(
                        DMAChannelDir.MM2S,
                        b_out_channel,
                        dest=block["b_out_ping"],
                        chain=block["c_in_0"],
                    )
                    for pp in ["ping", "pong"]:
                        with block[f"b_out_{pp}"]:
                            use_lock(
                                B_l2l1_fifo["get_lock"],
                                LockAction.AcquireGreaterEqual,
                                value=1,
                            )
                            assert get_memref_len_elems(
                                B_l2_memref_ty
                            ) == get_memref_len_elems(B_l1_memref_ty)
                            dma_bd(
                                B_l2l1_fifo[f"{pp}_buf"],
                                offset=0,
                                len=get_memref_len_elems(B_l1_memref_ty),
                                dimensions=[
                                    (n // t, t * k),
                                    (k // s, s),
                                    (t, k),
                                    (s, 1),
                                ],
                            )
                            use_lock(
                                B_l2l1_fifo["put_lock"], LockAction.Release, value=1
                            )
                            next_bd(
                                block[f"b_out_{'pong' if pp == 'ping' else 'ping'}"]
                            )

                # C input
                for row in range(n_aie_rows):
                    C_l2l3_fifo = C_l2l3_fifos[col]["prod"]
                    with block[f"c_in_{row}"]:
                        C_l1l2_fifo = C_l1l2_fifos[row][col]["cons"]
                        _, _, c_in_channel = C_l1l2_fifo["endpoint"]
                        _ = block[f"c_in_{row}_ping"], block[f"c_in_{row}_pong"]
                        dma_start(
                            DMAChannelDir.S2MM,
                            c_in_channel,
                            dest=block[f"c_in_{row}_ping"],
                            chain=block[
                                f"c_in_{row+1}" if row + 1 < n_aie_rows else "c_out"
                            ],
                        )
                        for pp in ["ping", "pong"]:
                            with block[f"c_in_{row}_{pp}"]:
                                use_lock(
                                    C_l2l3_fifo["put_lock"],
                                    LockAction.AcquireGreaterEqual,
                                    value=1,
                                )
                                dma_bd(
                                    C_l2l3_fifo[f"{pp}_buf"],
                                    offset=row * get_memref_len_elems(C_l1_memref_ty),
                                    len=get_memref_len_elems(C_l1_memref_ty),
                                )
                                use_lock(
                                    C_l2l3_fifo["get_lock"], LockAction.Release, value=1
                                )
                                next_bd(
                                    block[
                                        f"c_in_{row}_{'pong' if pp == 'ping' else 'ping'}"
                                    ]
                                )

                # C output
                with block["c_out"]:
                    _, _, c_out_channel = C_l2l3_fifo["endpoint"]
                    _ = block["c_out_ping"], block["c_out_pong"]
                    dma_start(
                        DMAChannelDir.MM2S,
                        c_out_channel,
                        dest=block["c_out_ping"],
                        chain=block["end"],
                    )
                    for pp in ["ping", "pong"]:
                        with block[f"c_out_{pp}"]:
                            use_lock(
                                C_l2l3_fifo["get_lock"],
                                LockAction.AcquireGreaterEqual,
                                value=4,
                            )
                            assert get_memref_len_elems(
                                C_l2_memref_ty
                            ) == 4 * get_memref_len_elems(C_l1_memref_ty)
                            dma_bd(
                                C_l2l3_fifo[f"{pp}_buf"],
                                offset=0,
                                len=get_memref_len_elems(C_l2_memref_ty),
                                dimensions=[
                                    (m // r, r * n),
                                    (r, t),
                                    (n // t, r * t),
                                    (t, 1),
                                ],
                            )
                            use_lock(
                                C_l2l3_fifo["put_lock"], LockAction.Release, value=4
                            )
                            next_bd(
                                block[f"c_out_{'pong' if pp == 'ping' else 'ping'}"]
                            )

                with block["end"]:
                    EndOp()

        # core DMAs
        for row in range(n_aie_rows):
            for col in range(n_aie_cols):

                @mem(core_tiles[row][col])
                def core_mem_body(block):

                    # A input
                    A_l2l1_fifo = A_l2l1_fifos[row]["cons"][col]
                    _, _, a_in_channel = A_l2l1_fifo["endpoint"]
                    _ = block["a_in_ping"], block["a_in_pong"]
                    dma_start(
                        DMAChannelDir.S2MM,
                        a_in_channel,
                        dest=block["a_in_ping"],
                        chain=block["b_in"],
                    )
                    for pp in ["ping", "pong"]:
                        with block[f"a_in_{pp}"]:
                            use_lock(
                                A_l2l1_fifo["put_lock"],
                                LockAction.AcquireGreaterEqual,
                                value=1,
                            )
                            dma_bd(
                                A_l2l1_fifo[f"{pp}_buf"],
                                offset=0,
                                len=get_memref_len_elems(A_l1_memref_ty),
                            )
                            use_lock(
                                A_l2l1_fifo["get_lock"], LockAction.Release, value=1
                            )
                            next_bd(block[f"a_in_{'pong' if pp == 'ping' else 'ping'}"])

                    # B input
                    with block["b_in"]:
                        B_l2l1_fifo = B_l2l1_fifos[col]["cons"][row]
                        _, _, b_in_channel = B_l2l1_fifo["endpoint"]
                        _ = block["b_in_ping"], block["b_in_pong"]
                        dma_start(
                            DMAChannelDir.S2MM,
                            b_in_channel,
                            dest=block["b_in_ping"],
                            chain=block["c_out"],
                        )
                        for pp in ["ping", "pong"]:
                            with block[f"b_in_{pp}"]:
                                use_lock(
                                    B_l2l1_fifo["put_lock"],
                                    LockAction.AcquireGreaterEqual,
                                    value=1,
                                )
                                dma_bd(
                                    B_l2l1_fifo[f"{pp}_buf"],
                                    offset=0,
                                    len=get_memref_len_elems(B_l1_memref_ty),
                                )
                                use_lock(
                                    B_l2l1_fifo["get_lock"], LockAction.Release, value=1
                                )
                                next_bd(
                                    block[f"b_in_{'pong' if pp == 'ping' else 'ping'}"]
                                )

                    # C output
                    with block["c_out"]:
                        C_l1l2_fifo = C_l1l2_fifos[row][col]["prod"]
                        _, _, c_out_channel = C_l1l2_fifo["endpoint"]
                        _ = block["c_out_ping"], block["c_out_pong"]
                        dma_start(
                            DMAChannelDir.MM2S,
                            c_out_channel,
                            dest=block["c_out_ping"],
                            chain=block["end"],
                        )
                        for pp in ["ping", "pong"]:
                            with block[f"c_out_{pp}"]:
                                use_lock(
                                    C_l1l2_fifo["get_lock"],
                                    LockAction.AcquireGreaterEqual,
                                    value=1,
                                )
                                dma_bd(
                                    C_l1l2_fifo[f"{pp}_buf"],
                                    offset=0,
                                    len=get_memref_len_elems(C_l1_memref_ty),
                                )
                                use_lock(
                                    C_l1l2_fifo["put_lock"], LockAction.Release, value=1
                                )
                                next_bd(
                                    block[f"c_out_{'pong' if pp == 'ping' else 'ping'}"]
                                )

                    with block["end"]:
                        EndOp()

        # Set up compute tiles
        for row in range(n_aie_rows):
            for col in range(n_aie_cols):

                @core(core_tiles[row][col], f"mm_{m}x{k}x{n}.o")
                def core_body():
                    C_fifo = C_l1l2_fifos[row][col]["prod"]
                    A_fifo = A_l2l1_fifos[row]["cons"][col]
                    B_fifo = B_l2l1_fifos[col]["cons"][row]

                    c_0 = index_dialect.constant(0)
                    c_1 = index_dialect.constant(1)
                    c_2 = index_dialect.constant(2)
                    c_maxint = index_dialect.constant(0xFFFFFFFF)

                    run_loop = ForOp(
                        lower_bound=c_0, upper_bound=c_maxint, step=c_1, iter_args=[c_0]
                    )
                    with InsertionPoint(run_loop.body):
                        c_pp_outer = run_loop.inner_iter_args[0]

                        # Wait for "ready" signal through RTP and read RTP.
                        call(await_rtp, [rtp_bufs[row][col]])
                        rtp_K_div_k_div_2_i32 = call(
                            get_volatile_rtp, [rtp_bufs[row][col], 1]
                        )
                        rtp_K_div_k_div_2 = index_dialect.castu(
                            T.index(), rtp_K_div_k_div_2_i32
                        )
                        rtp_n_tiles_per_core_i32 = call(
                            get_volatile_rtp, [rtp_bufs[row][col], 2]
                        )
                        rtp_n_tiles_per_core = index_dialect.castu(
                            T.index(), rtp_n_tiles_per_core_i32
                        )

                        tile_loop = for_(rtp_n_tiles_per_core, iter_args=[T.index()])
                        tile_loop = ForOp(
                            lower_bound=c_0,
                            upper_bound=rtp_n_tiles_per_core,
                            step=c_1,
                            iter_args=[c_pp_outer],
                        )
                        with InsertionPoint(tile_loop.body):
                            c_pp_inner = tile_loop.inner_iter_args[
                                0
                            ]  # this variable flips between 0 and 1 each iteration
                            c_pp_cond = index_dialect.cmp("eq", c_pp_inner, c_0)
                            ifop = IfOp(c_pp_cond, [C_l1_memref_ty], hasElse=True)
                            # ifop.thenRegion.blocks.append()
                            with InsertionPoint(ifop.thenRegion.blocks[0]):
                                yield_([C_fifo["ping_buf"]])
                            # ifop.elseRegion.blocks.append()
                            with InsertionPoint(ifop.elseRegion.blocks[0]):
                                yield_([C_fifo["pong_buf"]])

                            use_lock(
                                C_fifo["put_lock"],
                                LockAction.AcquireGreaterEqual,
                                value=1,
                            )
                            elem_out = ifop.results_[0]
                            call(zero, [elem_out])
                            for j in for_(rtp_K_div_k_div_2):
                                for ab_pp in ["ping", "pong"]:
                                    use_lock(
                                        A_fifo["get_lock"],
                                        LockAction.AcquireGreaterEqual,
                                        value=1,
                                    )
                                    use_lock(
                                        B_fifo["get_lock"],
                                        LockAction.AcquireGreaterEqual,
                                        value=1,
                                    )
                                    elem_in_a = A_fifo[f"{ab_pp}_buf"]
                                    elem_in_b = B_fifo[f"{ab_pp}_buf"]
                                    call(matmul, [elem_in_a, elem_in_b, elem_out])
                                    use_lock(
                                        A_fifo["put_lock"], LockAction.Release, value=1
                                    )
                                    use_lock(
                                        B_fifo["put_lock"], LockAction.Release, value=1
                                    )
                                yield_([])
                            use_lock(C_fifo["get_lock"], LockAction.Release, value=1)

                            c_pp_inner_plus = index_dialect.add(c_pp_inner, c_1)
                            c_pp_inner_next = index_dialect.rems(c_pp_inner_plus, c_2)
                            yield_([c_pp_inner_next])

                        yield_([tile_loop.results_[0]])

        # To/from AIE-array data movement
        @runtime_sequence(A_l3_memref_ty, B_l3_memref_ty, C_l3_memref_ty)
        def sequence(A, B, C):
            # Write number of inner loop iterations for cores to use as run-time parameter.
            # This allows for processing different problem sizes by only swapping the insts.txt.
            assert (K // k) % 2 == 0
            rtp_K_div_k_div_2 = K // k // 2
            for row in range(n_aie_rows):
                for col in range(n_aie_cols):
                    sym_ref = FlatSymbolRefAttr.get(rtp_bufs[row][col].get_name()[1:])
                    npu_rtp_write(sym_ref, 1, rtp_K_div_k_div_2)
                    npu_rtp_write(sym_ref, 2, n_tiles_per_core)
                    npu_rtp_write(sym_ref, 0, 1)  # indicate "ready"

            # We are limited in the number of BDs. After synchronizing, we can reuse BDs.
            # We only transfer 6 rows of tiles at once before starting a new transfer block.
            tb_max_n_rows = (
                4  # tb = transfer block; block of transfers before sync call
            )
            for tb in range(ceildiv(M // m // n_aie_rows, tb_max_n_rows)):
                for pingpong in [0, 1]:
                    M // m // n_aie_rows // tb_max_n_rows
                    row_base = tb * tb_max_n_rows + pingpong * tb_max_n_rows // 2
                    bd_id_base = 8 * pingpong
                    tb_n_rows = min(
                        [tb_max_n_rows // 2, M // m // n_aie_rows - row_base]
                    )
                    if tb_n_rows <= 0:
                        # for small input sizes, we may not even need a "pong" iteration
                        break
                    for col in range(n_aie_cols):

                        # C Output Transfer:
                        # The smallest transfer unit is a (m*n_aie_rows)-x-(n)-sized sub-tile of the matrix.
                        # Transfer one such tile for every (n_aie_cols)-th column, evenly spaced,
                        # then repeat that (tb_n_rows) times for the next contiguous blocks of rows.
                        # Each shim will start at a different column offset, transferring interleaved
                        # columns. For example, shim 0 may transfer the blocks marked 0 below, and shim 1
                        # may transfer the blocks marked 1.
                        #
                        #             N
                        #      ----------------
                        #     |0011    0011    |
                        #     |0011    0011    |
                        #     |0011    0011    |
                        # M   |0011    0011    |
                        #     |                |
                        #     |                |
                        #     |                |
                        #     |                |
                        #      ----------------
                        C_row_offset = row_base * m * n_aie_rows * N
                        C_col_offset = col * n
                        C_offset = C_col_offset + C_row_offset
                        npu_dma_memcpy_nd(
                            metadata=C_l2l3_fifos[col]["cons"][
                                "shim_dma_alloc"
                            ].sym_name.value,
                            bd_id=bd_id_base,
                            mem=C,
                            offsets=[0, 0, 0, C_offset],
                            sizes=[tb_n_rows, N // n // n_aie_cols, m * n_aie_rows, n],
                            strides=[m * n_aie_rows * N, n * n_aie_cols, N, 1],
                        )

                        for tile_row in range(tb_n_rows):

                            # A input transfer:
                            #
                            # The smallest transfer unit is a (m*n_A_tiles_per_shim)-sized sub-tile of the input matrix.
                            # Transfer one such tile for every column, contiguously.
                            # Repeat this transfer with identical tiles a total of (N//n//n_aie_cols) times.
                            # Each shim transfers the tiles for separate rows. For example, shim 0 may transfer the
                            # tiles marked 0 below, and shim 1 may transfer the tiles marked 1.
                            #             K
                            #      ----------------
                            #     |0000000000000000|    (repeated N//n//n_aie_cols times)
                            #     |0000000000000000|
                            #     |1111111111111111|
                            # M   |1111111111111111|
                            #     |                |
                            #     |                |
                            #     |                |
                            #     |                |
                            #      ----------------
                            A_block_offset = (
                                (row_base + tile_row) * n_aie_rows * m * K
                            )  # base address for this transfer block for all BDs
                            A_row_offset = (
                                col * n_A_tiles_per_shim * m * K
                            )  # base address for the shim in this column
                            A_offset = A_block_offset + A_row_offset
                            npu_dma_memcpy_nd(
                                metadata=A_l3l2_fifos[col]["prod"][
                                    "shim_dma_alloc"
                                ].sym_name.value,
                                bd_id=bd_id_base + 2 * tile_row + 1,
                                mem=A,
                                offsets=[0, 0, 0, A_offset],
                                sizes=[
                                    N // n // n_aie_cols,
                                    K // k,
                                    m * n_A_tiles_per_shim,
                                    k,
                                ],
                                strides=[0, k, K, 1],
                            )

                            # B input transfer:
                            # Transfer the first a (n)-wide block of columns of B,
                            # Then transfer the (n_aie_columns)-th such block, and so on.
                            # Each shim will start at a different column offset.
                            # For example, shim 0 may transfer the tiles marked 0 below,
                            # and shim 1 may transfer the tiles marked 1.
                            #
                            #             N
                            #      ----------------
                            #     |0011    0011    |
                            #     |0011    0011    |
                            #     |0011    0011    |
                            # K   |0011    0011    |
                            #     |0011    0011    |
                            #     |0011    0011    |
                            #     |0011    0011    |
                            #     |0011    0011    |
                            #      ----------------
                            B_col_offset = col * n * K
                            npu_dma_memcpy_nd(
                                metadata=B_l3l2_fifos[col]["prod"][
                                    "shim_dma_alloc"
                                ].sym_name.value,
                                bd_id=bd_id_base + 2 * tile_row + 2,
                                mem=B,
                                offsets=[0, 0, 0, B_col_offset],
                                sizes=[N // n // n_aie_cols, K // k, n, k],
                                strides=[n * n_aie_cols * K, k, K, 1],
                            )
                    if tb > 0 or (tb == 0 and pingpong > 0):
                        for col in range(n_aie_cols):
                            npu_sync(
                                column=col, row=0, direction=0, channel=0
                            )  # C done
            for col in range(n_aie_cols):
                npu_sync(column=col, row=0, direction=0, channel=0)


if __name__ == "__main__":
    main()
else:
    print("Not meant to be imported")
    sys.exit(1)
