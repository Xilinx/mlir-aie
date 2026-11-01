# test_packet_flow_sharing.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %run_on_npu2% %pytest %s
# REQUIRES: xrt_python_bindings

"""On-device test of PacketFlow channels the compiler assigns or flows share.

One shim channel sends two packet ids to two shims; two shims send into one
shared shim channel, with and without ``keep_pkt_header``; and two flows share
one core S2MM channel, beside a third input or not, on a placed or an unplaced
core. A merged channel takes its packets whole, in either order.
"""

import aie.iron as iron
import numpy as np
import pytest
from aie.dialects._aie_enum_gen import AIETileType, DMAChannelDir
from aie.iron import (
    Acquire,
    Bd,
    Buffer,
    CompileTime,
    DmaChannel,
    Flow,
    In,
    Lock,
    Out,
    PacketFlow,
    Program,
    Release,
    Runtime,
    TileDma,
    Worker,
)
from aie.iron.device import Tile

H = 512


@iron.jit
def split(a: In, b: Out, c: Out):
    in_ty = np.ndarray[(2 * H,), np.dtype[np.int32]]
    out_ty = np.ndarray[(H,), np.dtype[np.int32]]
    s0, s1, s2 = (Tile(c, 0, tile_type=AIETileType.ShimNOCTile) for c in range(3))
    f1 = PacketFlow(3, s0, s1, src_channel=None, dst_channel=None)
    f2 = PacketFlow(4, s0, s2, src_channel=f1.endpoint(s0), dst_channel=None)

    def seq(A, B, C):
        f1.fill(A, tap=A[:H])
        f2.fill(A, tap=A[H:])
        f1.drain(B, wait=True)
        f2.drain(C, wait=True)

    rt = Runtime(seq, [in_ty, out_ty, out_ty])
    rt.add_flow(f1)
    rt.add_flow(f2)
    return Program(iron.get_current_device(), rt).resolve_program()


@iron.jit
def merge(a: In, c: In, d: Out, *, keep: CompileTime[bool]):
    in_ty = np.ndarray[(H,), np.dtype[np.int32]]
    out_ty = np.ndarray[(2 * H + (2 if keep else 0),), np.dtype[np.int32]]
    s0, s1, s2 = (Tile(c, 0, tile_type=AIETileType.ShimNOCTile) for c in range(3))
    g1 = PacketFlow(5, s0, s2, src_channel=None, dst_channel=None, keep_pkt_header=keep)
    g2 = PacketFlow(
        6, s1, s2, src_channel=None, dst_channel=g1.endpoint(s2), keep_pkt_header=keep
    )

    def seq(A, C, D):
        g1.fill(A)
        g2.fill(C)
        g1.drain(D, wait=True)

    rt = Runtime(seq, [in_ty, in_ty, out_ty])
    rt.add_flow(g1)
    rt.add_flow(g2)
    return Program(iron.get_current_device(), rt).resolve_program()


@iron.jit
def core_merge(
    a: In,
    c: In,
    e: In,
    d: Out,
    *,
    placed: CompileTime[bool],
    own_input: CompileTime[bool],
):
    in_ty = np.ndarray[(H,), np.dtype[np.int32]]
    pair_ty = np.ndarray[(2 * H,), np.dtype[np.int32]]
    s0, s1, s2, s3 = (Tile(c, 0, tile_type=AIETileType.ShimNOCTile) for c in range(4))
    core = Tile(0, 2) if placed else Tile(tile_type=AIETileType.CoreTile)
    worker = Worker(lambda: None, [], tile=core, while_true=False)
    g1 = PacketFlow(1, s0, core, src_channel=None, dst_channel=None)
    g2 = PacketFlow(2, s1, core, src_channel=None, dst_channel=g1.endpoint(core))
    own = Flow(s2, core)
    back = Flow(core, s3)
    pair = Buffer(tile=core, type=pair_ty, name="pair")
    single = Buffer(tile=core, type=in_ty, name="single")
    pair_free, pair_full = Lock(core, init=1), Lock(core, init=0)
    single_free, single_full = Lock(core, init=1), Lock(core, init=0)
    send = [Bd(pair, acquires=[Acquire(pair_full)], releases=[Release(pair_free)])]
    channels = [
        DmaChannel(
            DMAChannelDir.S2MM,
            g1.endpoint(core),
            [Bd(pair, acquires=[Acquire(pair_free)], releases=[Release(pair_full)])],
        )
    ]
    if own_input:
        channels.append(
            DmaChannel(
                DMAChannelDir.S2MM,
                own.endpoint(core),
                [
                    Bd(
                        single,
                        acquires=[Acquire(single_free)],
                        releases=[Release(single_full)],
                    )
                ],
            )
        )
        send.append(
            Bd(
                single,
                acquires=[Acquire(single_full)],
                releases=[Release(single_free)],
            )
        )
    channels.append(DmaChannel(DMAChannelDir.MM2S, back.endpoint(core), send))

    def seq(A, C, E, D):
        g1.fill(A)
        g2.fill(C)
        if own_input:
            own.fill(E)
        back.drain(D, wait=True)

    out_ty = np.ndarray[((3 if own_input else 2) * H,), np.dtype[np.int32]]
    rt = Runtime(seq, [in_ty, in_ty, in_ty, out_ty])
    for flow in (g1, g2, *([own] if own_input else []), back):
        rt.add_flow(flow)
    rt.add_tile_dma(TileDma(core, channels))
    return Program(iron.get_current_device(), rt, workers=[worker]).resolve_program()


def random_ints(seed: int, n: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(-(2**31), 2**31 - 1, n, dtype=np.int32)


def test_one_shim_channel_sends_two_packet_ids():
    host = random_ints(0, 2 * H)
    a = iron.tensor(host, dtype=np.int32, device="npu")
    b = iron.zeros(H, dtype=np.int32, device="npu")
    c = iron.zeros(H, dtype=np.int32, device="npu")
    split(a, b, c)
    b.to("cpu")
    c.to("cpu")
    np.testing.assert_array_equal(b.numpy(), host[:H])
    np.testing.assert_array_equal(c.numpy(), host[H:])


@pytest.mark.parametrize("keep", [False, True])
def test_two_shims_send_into_one_shared_shim_channel(keep):
    first, second = random_ints(1, H), random_ints(2, H)
    a = iron.tensor(first, dtype=np.int32, device="npu")
    c = iron.tensor(second, dtype=np.int32, device="npu")
    d = iron.zeros(2 * H + (2 if keep else 0), dtype=np.int32, device="npu")
    merge(a, c, d, keep=keep)
    d.to("cpu")
    got = d.numpy()
    if keep:
        # Each packet arrives behind its 32-bit header, whose low bits are its id.
        packets = {int(got[s]) & 0x1F: got[s + 1 : s + 1 + H] for s in (0, H + 1)}
        assert sorted(packets) == [5, 6]
        np.testing.assert_array_equal(packets[5], first)
        np.testing.assert_array_equal(packets[6], second)
    else:
        assert np.array_equal(got, np.concatenate([first, second])) or np.array_equal(
            got, np.concatenate([second, first])
        )


@pytest.mark.parametrize("placed", [True, False])
@pytest.mark.parametrize("own_input", [False, True])
def test_two_flows_share_one_core_channel(placed, own_input):
    first, second, third = random_ints(3, H), random_ints(4, H), random_ints(5, H)
    a, c, e = (
        iron.tensor(x, dtype=np.int32, device="npu") for x in (first, second, third)
    )
    d = iron.zeros((3 if own_input else 2) * H, dtype=np.int32, device="npu")
    core_merge(a, c, e, d, placed=placed, own_input=own_input)
    d.to("cpu")
    got = d.numpy()
    assert np.array_equal(
        got[: 2 * H], np.concatenate([first, second])
    ) or np.array_equal(got[: 2 * H], np.concatenate([second, first]))
    if own_input:
        np.testing.assert_array_equal(got[2 * H :], third)
