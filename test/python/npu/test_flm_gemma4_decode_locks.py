# test_flm_gemma4_decode_locks.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# RUN: %run_on_npu2_xrt% %pytest %s
# REQUIRES: xrt_python_bindings
"""The flm_gemma4 decode kernels' locks on a neighbour's buffer, on the NPU.

``decode_gate_layer_embedding`` reads ``x`` from a neighbouring tile and
``decode_proj_main`` (with ``send_x_output`` 0) writes ``y`` into one. Both
take that tile's locks at an id offset they derive from the buffer's address
(``_lock_offset`` in ``flm_gemma4/utils.h``), so each runs here with the
neighbour south, north and west of the core and on the core's own tile. Every
lock is an explicit ``Lock`` at the factory's default id, and the runtime
sequence sets ``proj_main``'s RTP lock, as the decode layer's sequence does.

``decode_attn_kv`` waits in ``attn_kv_begin`` on a lock of its own tile that
the runtime sequence sets, rather than one its qk neighbour releases; it runs
two rounds here, v arriving through its ping-pong pair, and writes ``o`` for
the down projection's three reads.
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
    ObjectFifo,
    Out,
    Program,
    Release,
    Runtime,
    TileDma,
    Worker,
    kernels,
)
from aie.iron.controlflow import range_
from aie.iron.device import Tile
from ml_dtypes import bfloat16

pytestmark = pytest.mark.supported_devices("npu2")

BF16 = np.dtype[bfloat16]
GEOMETRY = kernels.FLM_GEMMA4_E2B_DECODE
D, PLI = GEOMETRY.model_dim, GEOMETRY.pli_d

# (core, neighbour) placements: the neighbour holds the shared buffer.
RELATIONS = {
    "south": ((1, 3), (1, 2)),
    "north": ((1, 2), (1, 3)),
    "west": ((1, 2), (0, 2)),
    "own": ((1, 2), (1, 2)),
}


def _tiles(relation):
    (cc, cr), (nc, nr) = RELATIONS[relation]
    core = Tile(cc, cr, tile_type=AIETileType.CoreTile)
    near = (
        core if (nc, nr) == (cc, cr) else Tile(nc, nr, tile_type=AIETileType.CoreTile)
    )
    return Tile(cc, 0, tile_type=AIETileType.ShimNOCTile), core, near


def _pingpong(flow, tile, ping, pong, prod, cons, direction):
    return DmaChannel(
        direction=direction,
        channel=flow.endpoint(tile),
        bds=[
            Bd(buffer=b, acquires=[Acquire(prod)], releases=[Release(cons)])
            for b in (ping, pong)
        ],
    )


# --------------------------------------------------------------------------
# decode_gate_layer_embedding
# --------------------------------------------------------------------------

GATE_BLOCKS = (PLI // 32) * (D // 256)


@iron.jit
def gate_design(X: In, W: In, Y: Out, *, relation: CompileTime[str]):
    shim, core, near = _tiles(relation)
    kernel = kernels.flm_gemma4_decode_gate_layer_embedding(geometry=GEOMETRY)
    w_ty = np.ndarray[(32 * 256,), BF16]
    x = Buffer(type=np.ndarray[(D,), BF16], tile=near, name="x")
    w_ping = Buffer(type=w_ty, tile=core, name="w_ping")
    w_pong = Buffer(type=w_ty, tile=core, name="w_pong")
    y = Buffer(type=np.ndarray[(D + PLI,), BF16], tile=core, name="y")
    final_x_prod = Lock(near, lock_id=7, init=1, name="final_x_prod")
    final_x_cons = Lock(near, lock_id=8, init=0, name="final_x_cons")
    w_prod = Lock(core, lock_id=2, init=2, name="w_prod")
    w_cons = Lock(core, lock_id=3, init=0, name="w_cons")
    y_prod = Lock(core, lock_id=4, init=1, name="y_prod")
    y_cons = Lock(core, lock_id=5, init=0, name="y_cons")
    x_flow = Flow(src=shim, dst=near)
    w_flow = Flow(src=shim, dst=core)
    y_flow = Flow(src=core, dst=shim)
    x_channel = DmaChannel(
        direction=DMAChannelDir.S2MM,
        channel=x_flow.endpoint(near),
        bds=[
            Bd(
                buffer=x,
                acquires=[Acquire(final_x_prod)],
                releases=[Release(final_x_cons)],
            )
        ],
    )
    core_channels = [
        _pingpong(w_flow, core, w_ping, w_pong, w_prod, w_cons, DMAChannelDir.S2MM),
        DmaChannel(
            direction=DMAChannelDir.MM2S,
            channel=y_flow.endpoint(core),
            bds=[Bd(buffer=y, acquires=[Acquire(y_cons)], releases=[Release(y_prod)])],
        ),
    ]
    dmas = (
        [TileDma(tile=core, channels=[x_channel, *core_channels])]
        if near is core
        else [
            TileDma(tile=near, channels=[x_channel]),
            TileDma(tile=core, channels=core_channels),
        ]
    )

    def body(k, x, w_ping, w_pong, y):
        k(x, w_ping, w_pong, y)

    worker = Worker(body, [kernel, x, w_ping, w_pong, y], tile=core)

    def sequence(X, W, Y):
        x_flow.fill(X)
        w_flow.fill(W)
        y_flow.drain(Y, wait=True)

    rt = Runtime(
        sequence,
        [
            np.ndarray[(D,), BF16],
            np.ndarray[(GATE_BLOCKS * 32 * 256,), BF16],
            np.ndarray[(D + PLI,), BF16],
        ],
    )
    for flow in (x_flow, w_flow, y_flow):
        rt.add_flow(flow)
    for dma in dmas:
        rt.add_tile_dma(dma)
    return Program(iron.get_current_device(), rt, workers=[worker]).resolve_program()


def _gate_inputs(rng):
    # x in {-1, 0, 1} and weights in {-1, 0, 1} / 32: every projection is a
    # multiple of 1/32 below 8 in magnitude, exact in bf16 and in any float32
    # summation order, and it spans the GELU table's clamps.
    x = rng.choice([-1.0, 0.0, 1.0], D, p=[0.15, 0.7, 0.15])
    w = rng.choice([-1.0, 0.0, 1.0], (GATE_BLOCKS, 256, 32), p=[0.2, 0.6, 0.2]) / 32
    return x.astype(bfloat16), w.astype(bfloat16)


def _gate_projection(x, w):
    # Block (i, j) is stored column by column: w[i, j, k, m] multiplies
    # x[j * 256 + k] into out-feature i * 32 + m.
    blocks = np.asarray(w, np.float64).reshape(PLI // 32, D // 256, 256, 32)
    xs = np.asarray(x, np.float64).reshape(D // 256, 256)
    return np.einsum("ijkm,jk->im", blocks, xs).reshape(PLI)


@pytest.mark.parametrize("relation", list(RELATIONS))
def test_gate_layer_embedding_reads_a_neighbours_x(relation):
    rng = np.random.default_rng(31)
    x, w = _gate_inputs(rng)
    proj = _gate_projection(x, w)
    assert np.abs(proj).max() < 8 and np.all(proj * 32 == np.round(proj * 32))
    y = iron.zeros(D + PLI, dtype=bfloat16, device="npu")
    gate_design(
        iron.tensor(x, dtype=bfloat16, device="npu"),
        iron.tensor(w.reshape(-1), dtype=bfloat16, device="npu"),
        y,
        relation=relation,
    )
    y.to("cpu")
    got = y.numpy()
    np.testing.assert_array_equal(got[:D].view(np.uint16), x.view(np.uint16))
    gelu = kernels.flm_gemma4_pli_gelu_core(geometry=GEOMETRY)
    inputs = [proj.astype(bfloat16).reshape(1, PLI)]
    verdict = gelu.judge(
        got[D:].reshape(1, PLI), gelu.expected(inputs), calls=1, inputs=inputs
    )
    assert verdict, f"{relation}: {verdict.detail}"


# --------------------------------------------------------------------------
# decode_proj_main
# --------------------------------------------------------------------------

M_BLOCK, K_BLOCK = 32, 256
BLOCK_BYTES = M_BLOCK * K_BLOCK * 5 // 8
Y_WORDS = 2 * M_BLOCK + 16
MVM_CORES = 16


def _proj_main_shapes():
    # IS_SWA = 1 and SKIP_KV = 0 on the double-wide-MLP E2B build: qkv, o,
    # up/gate and down, each this core's 1/16 share of the rows.
    swa_q = GEOMETRY.num_attn_heads * GEOMETRY.swa_dh
    swa_kv = GEOMETRY.num_kv_heads * GEOMETRY.swa_dh
    inter = GEOMETRY.intermediate_size
    assert GEOMETRY.double_wide_mlp
    return [
        ((swa_q + 2 * swa_kv) // MVM_CORES, D),
        (D // MVM_CORES, swa_q),
        (2 * inter // MVM_CORES, D),
        (D // MVM_CORES, inter),
    ]


SHAPES = _proj_main_shapes()
PROJ_BLOCKS = sum((m // M_BLOCK) * (k // K_BLOCK) for m, k in SHAPES)
Y_SLOTS = sum(m // M_BLOCK for m, _ in SHAPES)
X_WORDS = sum((m // M_BLOCK) * k for m, k in SHAPES)


@iron.jit
def proj_main_design(X: In, W: In, Y: Out, *, relation: CompileTime[str]):
    shim, core, near = _tiles(relation)
    kernel = kernels.flm_gemma4_decode_proj_main(geometry=GEOMETRY)
    y_ty = np.ndarray[(Y_WORDS,), BF16]
    w_ty = np.ndarray[(BLOCK_BYTES // 2,), BF16]
    x_ty = np.ndarray[(K_BLOCK,), BF16]
    rtp_ty = np.ndarray[(16,), np.dtype[np.int32]]
    y_ping = Buffer(type=y_ty, tile=near, name="y_ping")
    y_pong = Buffer(type=y_ty, tile=near, name="y_pong")
    w_ping = Buffer(type=w_ty, tile=core, name="w_ping")
    w_pong = Buffer(type=w_ty, tile=core, name="w_pong")
    x_ping = Buffer(type=x_ty, tile=core, name="x_ping")
    x_pong = Buffer(type=x_ty, tile=core, name="x_pong")
    is_swa = Buffer(
        type=rtp_ty,
        tile=core,
        name="is_swa",
        initial_value=np.eye(1, 16, dtype=np.int32)[0],
    )
    skip_kv = Buffer(
        type=rtp_ty, tile=core, name="skip_kv", initial_value=np.zeros(16, np.int32)
    )
    x_prod = Lock(core, lock_id=0, init=2, name="x_prod")
    x_cons = Lock(core, lock_id=1, init=0, name="x_cons")
    w_prod = Lock(core, lock_id=2, init=2, name="w_prod")
    w_cons = Lock(core, lock_id=3, init=0, name="w_cons")
    rtp_available = Lock(core, lock_id=6, init=0, name="rtp_available")
    y_prod_ping = Lock(near, lock_id=4, init=1, name="y_prod_ping")
    y_prod_pong = Lock(near, lock_id=5, init=1, name="y_prod_pong")
    y_cons_ping = Lock(near, lock_id=7, init=0, name="y_cons_ping")
    y_cons_pong = Lock(near, lock_id=8, init=0, name="y_cons_pong")
    w_flow = Flow(src=shim, dst=core)
    x_flow = Flow(src=shim, dst=core)
    y_flow = Flow(src=near, dst=shim)
    y_channel = DmaChannel(
        direction=DMAChannelDir.MM2S,
        channel=y_flow.endpoint(near),
        bds=[
            Bd(
                buffer=y_ping,
                acquires=[Acquire(y_cons_ping)],
                releases=[Release(y_prod_ping)],
            ),
            Bd(
                buffer=y_pong,
                acquires=[Acquire(y_cons_pong)],
                releases=[Release(y_prod_pong)],
            ),
        ],
    )
    core_channels = [
        _pingpong(w_flow, core, w_ping, w_pong, w_prod, w_cons, DMAChannelDir.S2MM),
        _pingpong(x_flow, core, x_ping, x_pong, x_prod, x_cons, DMAChannelDir.S2MM),
    ]
    dmas = (
        [TileDma(tile=core, channels=[*core_channels, y_channel])]
        if near is core
        else [
            TileDma(tile=core, channels=core_channels),
            TileDma(tile=near, channels=[y_channel]),
        ]
    )

    # The kernel takes rtp_available by id; passing it declares it on the core.
    def body(k, y_ping, w_ping, x_ping, y_pong, w_pong, x_pong, is_swa, skip_kv, _):
        k(y_ping, w_ping, x_ping, y_pong, w_pong, x_pong, is_swa, skip_kv, 0)

    worker = Worker(
        body,
        [
            kernel,
            y_ping,
            w_ping,
            x_ping,
            y_pong,
            w_pong,
            x_pong,
            is_swa,
            skip_kv,
            rtp_available,
        ],
        tile=core,
    )

    def sequence(X, W, Y):
        rtp_available.set(1)
        w_flow.fill(W)
        x_flow.fill(X)
        y_flow.drain(Y, wait=True)

    rt = Runtime(
        sequence,
        [
            np.ndarray[(X_WORDS,), BF16],
            np.ndarray[(PROJ_BLOCKS * BLOCK_BYTES // 2,), BF16],
            np.ndarray[(Y_SLOTS * Y_WORDS,), BF16],
        ],
    )
    for flow in (w_flow, x_flow, y_flow):
        rt.add_flow(flow)
    for dma in dmas:
        rt.add_tile_dma(dma)
    return Program(iron.get_current_device(), rt, workers=[worker]).resolve_program()


def _q4nx_block(codes, scales, mins):
    """Pack one 32 by 256 q4nx block: scales and minima [group, row], then codes.

    The codes sit in strips of 16 rows, each strip column-major, two to a
    byte, low nibble first.
    """
    order = codes.reshape(2, 16, 8, 4, 8).transpose(0, 2, 3, 4, 1).reshape(-1)
    packed = (order[0::2] | (order[1::2] << 4)).astype(np.uint8)
    head = np.concatenate([scales.reshape(-1), mins.reshape(-1)]).astype(bfloat16)
    return np.concatenate([head.view(np.uint8), packed])


def _proj_main_inputs(rng):
    # Codes in [0, 3], x in {-1, 0, 1}, power-of-two scales and integer
    # minima: each group's code sum (at most 96) and x sum (at most 32) are
    # exact in bf16, and every float32 partial sum is exact.
    x_stream, w_stream, expected = [], [], []
    for m, k in SHAPES:
        x = rng.choice([-1.0, 0.0, 1.0], k).astype(bfloat16)
        y = np.zeros(m)
        for i in range(m // M_BLOCK):
            for j in range(k // K_BLOCK):
                codes = rng.integers(0, 4, (M_BLOCK, K_BLOCK))
                scales = rng.choice([0.5, 1.0, 2.0], (K_BLOCK // 32, M_BLOCK))
                mins = rng.integers(-2, 3, (K_BLOCK // 32, M_BLOCK)).astype(float)
                w_stream.append(_q4nx_block(codes, scales, mins))
                xs = x[j * K_BLOCK : (j + 1) * K_BLOCK].astype(np.float64)
                x_stream.append(x[j * K_BLOCK : (j + 1) * K_BLOCK])
                groups = np.arange(K_BLOCK) // 32
                dequant = scales[groups].T * codes + mins[groups].T
                y[i * M_BLOCK : (i + 1) * M_BLOCK] += dequant @ xs
        expected.append(y)
    return (
        np.concatenate(x_stream),
        np.concatenate(w_stream).view(bfloat16),
        np.concatenate(expected),
    )


@pytest.mark.parametrize("relation", list(RELATIONS))
def test_proj_main_writes_a_neighbours_y(relation):
    rng = np.random.default_rng(37)
    x, w, expected = _proj_main_inputs(rng)
    y = iron.zeros(Y_SLOTS * Y_WORDS, dtype=bfloat16, device="npu")
    proj_main_design(
        iron.tensor(x, dtype=bfloat16, device="npu"),
        iron.tensor(w, dtype=bfloat16, device="npu"),
        y,
        relation=relation,
    )
    y.to("cpu")
    # With send_x_output 0 the core fills the second 32-output slot.
    got = y.numpy().reshape(Y_SLOTS, Y_WORDS)[:, 16 + M_BLOCK :].astype(np.float64)
    got = got.reshape(-1)
    _, e = np.frexp(np.abs(expected))
    ulp = np.ldexp(1.0, e - 8)
    bad = np.abs(got - expected) > ulp
    assert not bad.any(), (
        f"{relation}: {bad.sum()} of {bad.size} outputs off by more than one "
        f"bf16 ulp; first at {np.flatnonzero(bad)[0]}: got "
        f"{got[bad][0]}, expected {expected[bad][0]}"
    )


# --------------------------------------------------------------------------
# decode_attn_kv
# --------------------------------------------------------------------------

DH = GEOMETRY.dh
KEYS = 16
S_WORDS = KEYS * 8 + 32
ROUNDS = 2
O_REPEATS = D // MVM_CORES // M_BLOCK


@iron.jit
def attn_kv_design(S: In, V: In, O: Out):
    shim = Tile(1, 0, tile_type=AIETileType.ShimNOCTile)
    core = Tile(1, 2, tile_type=AIETileType.CoreTile)
    begin = kernels.flm_gemma4_decode_attn_kv(geometry=GEOMETRY)
    v_ty = np.ndarray[(KEYS, DH), BF16]
    o_ty = np.ndarray[(8 * DH,), BF16]
    s_fifo = ObjectFifo(np.ndarray[(S_WORDS,), BF16], name="s")
    v_ping = Buffer(type=v_ty, tile=core, name="v_ping")
    v_pong = Buffer(type=v_ty, tile=core, name="v_pong")
    y = Buffer(type=np.ndarray[(8 * DH,), np.dtype[np.float32]], tile=core, name="y")
    l = Buffer(type=np.ndarray[(8,), np.dtype[np.float32]], tile=core, name="l")
    o = Buffer(type=o_ty, tile=core, name="o")
    o_prod = Lock(core, lock_id=0, init=O_REPEATS, name="o_prod")
    o_cons = Lock(core, lock_id=1, init=0, name="o_cons")
    v_prod = Lock(core, lock_id=2, init=2, name="v_prod")
    v_cons = Lock(core, lock_id=3, init=0, name="v_cons")
    l_cons = Lock(core, lock_id=8, init=0, name="l_cons")
    v_flow = Flow(src=shim, dst=core)
    o_flow = Flow(src=core, dst=shim)
    dma = TileDma(
        tile=core,
        channels=[
            _pingpong(v_flow, core, v_ping, v_pong, v_prod, v_cons, DMAChannelDir.S2MM),
            DmaChannel(
                direction=DMAChannelDir.MM2S,
                channel=o_flow.endpoint(core),
                bds=[
                    Bd(buffer=o, acquires=[Acquire(o_cons)], releases=[Release(o_prod)])
                ],
            ),
        ],
    )

    # The kernel takes l_cons by id; passing it declares it on the core.
    def body(begin, round_, finish, s_in, v_ping, v_pong, y, l, o, _):
        begin(y, l)
        for _ in range_(ROUNDS):
            s = s_in.acquire(1)
            round_(s, v_ping, v_pong, y, l)
            s_in.release(1)
        finish(y, o, l)

    worker = Worker(
        body,
        [
            begin,
            begin.attn_kv_round,
            begin.attn_kv_finish,
            s_fifo.cons(),
            v_ping,
            v_pong,
            y,
            l,
            o,
            l_cons,
        ],
        tile=core,
    )

    def sequence(S, V, O, s_out):
        l_cons.set(1)
        s_out.fill(S)
        v_flow.fill(V)
        o_flow.drain(O, wait=True)

    rt = Runtime(
        sequence,
        [
            np.ndarray[(ROUNDS * S_WORDS,), BF16],
            np.ndarray[(ROUNDS * KEYS * DH,), BF16],
            np.ndarray[(O_REPEATS * 8 * DH,), BF16],
            s_fifo.prod(),
        ],
    )
    rt.add_flow(v_flow)
    rt.add_flow(o_flow)
    rt.add_tile_dma(dma)
    return Program(iron.get_current_device(), rt, workers=[worker]).resolve_program()


def _attn_kv_rounds(rng):
    # As flm_gemma4_attn_kv_core's sample: scores in [0, 1) and corrections c
    # in [0.05, 1) as float32 in the scores' tail. v is positive, so y does
    # not cancel and the bf16 mmul's bound stays a few percent of o.
    s = np.zeros((ROUNDS, S_WORDS), bfloat16)
    s[:, :128] = rng.uniform(0.0, 1.0, (ROUNDS, 128)).astype(bfloat16)
    c = rng.uniform(0.05, 1.0, (ROUNDS, 8)).astype(np.float32)
    s[:, 128:144] = c.view(bfloat16)
    v = rng.uniform(0.25, 1.0, (ROUNDS, KEYS * DH)).astype(bfloat16)
    return s, v, c


def _attn_kv_expected(s, v, c):
    """Return o = y / l per query head, in y's packed layout, and its bound.

    Each round is flm_gemma4_attn_kv_core's reference and bound, a round's
    bound carried into the next scaled by its c. o adds y's narrowing to bf16,
    the getInvBf16 table (its input and its entry each round to bf16) and
    o's own bf16 store, 2**-8 relative each.
    """
    core = kernels.flm_gemma4_attn_kv_core(geometry=GEOMETRY)
    bound = core.contract.tolerance.bound
    ly = np.zeros((1, 8 * DH + 16), np.float32)
    err = np.zeros(8 * DH + 16)
    head = (np.arange(8 * DH) // 8) % 8
    for r in range(ROUNDS):
        sv = np.concatenate([s[r], v[r]])[None]
        scale = np.concatenate([c[r][head], c[r], np.zeros(8)])
        err = scale * err + bound(sv, ly)[0]
        ly = kernels.flm_gemma4_attn_kv_core_ref(sv, ly, dh=DH)
    y, l = ly[0, : 8 * DH].astype(np.float64), ly[0, 8 * DH : 8 * DH + 8].astype(
        np.float64
    )
    dy, dl = err[: 8 * DH], err[8 * DH : 8 * DH + 8]
    o = y / l[head]
    tol = np.abs(o) * 4 * 2.0**-8 + (dy + np.abs(o) * dl[head]) / (l - dl)[head]
    return o, tol


def test_attn_kv_waits_on_the_lock_its_sequence_sets():
    rng = np.random.default_rng(41)
    s, v, c = _attn_kv_rounds(rng)
    o = iron.zeros(O_REPEATS * 8 * DH, dtype=bfloat16, device="npu")
    attn_kv_design(
        iron.tensor(s.reshape(-1), dtype=bfloat16, device="npu"),
        iron.tensor(v.reshape(-1), dtype=bfloat16, device="npu"),
        o,
    )
    o.to("cpu")
    reads = o.numpy().reshape(O_REPEATS, 8 * DH)
    for r in range(1, O_REPEATS):
        np.testing.assert_array_equal(
            reads[r].view(np.uint16), reads[0].view(np.uint16)
        )
    expected, tol = _attn_kv_expected(s, v, c)
    err = np.abs(reads[0].astype(np.float64) - expected)
    assert np.all(err <= tol), (
        f"{int((err > tol).sum())} of {err.size} outputs past the bound; worst "
        f"{err.max():.3g} at {int(err.argmax())} (bound {tol[err.argmax()]:.3g})"
    )
