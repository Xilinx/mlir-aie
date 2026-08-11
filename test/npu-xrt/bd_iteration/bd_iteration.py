# bd_iteration/bd_iteration.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""BD iteration: one buffer descriptor advancing its base address per execution.

The BD iteration fields (``iteration_size`` / ``iteration_stride``) make a
single ``aie.dma_bd`` advance its own base address by ``iteration_stride``
elements after each execution and wrap after ``iteration_size`` executions.
That lets one descriptor express a regular per-execution address progression
that otherwise takes an N-deep BD chain (one descriptor per offset).

This program is a minimal end-to-end check of the mechanism on device. A shim
tile streams a 256-element buffer into a MemTile buffer in four 64-element
chunks. A single S2MM ``aie.dma_bd`` with ``iteration_size=4`` and
``iteration_stride=64`` receives them, its base advancing one 64-element slot
per chunk. The MemTile buffer is read back to the host unmodified.

The BD is self-chained (``next="self"``), so loops indefinitely. What bounds it
to four executions here is the handshake (four chunks), not the descriptor.

``iteration_current`` is the initial value of a per-BD register that the DMA
advances each execution and wraps at ``iteration_size``, independent of its
position in a BD chain. It defaults to 0; the ``--start`` knob sets it nonzero
so chunk ``k`` lands at slot ``(start + k) % iteration_size``.

The ``--slots`` knob falsifies this test: ``iteration_size=1`` makes every chunk
land at offset 0 (each execution overwriting the last) and the run fails.

No compute tile.
"""

import argparse

import aie.iron as iron
import numpy as np
from aie.dialects._aie_enum_gen import (  # pyright: ignore[reportMissingImports]
    AIETileType,
    DMAChannelDir,
    WireBundle,
)
from aie.dialects.aie import EndOp  # pyright: ignore[reportAttributeAccessIssue]
from aie.dialects.aiex import (
    bds,
    dma_await_task,
    dma_configure_task,
    dma_free_task,
    dma_start_task,
    shim_dma_bd,  # pyright: ignore[reportAttributeAccessIssue]
)
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
    Program,
    Release,
    Runtime,
    TileDma,
)
from aie.iron.device import Tile
from aie.utils.hostruntime.argparse import add_compile_args, device_from_args
from aie.utils.hostruntime.cli import run_design_cli
from aie.utils.verify import assert_pass

CHUNK = 64  # elements (int32) moved by one BD execution
N_CHUNKS = 4  # number of BD executions -- fixed by the input stream length
N_SLOTS = N_CHUNKS  # default iteration_size: one distinct slot per chunk
TOTAL = CHUNK * N_CHUNKS  # backing-buffer size in elements


@iron.jit
def bd_iteration(
    a_in: In,
    c_out: Out,
    *,
    col: CompileTime[int] = 0,
    n_slots: CompileTime[int] = N_SLOTS,
    start: CompileTime[int] = 0,
):
    vector_ty = np.ndarray[(TOTAL,), np.dtype[np.int32]]

    shim_tile = Tile(col=col, row=0, tile_type=AIETileType.ShimNOCTile)
    mem_tile = Tile(col=col, row=1, tile_type=AIETileType.MemTile)

    mem_buf = Buffer(
        tile=mem_tile,
        type=vector_ty,
        name="mem_buf",
        initial_value=np.zeros(TOTAL, dtype=np.int32),
    )

    # The receive BD is self-chained, so slot_credit (one credit per chunk) and
    # the finite input stream are what bound it to N_CHUNKS executions. The
    # readback waits for all N_CHUNKS fills, so it never drains a partial buffer.
    slot_credit = Lock(tile=mem_tile, lock_id=0, init=N_CHUNKS, name="slot_credit")
    fill_count = Lock(tile=mem_tile, lock_id=1, init=0, name="fill_count")

    in_flow = Flow(
        src=shim_tile,
        dst=mem_tile,
        src_port=WireBundle.DMA,
        src_channel=0,
        dst_port=WireBundle.DMA,
        dst_channel=0,
    )
    out_flow = Flow(
        src=mem_tile,
        dst=shim_tile,
        src_port=WireBundle.DMA,
        src_channel=0,
        dst_port=WireBundle.DMA,
        dst_channel=0,
    )

    # One self-chained S2MM BD receives the stream in CHUNK-sized chunks; chunk k
    # lands at offset ((start + k) % n_slots) * CHUNK -- one BD, no BD chain.
    mem_dma = TileDma(
        tile=mem_tile,
        channels=[
            DmaChannel(
                direction=DMAChannelDir.S2MM,
                channel=0,
                bds=[
                    Bd(
                        buffer=mem_buf,
                        offset=0,
                        length=CHUNK,
                        iteration_size=n_slots,
                        iteration_stride=CHUNK,
                        iteration_current=start,
                        acquires=[Acquire(slot_credit, value=1, greater_equal=True)],
                        releases=[Release(fill_count, value=1)],
                        next="self",
                    ),
                ],
            ),
            DmaChannel(
                direction=DMAChannelDir.MM2S,
                channel=0,
                bds=[
                    Bd(
                        buffer=mem_buf,
                        offset=0,
                        length=TOTAL,
                        acquires=[
                            Acquire(fill_count, value=N_CHUNKS, greater_equal=True)
                        ],
                        releases=[Release(slot_credit, value=1)],
                        next="self",
                    ),
                ],
            ),
        ],
    )

    def sequence(a, c):
        in_task = dma_configure_task(shim_tile.op, DMAChannelDir.MM2S, 0)
        with bds(in_task) as bd:
            with bd[0]:
                shim_dma_bd(
                    a.op, offset=0, sizes=[1, 1, 1, TOTAL], strides=[0, 0, 0, 1]
                )
                EndOp()

        out_task = dma_configure_task(
            shim_tile.op, DMAChannelDir.S2MM, 0, issue_token=True
        )
        with bds(out_task) as bd:
            with bd[0]:
                shim_dma_bd(
                    c.op, offset=0, sizes=[1, 1, 1, TOTAL], strides=[0, 0, 0, 1]
                )
                EndOp()

        dma_start_task(in_task, out_task)
        dma_await_task(out_task)
        dma_free_task(in_task)

    rt = Runtime(sequence, [vector_ty, vector_ty])
    rt.add_flow(in_flow)
    rt.add_flow(out_flow)
    for lock in (slot_credit, fill_count):
        rt.add_lock(lock)
    rt.add_tile_dma(mem_dma)

    return Program(iron.get_current_device(), rt).resolve_program()


def _input_data() -> np.ndarray:
    a_np = np.zeros(TOTAL, dtype=np.int32)
    for k in range(N_CHUNKS):
        a_np[k * CHUNK : (k + 1) * CHUNK] = (k + 1) * 1000 + np.arange(
            CHUNK, dtype=np.int32
        )
    return a_np


def _expected(a_np: np.ndarray, n_slots: int, start: int) -> np.ndarray:
    buf = np.zeros(TOTAL, dtype=np.int32)
    for k in range(N_CHUNKS):
        off = ((start + k) % n_slots) * CHUNK
        buf[off : off + CHUNK] = a_np[k * CHUNK : (k + 1) * CHUNK]
    return buf


def _compile_kwargs(opts):
    return dict(col=opts.col, n_slots=opts.slots, start=opts.start)


def _run_and_verify(opts):
    a_np = _input_data()
    a_t = iron.tensor(a_np, dtype=np.int32, device="npu")
    c_t = iron.zeros_like(a_t)

    bd_iteration(a_t, c_t, **_compile_kwargs(opts))

    expected = _expected(a_np, N_CHUNKS, opts.start)
    assert_pass(
        c_t.numpy(), expected, fail_msg="BD iteration sub-buffer placement mismatch"
    )


def main():
    p = argparse.ArgumentParser(prog="AIE BD Iteration")
    add_compile_args(p, dev_choices=("npu2",), default_dev="npu2", with_emit_mlir=True)
    p.add_argument("-c", "--col", type=int, default=0)
    p.add_argument(
        "-s",
        "--slots",
        type=int,
        default=N_SLOTS,
        help=(
            "BD iteration_size (default: %(default)s, one slot per chunk). "
            "Set to 1 to falsify: all chunks collapse onto slot 0."
        ),
    )
    p.add_argument(
        "--start",
        type=int,
        default=0,
        help=(
            "BD iteration_current (default: %(default)s). Nonzero shifts chunk k "
            "to slot (start + k) mod iteration_size."
        ),
    )
    opts = p.parse_args()
    run_design_cli(
        bd_iteration,
        opts,
        compile_kwargs=_compile_kwargs,
        run_and_verify=_run_and_verify,
        device=lambda o: device_from_args(o, n_cols=1),
    )


if __name__ == "__main__":
    main()
