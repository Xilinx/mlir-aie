# flow_endpoints/flow_endpoints.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# REQUIRES: ryzen_ai_npu2, peano, xrt_python_bindings
#
# RUN: %python %s -d npu2 --emit-mlir | aie-opt --aie-place-tiles \
# RUN:   --aie-objectfifo-allocate | FileCheck %s --check-prefix=ALLOC
# RUN: %run_on_npu2% %python %s -d npu2
"""Flow endpoints: DMA programs on channels the compiler picks, on tiles it places.

No tile is pinned and no Flow names a channel. A DMA program, static
(``TileDma``) or issued from the sequence (``tile_dma_chain``), runs on
``flow.endpoint(tile)`` instead of an index, and ``--aie-objectfifo-allocate``
gives every endpoint its channel.

The mem tile carries five such endpoints: three S2MM and two MM2S.

* A shim fill lands in ``staged`` through a static S2MM program.
* The sequence broadcasts ``staged`` to two cores with a chain on the mem
  tile's MM2S end of the broadcast.
* Each core's static program sends its copy back to the mem tile. Core 1 sends
  it transposed, so the two paths land differently.
* The mem tile gathers both copies into ``gathered`` (two static S2MM
  programs) and drains it to the host through a static MM2S program.

A channel the compiler handed to two endpoints, or a program left on a
channel its flow does not use, loses data or hangs the run.

No compute core runs: the core tiles only use their DMAs.
"""

import argparse

import aie.iron as iron
import numpy as np
from aie.dialects._aie_enum_gen import (  # pyright: ignore[reportMissingImports]
    AIETileType,
    DMAChannelDir,
)
from aie.iron import (
    Acquire,
    Bd,
    Buffer,
    DmaChannel,
    Flow,
    In,
    Lock,
    Out,
    Program,
    Release,
    Runtime,
    TileDma,
    tile_dma_chain,
)
from aie.iron.device import Tile
from aie.utils.hostruntime.argparse import add_compile_args, device_from_args
from aie.utils.hostruntime.cli import run_design_cli
from aie.utils.verify import assert_pass

SIDE = 32
N = SIDE * SIDE  # int32 elements in one copy

# After allocation every mem tile program names an index and the broadcast's
# chain runs on the channel the broadcast's flows leave from.
# ALLOC-LABEL: aie.memtile_dma
# ALLOC-COUNT-3: aie.dma_start(S2MM, {{[0-9]}},
# ALLOC: aie.dma_start(MM2S, {{[0-9]}},
# ALLOC-LABEL: aie.runtime_sequence
# ALLOC: aiex.dma_configure_task_for @flow0_src_shim_alloc
# ALLOC: aiex.dma_configure_task(%[[MEM:.*]], MM2S, [[B:[0-9]]])
# ALLOC: aiex.dma_configure_task_for @flow4_dst_shim_alloc
# ALLOC-COUNT-2: aie.flow(%[[MEM]], DMA : [[B]], %{{.*}}, DMA : {{[0-9]}})


@iron.jit
def flow_endpoints(a_in: In, c_out: Out):
    copy_ty = np.ndarray[(N,), np.dtype[np.int32]]
    pair_ty = np.ndarray[(2 * N,), np.dtype[np.int32]]

    shim = Tile(tile_type=AIETileType.ShimNOCTile)
    mem = Tile(tile_type=AIETileType.MemTile)
    cores = [Tile(tile_type=AIETileType.CoreTile) for _ in range(2)]

    into = Flow(shim, mem)
    spread = Flow(mem, cores)
    gather = [Flow(core, mem) for core in cores]
    out = Flow(mem, shim)

    staged = Buffer(tile=mem, type=copy_ty, name="staged")
    gathered = Buffer(tile=mem, type=pair_ty, name="gathered")
    landed = [
        Buffer(tile=core, type=copy_ty, name=f"landed{i}")
        for i, core in enumerate(cores)
    ]

    staged_free = Lock(tile=mem, init=1, name="staged_free")
    staged_full = Lock(tile=mem, init=0, name="staged_full")
    gathered_free = Lock(tile=mem, init=2, name="gathered_free")
    gathered_full = Lock(tile=mem, init=0, name="gathered_full")
    landed_free = [
        Lock(tile=core, init=1, name=f"landed{i}_free") for i, core in enumerate(cores)
    ]
    landed_full = [
        Lock(tile=core, init=0, name=f"landed{i}_full") for i, core in enumerate(cores)
    ]

    def take(lock, value=1):
        return Acquire(lock, value=value, greater_equal=True)

    mem_dma = TileDma(
        mem,
        [
            DmaChannel(
                DMAChannelDir.S2MM,
                into.endpoint(mem),
                [
                    Bd(
                        staged,
                        acquires=[take(staged_free)],
                        releases=[Release(staged_full, value=1)],
                    )
                ],
            ),
            *(
                DmaChannel(
                    DMAChannelDir.S2MM,
                    fl.endpoint(mem),
                    [
                        Bd(
                            gathered,
                            offset=i * N,
                            length=N,
                            acquires=[take(gathered_free)],
                            releases=[Release(gathered_full, value=1)],
                        )
                    ],
                )
                for i, fl in enumerate(gather)
            ),
            DmaChannel(
                DMAChannelDir.MM2S,
                out.endpoint(mem),
                [
                    Bd(
                        gathered,
                        acquires=[take(gathered_full, 2)],
                        releases=[Release(gathered_free, value=2)],
                    )
                ],
            ),
        ],
    )

    def send_back(i):
        # Core 1 sends its copy transposed.
        return Bd(
            landed[i],
            sizes=[SIDE, SIDE] if i else [],
            strides=[1, SIDE] if i else [],
            acquires=[take(landed_full[i])],
            releases=[Release(landed_free[i], value=1)],
        )

    core_dmas = [
        TileDma(
            core,
            [
                DmaChannel(
                    DMAChannelDir.S2MM,
                    spread.endpoint(core),
                    [
                        Bd(
                            landed[i],
                            acquires=[take(landed_free[i])],
                            releases=[Release(landed_full[i], value=1)],
                        )
                    ],
                ),
                DmaChannel(
                    DMAChannelDir.MM2S, gather[i].endpoint(core), [send_back(i)]
                ),
            ],
        )
        for i, core in enumerate(cores)
    ]

    def sequence(a, c):
        into.fill(a)
        tile_dma_chain(
            mem,
            DMAChannelDir.MM2S,
            spread.endpoint(mem),
            [
                Bd(
                    staged,
                    acquires=[take(staged_full)],
                    releases=[Release(staged_free, value=1)],
                )
            ],
        )
        out.drain(c, wait=True)

    rt = Runtime(sequence, [copy_ty, pair_ty])
    for fl in (into, spread, *gather, out):
        rt.add_flow(fl)
    for lock in (
        staged_free,
        staged_full,
        gathered_free,
        gathered_full,
        *landed_free,
        *landed_full,
    ):
        rt.add_lock(lock)
    for td in (mem_dma, *core_dmas):
        rt.add_tile_dma(td)
    rt.add_buffer(staged)

    return Program(iron.get_current_device(), rt).resolve_program()


def _run_and_verify(opts):
    a_np = np.arange(N, dtype=np.int32) + 1000
    a_t = iron.tensor(a_np, dtype=np.int32, device="npu")
    c_t = iron.zeros(2 * N, dtype=np.int32, device="npu")

    flow_endpoints(a_t, c_t)

    expected = np.concatenate([a_np, a_np.reshape(SIDE, SIDE).T.ravel()])
    assert_pass(c_t.numpy(), expected, fail_msg="gathered copies mismatch")


def main():
    p = argparse.ArgumentParser(prog="AIE Flow Endpoints")
    add_compile_args(p, dev_choices=("npu2",), default_dev="npu2", with_emit_mlir=True)
    opts = p.parse_args()
    run_design_cli(
        flow_endpoints,
        opts,
        compile_kwargs=lambda opts: {},
        run_and_verify=_run_and_verify,
        device=lambda o: device_from_args(o, n_cols=1),
    )


if __name__ == "__main__":
    main()
