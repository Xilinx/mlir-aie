#!/usr/bin/env python3
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import argparse
import csv
import sys

import aie.iron as iron
import numpy as np
from aie.dialects._aie_enum_gen import (  # pyright: ignore[reportMissingImports]
    AIETileType,
    WireBundle,
)
from aie.dialects.aie import event  # pyright: ignore[reportAttributeAccessIssue]
from aie.iron import (
    CompileTime,
    DeviceConfiguration,
    Flow,
    ObjectFifo,
    Out,
    Program,
    Runtime,
    Worker,
)
from aie.iron.device import NPU2, Tile
from aie.utils.benchmark import run_iters

NPU2_COLUMNS = 8
NPU2_CORE_ROWS = 4
MODES = ("separate-dispatch", "load-pdi", "expand-load-pdis")


@iron.jit
def reconfigure(
    output: Out,
    *,
    mode: CompileTime[str],
    cols: CompileTime[int],
    rows: CompileTime[int],
    nops: CompileTime[int],
    switchboxes: CompileTime[int],
    reconfigs: CompileTime[int],
):
    if not 1 <= cols <= NPU2_COLUMNS:
        raise ValueError(f"cols must be in [1, {NPU2_COLUMNS}]; got {cols}")
    if not 1 <= rows <= NPU2_CORE_ROWS:
        raise ValueError(f"rows must be in [1, {NPU2_CORE_ROWS}]; got {rows}")
    if mode not in MODES:
        raise ValueError(f"unsupported mode: {mode}")

    element_type = np.ndarray[(1,), np.dtype[np.int32]]
    column_type = np.ndarray[(rows,), np.dtype[np.int32]]
    tensor_type = np.ndarray[(cols * rows,), np.dtype[np.int32]]
    full_elf = mode != "separate-dispatch"
    column_fifos = []
    workers = []

    def core(output_fifo, value):
        element = output_fifo.acquire(1)
        element[0] = value
        output_fifo.release(1)
        for _ in range(nops):
            event(0)

    for col in range(cols):
        column_fifo = ObjectFifo(column_type, name=f"column_{col}")
        core_fifos = column_fifo.prod().join(
            list(range(rows)),
            tile=Tile(col, 1),
            obj_types=[element_type] * rows,
            names=[f"core_{col}_{row}" for row in range(rows)],
        )
        column_fifos.append(column_fifo)
        for row, core_fifo in enumerate(core_fifos):
            workers.append(
                Worker(
                    core,
                    [core_fifo.prod(), col * rows + row],
                    tile=Tile(col, row + 2),
                )
            )

    column_handles = [fifo.cons() for fifo in column_fifos]

    def worker_sequence(tensor, *handles):
        for col, handle in enumerate(handles):
            start = col * rows
            handle.drain(tensor, tap=tensor[start : start + rows], wait=True)

    worker_runtime = Runtime(
        worker_sequence,
        [tensor_type, *column_handles],
        implicit_configure=False,
    )

    configuration = None

    def fused_sequence(tensor):
        assert configuration is not None
        for _ in range(reconfigs):
            with configuration.configure():
                worker_runtime.call(tensor)

    fused_runtime = Runtime(
        fused_sequence,
        [tensor_type],
        implicit_configure=False,
    )
    entry = fused_runtime if full_elf else worker_runtime
    configuration = DeviceConfiguration(
        NPU2(),
        workers=workers,
        runtimes=[worker_runtime, fused_runtime] if full_elf else [worker_runtime],
    )

    padding_tiles = [
        (col, row)
        for col in range(1, NPU2_COLUMNS - 1)
        for row in range(2, 2 + NPU2_CORE_ROWS)
        if col >= cols or row >= 2 + rows
    ]
    if not 0 <= switchboxes <= len(padding_tiles):
        raise ValueError(
            f"switchboxes must be in [0, {len(padding_tiles)}]; got {switchboxes}"
        )
    for col, row in padding_tiles[:switchboxes]:
        tile = Tile(col, row, tile_type=AIETileType.CoreTile)
        for channel in range(4):
            configuration.add_flow(
                Flow(
                    tile,
                    tile,
                    src_port=WireBundle.West,
                    src_channel=channel,
                    dst_port=WireBundle.East,
                    dst_channel=channel,
                )
            )
            configuration.add_flow(
                Flow(
                    tile,
                    tile,
                    src_port=WireBundle.East,
                    src_channel=channel,
                    dst_port=WireBundle.West,
                    dst_channel=channel,
                )
            )

    return Program.compose(
        [configuration],
        entry=entry,
        expand_load_pdis=True if mode == "expand-load-pdis" else None,
    ).resolve_program()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=MODES,
        default="load-pdi",
    )
    parser.add_argument("--cols", type=int, default=1)
    parser.add_argument("--rows", type=int, default=1)
    parser.add_argument("--nops", type=int, default=0)
    parser.add_argument("--switchboxes", type=int, default=0)
    parser.add_argument("--reconfigs", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--case", default="manual")
    parser.add_argument("--csv", action="store_true")
    args = parser.parse_args()

    output = iron.zeros(args.cols * args.rows, dtype=np.int32, device="npu")
    design = reconfigure.specialize(
        full_elf=args.mode != "separate-dispatch",
        mode=args.mode,
        cols=args.cols,
        rows=args.rows,
        nops=args.nops,
        switchboxes=args.switchboxes,
        reconfigs=args.reconfigs,
    )
    benchmark = run_iters(
        design,
        output,
        warmup=args.warmup,
        iters=args.iters,
    )
    output.to("cpu")
    np.testing.assert_array_equal(
        output.numpy(), np.arange(args.cols * args.rows, dtype=np.int32)
    )

    stats = benchmark.npu if benchmark.npu is not None else benchmark.e2e
    scope = "npu" if benchmark.npu is not None else "e2e"
    if args.csv:
        writer = csv.writer(sys.stdout)
        for iteration, runtime_us in enumerate(stats.samples_us):
            writer.writerow(
                [
                    args.case,
                    args.mode,
                    args.cols,
                    args.rows,
                    args.nops,
                    args.switchboxes,
                    args.reconfigs,
                    iteration,
                    scope,
                    f"{runtime_us:.3f}",
                ]
            )
    else:
        print("runtimes_us: " + ",".join(f"{t:.3f}" for t in stats.samples_us))
        print(f"stats_us: {stats.avg_us:.3f},{stats.min_us:.3f},{stats.max_us:.3f}")
        print("PASS!")


if __name__ == "__main__":
    main()
