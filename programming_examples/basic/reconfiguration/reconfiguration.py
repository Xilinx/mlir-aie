# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Reconfiguration example.
#
# A 2D array of `cols` x `rows` compute cores each write a single i32 (their own
# global index) into a dedicated ObjectFIFO.  A shim tile has two S2MM channels,
# so the mem tile of each column joins the `rows` core FIFOs and forwards them
# to the shim tile.  A runtime sequence drains every column's values into the
# host output buffer, which then equals [0, 1, ..., cols*rows - 1].
#
# Three build variants are emitted from the same core/join/drain building blocks:
#
#   --flow reconfig  (default): three `aie.device`s (@worker, @empty, @main).
#       @main's runtime sequence, for each reconfiguration, loads @empty (reset)
#       then loads and runs @worker via aiex.configure / aiex.run.  Built for
#       the full-ELF flow (aiecc --get-full-elf, optionally --expand-load-pdis
#       or --load-pdi-to-ctrl-pkt).
#
#   --flow single: a single `aie.device` with no reconfiguration (no load_pdi).
#       The cores loop, so the design is re-run through the ordinary xclbin +
#       insts flow (a single run, or an xrt::runlist of runs).
#
#   --flow empty: a single empty `aie.device`.  Loading its xclbin/PDI resets
#       the array.  The xclbin and runlist benchmarks load it between iterations
#       to force a reconfiguration, because the device caches the configuration.
#
# Parameters: --cols, --rows, --nops (program-memory padding per core),
# --reconfigs (reconfig flow only), --switchboxes (unused switchboxes filled
# with padding stream-switch configuration).
#
# Usage:
#   python3 reconfiguration.py [--flow reconfig|single|empty] \
#       [--cols C] [--rows R] [--nops M] [--reconfigs N] [--switchboxes S] > aie.mlir

import argparse
import sys

import numpy as np
from aie.dialects import arith, memref  # pyright: ignore[reportAttributeAccessIssue]
from aie.dialects._aie_enum_gen import (  # pyright: ignore[reportMissingImports]
    AIEDevice,
    WireBundle,
)
from aie.dialects.aie import (
    EndOp,  # pyright: ignore[reportAttributeAccessIssue]
    ObjectFifoPort,  # pyright: ignore[reportAttributeAccessIssue]
    T,  # pyright: ignore[reportAttributeAccessIssue]
    connect,  # pyright: ignore[reportAttributeAccessIssue]
    core,
    device,
    dma_bd,
    event,  # pyright: ignore[reportAttributeAccessIssue]
    object_fifo,
    object_fifo_link,
    switchbox,
    tile,
)
from aie.dialects.aiex import (
    ConfigureOp,  # pyright: ignore[reportAttributeAccessIssue]
    bds,
    dma_await_task,
    dma_configure_task_for,
    dma_start_task,
    run,  # pyright: ignore[reportAttributeAccessIssue]
    runtime_sequence,
)
from aie.extras.context import mlir_mod_ctx  # pyright: ignore[reportMissingImports]
from aie.ir import (  # pyright: ignore[reportMissingImports]
    Block,  # pyright: ignore[reportAttributeAccessIssue]
    InsertionPoint,  # pyright: ignore[reportAttributeAccessIssue]
)
from aie.iron.controlflow import range_

# Full Strix NPU2: 8 columns, 4 compute rows (rows 2-5), mem tile on row 1.
NPU2_COLUMNS = 8
NPU2_CORE_ROWS = 4

ELEM_TY = np.ndarray[(1,), np.dtype[np.int32]]


def _validate(cols: int, rows: int):
    if not 1 <= cols <= NPU2_COLUMNS:
        raise ValueError(f"cols must be in [1, {NPU2_COLUMNS}]; got {cols}")
    if not 1 <= rows <= NPU2_CORE_ROWS:
        raise ValueError(f"rows must be in [1, {NPU2_CORE_ROWS}]; got {rows}")


def _emit_send(fifo, value_i: int, nop_count: int):
    elem = fifo.acquire(ObjectFifoPort.Produce, 1)
    idx0 = arith.constant(T.index(), 0)
    value = arith.constant(T.i32(), value_i)
    memref.store(value, elem, [idx0])
    fifo.release(ObjectFifoPort.Produce, 1)
    for _ in range(nop_count):
        event(0)


def _get_tile(tiles, c: int, r: int):
    key = (c, r)
    if key not in tiles:
        tiles[key] = tile(c, r)
    return tiles[key]


def _build_core_array(tiles, cols: int, rows: int, nop_count: int, looping: bool):
    """Build the cols x rows core array with per-column mem-tile joins.

    Returns the list of shim-side ObjectFIFOs (one per column), each carrying
    that column's `rows` values.
    """
    col_ty = np.ndarray[(rows,), np.dtype[np.int32]]
    shim_fifos = []
    core_specs = []

    def build_core(core_tile, fifo, value_i):
        @core(core_tile)
        def core_body():
            if looping:
                for _ in range_(sys.maxsize):
                    _emit_send(fifo, value_i, nop_count)
            else:
                _emit_send(fifo, value_i, nop_count)

    for c in range(cols):
        mem_tile = _get_tile(tiles, c, 1)
        shim_tile = _get_tile(tiles, c, 0)
        core_fifos = []
        for r in range(rows):
            core_tile = _get_tile(tiles, c, 2 + r)
            f = object_fifo(f"of_core_{c}_{r}", core_tile, mem_tile, 2, ELEM_TY)
            core_fifos.append(f)
            core_specs.append((core_tile, f, c * rows + r))

        shim_fifo = object_fifo(f"of_shim_{c}", mem_tile, shim_tile, 2, col_ty)
        object_fifo_link(core_fifos, [shim_fifo], list(range(rows)), [])
        shim_fifos.append(shim_fifo)

    # All ObjectFIFOs and links must exist before the cores that reference them,
    # so the objectfifo lowering allocates their locks before any core use.
    for core_tile, fifo, value_i in core_specs:
        build_core(core_tile, fifo, value_i)

    return shim_fifos


def _fill_conns(col: int, row: int):
    """Legal horizontal pass-through connections that fill a compute-tile
    switchbox.

    Each entry is (src_bundle, src_channel, dst_bundle, dst_channel).  The
    connections only carry configuration -- no flow drives them -- so every
    destination is distinct and every connection is individually legal, letting
    the whole set route.  Only the East/West (horizontal) transit is used, which
    needs both an East and a West neighbor; the North/South channels are left
    entirely free so the control-packet reconfiguration approach, which delivers
    its configuration up the columns, still has a legal route to every tile.
    """
    if not 0 < col < 7:
        return []
    return [(WireBundle.West, k, WireBundle.East, k) for k in range(4)] + [
        (WireBundle.East, k, WireBundle.West, k) for k in range(4)
    ]


def _free_switchbox_tiles(cols: int, rows: int):
    """Compute-row switchboxes NOT used by the core array or its drain path.

    The array occupies, per column c < cols: rows 0..(1 + rows) (shim, mem tile
    and `rows` cores).  Every remaining compute-row (2..5) switchbox on an inner
    column is free to fill; outer columns (0 and 7) have no horizontal transit
    (`_fill_conns` is empty) and are excluded.
    """
    free = []
    for c in range(NPU2_COLUMNS):
        used = set(range(2, 2 + rows)) if c < cols else set()
        for r in range(2, 6):
            if r not in used and _fill_conns(c, r):
                free.append((c, r))
    return free


def max_switchboxes(cols: int, rows: int) -> int:
    return len(_free_switchbox_tiles(cols, rows))


def _fill_switchboxes(tiles, cols: int, rows: int, count: int):
    """Emit `count` filled ``aie.switchbox`` regions on free compute tiles.

    Programming stream-switch connections directly (rather than routing flows)
    grows the switchbox configuration a reconfiguration must load, with no data
    moving.  The regions are emitted before the cores so their tile declarations
    dominate the ObjectFIFO lowering that follows.
    """
    if count <= 0:
        return
    free = _free_switchbox_tiles(cols, rows)
    if count > len(free):
        raise ValueError(
            f"cannot fill {count} switchboxes in a {cols}x{rows} array "
            f"(max {len(free)})"
        )

    def emit(tile_op, conns):
        @switchbox(tile_op)
        def _sb():
            for conn in conns:
                connect(*conn)
            EndOp()

    for c, r in free[:count]:
        emit(_get_tile(tiles, c, r), _fill_conns(c, r))


def _emit_drain(shim_fifos, rows: int, out):
    tasks = []
    for c, fifo in enumerate(shim_fifos):
        task = dma_configure_task_for(fifo, issue_token=True)
        with bds(task) as bd:
            with bd[0]:
                dma_bd(out, offset=c * rows, transfer_len=rows)
                EndOp()
        tasks.append(task)
    for task in tasks:
        dma_start_task(task)
    for task in tasks:
        dma_await_task(task)


def build_reconfig_module(
    cols: int, rows: int, nop_count: int, num_reconfigs: int, num_switchboxes: int = 0
):
    _validate(cols, rows)
    out_ty = np.ndarray[(cols * rows,), np.dtype[np.int32]]

    with mlir_mod_ctx() as ctx:  # pyright: ignore[reportGeneralTypeIssues]

        @device(AIEDevice.npu2, sym_name="worker")
        def worker_body():
            tiles = {}
            _fill_switchboxes(tiles, cols, rows, num_switchboxes)
            shim_fifos = _build_core_array(tiles, cols, rows, nop_count, looping=False)

            @runtime_sequence(out_ty)
            def worker_sequence(out):
                _emit_drain(shim_fifos, rows, out)

        # Loading a PDI resets the whole device.  The firmware treats a second
        # load of the same PDI back-to-back as a no-op, so each reconfiguration
        # loads @empty first to force an actual @worker reload (which also makes
        # the sequence self-resetting when the ELF is re-run: it ends on @worker
        # and the next run starts on @empty).
        @device(AIEDevice.npu2, sym_name="empty")
        def empty_body():
            pass

        @device(AIEDevice.npu2, sym_name="main")
        def main_body():
            def configure(device_symbol, run_symbol=None, run_args=()):
                config = ConfigureOp(device_symbol)
                body = Block.create_at_start(config.body)
                if run_symbol is not None:
                    with InsertionPoint(body):
                        run(run_symbol, list(run_args))

            @runtime_sequence(out_ty)
            def sequence(out):
                for _ in range(num_reconfigs):
                    configure("empty")
                    configure("worker", "worker_sequence", [out])

    return ctx.module


def build_single_module(cols: int, rows: int, nop_count: int, num_switchboxes: int = 0):
    _validate(cols, rows)
    out_ty = np.ndarray[(cols * rows,), np.dtype[np.int32]]

    with mlir_mod_ctx() as ctx:  # pyright: ignore[reportGeneralTypeIssues]

        @device(AIEDevice.npu2)
        def device_body():
            tiles = {}
            _fill_switchboxes(tiles, cols, rows, num_switchboxes)
            shim_fifos = _build_core_array(tiles, cols, rows, nop_count, looping=True)

            @runtime_sequence(out_ty)
            def sequence(out):
                _emit_drain(shim_fifos, rows, out)

    return ctx.module


def build_empty_module():
    with mlir_mod_ctx() as ctx:  # pyright: ignore[reportGeneralTypeIssues]

        @device(AIEDevice.npu2)
        def device_body():
            @runtime_sequence()
            def sequence():
                pass

    return ctx.module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--flow",
        choices=["reconfig", "single", "empty"],
        default="reconfig",
    )
    parser.add_argument("--cols", type=int, default=1)
    parser.add_argument("--rows", type=int, default=1)
    parser.add_argument(
        "--nops",
        type=int,
        default=0,
        help="Number of no-op instructions padding each core's program memory.",
    )
    parser.add_argument(
        "--reconfigs",
        type=int,
        default=1,
        help="Number of empty+worker reconfigurations (reconfig flow only).",
    )
    parser.add_argument(
        "--switchboxes",
        type=int,
        default=0,
        help="Number of unused compute-tile switchboxes filled with padding "
        "stream-switch configuration.",
    )
    args = parser.parse_args()
    if args.flow == "single":
        print(build_single_module(args.cols, args.rows, args.nops, args.switchboxes))
    elif args.flow == "empty":
        print(build_empty_module())
    else:
        print(
            build_reconfig_module(
                args.cols, args.rows, args.nops, args.reconfigs, args.switchboxes
            )
        )


if __name__ == "__main__":
    main()
