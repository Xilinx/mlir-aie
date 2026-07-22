# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Reconfiguration example.
#
# An array of compute cores each write a single i32 (their own index) into a
# dedicated ObjectFIFO routed down to the shim, followed by a run of no-op
# `aie.event` instructions padding their program memory.  A runtime sequence
# drains one i32 from every core into the host output buffer.
#
# Two flows are emitted from the same core/drain building blocks:
#
#   --flow reconfig  (default): three `aie.device`s (@worker, @empty, @main).
#       @main's runtime sequence repeatedly reconfigures and runs @worker via
#       aiex.configure / aiex.run.  Each reconfiguration reloads a PDI, which
#       resets the device and restarts @worker's run-once cores.  @empty is
#       loaded between @worker loads to force an actual reload.  Built for the
#       full-ELF flow (aiecc --generate-full-elf).
#
#   --flow single: a single `aie.device` with no reconfiguration (no load_pdi).
#       The cores loop, so the design can be re-run through the ordinary
#       xclbin + insts flow (a single run, or an xrt::runlist of runs).
#
# The design is parametrized by:
#   * number of cores         (--cores     / RECONFIG_NUM_CORES)
#   * program-memory padding  (--nops      / RECONFIG_NOP_COUNT)   no-ops per core
#   * number of reconfigs     (--reconfigs / RECONFIG_NUM_RECONFIGS)
#     (reconfig flow only; the single flow's re-runs are driven host-side)
#
# Usage:
#   python3 reconfiguration.py [--flow reconfig|single] \
#       [--cores N] [--nops M] [--reconfigs R] > aie.mlir

import argparse
import os
import sys

import numpy as np

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.dialects import arith, memref
from aie.extras.context import mlir_mod_ctx
from aie.ir import Block, InsertionPoint
from aie.iron.controlflow import range_

# Full Strix NPU2: one shim/compute column per core, cores on row 2.
NPU2_COLUMNS = 8

ELEM_TY = np.ndarray[(1,), np.dtype[np.int32]]


def _validate(num_cores: int):
    if not 1 <= num_cores <= NPU2_COLUMNS:
        raise ValueError(
            f"num_cores must be in [1, {NPU2_COLUMNS}] so each core gets its own "
            f"column/shim; got {num_cores}"
        )


def _emit_send(fifo, value_i: int, nop_count: int):
    elem = fifo.acquire(ObjectFifoPort.Produce, 1)
    idx0 = arith.constant(T.index(), 0)
    value = arith.constant(T.i32(), value_i)
    memref.store(value, elem, [idx0])
    fifo.release(ObjectFifoPort.Produce, 1)
    for _ in range(nop_count):
        event(0)


def _build_core_array(num_cores: int, nop_count: int, looping: bool):
    core_tiles = [tile(c, 2) for c in range(num_cores)]
    shim_tiles = [tile(c, 0) for c in range(num_cores)]
    fifos = [
        object_fifo(f"of_out_{i}", core_tiles[i], shim_tiles[i], 1, ELEM_TY)
        for i in range(num_cores)
    ]

    def build_core(core_tile, fifo, value_i):
        @core(core_tile)
        def core_body():
            if looping:
                for _ in range_(sys.maxsize):
                    _emit_send(fifo, value_i, nop_count)
            else:
                _emit_send(fifo, value_i, nop_count)

    for i in range(num_cores):
        build_core(core_tiles[i], fifos[i], i)
    return fifos


def _emit_drain(fifos, out):
    tasks = []
    for i, fifo in enumerate(fifos):
        task = dma_configure_task_for(fifo, issue_token=True)
        with bds(task) as bd:
            with bd[0]:
                dma_bd(out, offset=i, transfer_len=1)
                EndOp()
        tasks.append(task)
    for task in tasks:
        dma_start_task(task)
    for task in tasks:
        dma_await_task(task)


def build_reconfig_module(num_cores: int, nop_count: int, num_reconfigs: int):
    _validate(num_cores)
    out_ty = np.ndarray[(num_cores,), np.dtype[np.int32]]

    with mlir_mod_ctx() as ctx:

        @device(AIEDevice.npu2, sym_name="worker")
        def worker_body():
            fifos = _build_core_array(num_cores, nop_count, looping=False)

            @runtime_sequence(out_ty)
            def worker_sequence(out):
                _emit_drain(fifos, out)

        # Loading a PDI resets the whole device.  The firmware treats a second
        # load of the same PDI back-to-back as a no-op, so between two @worker
        # configurations we load this @empty device to force an actual reload
        # (and thus restart @worker's run-once cores).
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
                for i in range(num_reconfigs):
                    if i > 0:
                        configure("empty")
                    configure("worker", "worker_sequence", [out])

    return ctx.module


def build_single_module(num_cores: int, nop_count: int):
    _validate(num_cores)
    out_ty = np.ndarray[(num_cores,), np.dtype[np.int32]]

    with mlir_mod_ctx() as ctx:

        @device(AIEDevice.npu2)
        def device_body():
            fifos = _build_core_array(num_cores, nop_count, looping=True)

            @runtime_sequence(out_ty)
            def sequence(out):
                _emit_drain(fifos, out)

    return ctx.module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--flow",
        choices=["reconfig", "single"],
        default="reconfig",
        help="reconfig: multi-device aiex.configure/run (full-ELF). "
        "single: one device, no load_pdi (xclbin + insts flow).",
    )
    parser.add_argument(
        "--cores",
        type=int,
        default=int(os.environ.get("RECONFIG_NUM_CORES", "1")),
        help="Number of compute cores in the array.",
    )
    parser.add_argument(
        "--nops",
        type=int,
        default=int(os.environ.get("RECONFIG_NOP_COUNT", "0")),
        help="Number of no-op instructions padding each core's program memory.",
    )
    parser.add_argument(
        "--reconfigs",
        type=int,
        default=int(os.environ.get("RECONFIG_NUM_RECONFIGS", "1")),
        help="Number of times the worker device is configured and run "
        "(reconfig flow only).",
    )
    args = parser.parse_args()
    if args.flow == "single":
        print(build_single_module(args.cores, args.nops))
    else:
        print(build_reconfig_module(args.cores, args.nops, args.reconfigs))


if __name__ == "__main__":
    main()
