# argparse.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc.
"""Reusable ``argparse`` flag groups shared by the programming examples.

Almost every basic/ design repeats the same handful of CLI flags:
``-d/--dev`` (device selector), ``--xclbin-path``/``--insts-path``
(compile-only output paths), optionally ``--elf-path`` (xrt::elf
testbench) and ``--emit-mlir`` (legacy aiecc / vck5000 path), and the
benchmark/trace pair ``-w/--warmup`` / ``-i/--iters`` / ``-t/--trace_size``.

The helpers in this module mutate an existing ``argparse.ArgumentParser``
(so each design can keep its own design-specific flags and ``prog``).
"""

from __future__ import annotations

import argparse

DEFAULT_DEV_CHOICES: tuple[str, ...] = ("npu", "npu2")


def add_compile_args(
    parser: argparse.ArgumentParser,
    *,
    with_dev: bool = True,
    dev_choices: tuple[str, ...] = DEFAULT_DEV_CHOICES,
    default_dev: str = "npu",
    short_dev: str | None = "-d",
    with_elf: bool = False,
    with_emit_mlir: bool = False,
) -> None:
    """Add the standard compile-mode flags.

    Args:
        parser: Parser to mutate.
        with_dev: If True (default), add ``-d/--dev`` (or just ``--dev``
            when ``short_dev=None``).
        dev_choices: Allowed device-name strings (default
            ``("npu", "npu2")``).  Pass e.g. ``("npu", "npu2", "xcvc1902")``
            for designs that also accept the VCK5000 target.
        default_dev: Default value for ``--dev``.
        short_dev: Short-option for ``--dev``.  ``None`` to skip the short
            opt (matmul designs use ``--dev`` only because ``-d`` is
            already taken by something else).
        with_elf: When True, also add ``--elf-path`` (for testbenches
            that load the insts as an ``xrt::elf`` module instead of a
            raw ``insts.bin``).
        with_emit_mlir: When True, also add ``--emit-mlir`` (action
            ``store_true``) — for designs that have a vck5000 / aiecc
            print-MLIR path next to the @iron.jit NPU path.
    """
    if with_dev:
        names = ("--dev",) if short_dev is None else (short_dev, "--dev")
        parser.add_argument(
            *names,
            type=str,
            choices=list(dev_choices),
            default=default_dev,
            help="target device family (default: %(default)s)",
        )
    if with_emit_mlir:
        parser.add_argument(
            "--emit-mlir",
            action="store_true",
            help="print the resolved MLIR module to stdout (legacy aiecc / vck5000 path)",
        )
    parser.add_argument(
        "--xclbin-path",
        type=str,
        default=None,
        help="compile-only mode: write the xclbin here (pairs with --insts-path)",
    )
    parser.add_argument(
        "--insts-path",
        type=str,
        default=None,
        help="compile-only mode: write the instruction binary here (pairs with --xclbin-path)",
    )
    if with_elf:
        parser.add_argument(
            "--elf-path",
            type=str,
            default=None,
            help="optional ELF-wrapped insts (for the test.cpp xrt::elf flow)",
        )


def add_benchmark_args(
    parser: argparse.ArgumentParser,
    *,
    default_warmup: int = 2,
    default_iters: int = 5,
) -> None:
    """Add the standard benchmark flags: ``-w/--warmup`` and ``-i/--iters``."""
    parser.add_argument(
        "-w",
        "--warmup",
        type=int,
        default=default_warmup,
        help="benchmark warmup iterations (excluded from timings; default: %(default)s)",
    )
    parser.add_argument(
        "-i",
        "--iters",
        type=int,
        default=default_iters,
        help="benchmark timed iterations (default: %(default)s)",
    )


def add_trace_arg(
    parser: argparse.ArgumentParser,
    *,
    with_short: bool = True,
    default: int = 0,
) -> None:
    """Add the standard ``--trace_size`` flag.

    Args:
        parser: Parser to mutate.
        with_short: When True (default), exposes ``-t`` as a short opt.
            Set False for designs whose ``-t`` is already taken (e.g.
            matmul).
        default: Default trace size in bytes (``0`` disables tracing).
    """
    names = ("-t", "--trace_size") if with_short else ("--trace_size",)
    parser.add_argument(
        *names,
        type=int,
        default=default,
        help="hardware trace buffer size in bytes (0 disables tracing; default: %(default)s)",
    )


def add_runtime_args(
    parser: argparse.ArgumentParser,
    *,
    with_io_sizes: bool = False,
    with_benchmark: bool = False,
) -> None:
    """Add the standard runtime / test-harness flags.

    Pairs with :func:`add_compile_args`: ``add_compile_args`` covers the
    write-side flags (``--xclbin-path``, ``--insts-path``) used by JIT
    designs, while this helper covers the read-side flags (``--xclbin``,
    ``--instr``) used by ``test.py``-style host harnesses that load a
    pre-compiled xclbin and run it on the NPU.

    Args:
        parser: Parser to mutate.
        with_io_sizes: When True, adds ``--in1-size`` / ``--in2-size`` /
            ``--out-size`` (bytes, ``int``) for designs whose Makefile
            drives buffer sizes from the test harness.
        with_benchmark: When True, also calls :func:`add_benchmark_args`
            (adds ``-i/--iters`` and ``-w/--warmup``).  Off by default
            because most correctness-test harnesses do not benchmark.

    Adds (always): ``--xclbin``, ``--instr``, ``-k/--kernel``,
    ``-v/--verbosity``, ``--verify``/``--no-verify``, ``--trace-file``,
    ``--ddr-id``, ``--enable-ctrl-pkts``; and via :func:`add_trace_arg`,
    ``-t/--trace_size``.
    """
    parser.add_argument(
        "--xclbin",
        type=str,
        required=True,
        help="path to the pre-compiled xclbin to load",
    )
    parser.add_argument(
        "--instr",
        type=str,
        default="instr.txt",
        help="path of file containing userspace instructions sent to the NPU (default: %(default)s)",
    )
    parser.add_argument(
        "-k",
        "--kernel",
        type=str,
        default="MLIR_AIE",
        help="kernel name in the xclbin (default: %(default)s)",
    )
    parser.add_argument(
        "-v",
        "--verbosity",
        type=int,
        default=0,
        help="verbosity level (default: %(default)s)",
    )
    parser.add_argument(
        "--verify",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="verify the NPU output against the reference (default: %(default)s)",
    )
    add_trace_arg(parser)
    parser.add_argument(
        "--trace-file",
        type=str,
        default="trace.txt",
        help="where to store trace output (default: %(default)s)",
    )
    parser.add_argument(
        "--ddr-id",
        type=int,
        default=4,
        help="DDR buffer index for trace (0-4, or -1 to append after last tensor; default: %(default)s)",
    )
    parser.add_argument(
        "--enable-ctrl-pkts",
        action="store_true",
        help="enable control packets",
    )
    if with_io_sizes:
        parser.add_argument(
            "--in1-size",
            type=int,
            default=0,
            help="input 1 buffer size in bytes (default: %(default)s)",
        )
        parser.add_argument(
            "--in2-size",
            type=int,
            default=0,
            help="input 2 buffer size in bytes (default: %(default)s)",
        )
        parser.add_argument(
            "--out-size",
            type=int,
            default=0,
            help="output buffer size in bytes (default: %(default)s)",
        )
    if with_benchmark:
        add_benchmark_args(parser)
