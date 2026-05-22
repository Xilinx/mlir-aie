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
            *names, type=str, choices=list(dev_choices), default=default_dev
        )
    if with_emit_mlir:
        parser.add_argument(
            "--emit-mlir",
            action="store_true",
            help="print the resolved MLIR module to stdout (legacy aiecc / vck5000 path)",
        )
    parser.add_argument("--xclbin-path", type=str, default=None)
    parser.add_argument("--insts-path", type=str, default=None)
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
    parser.add_argument("-w", "--warmup", type=int, default=default_warmup)
    parser.add_argument("-i", "--iters", type=int, default=default_iters)


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
    if with_short:
        parser.add_argument("-t", "--trace_size", type=int, default=default)
    else:
        parser.add_argument("--trace_size", type=int, default=default)
