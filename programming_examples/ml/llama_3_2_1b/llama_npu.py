#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""How the example's tests get the NPU kernel for the design they check."""

import argparse
import sys

from aie.utils import NPUKernel


def add_artifact_args(parser: argparse.ArgumentParser) -> None:
    """Add `--xclbin`/`--instr`, a prebuilt design to run, and `-v`."""
    parser.add_argument(
        "--xclbin",
        help="run this prebuilt xclbin instead of compiling the design",
    )
    parser.add_argument("--instr", help="the prebuilt xclbin's instructions")
    parser.add_argument("-v", "--verbosity", type=int, default=0)


def design_kernel(design, opts: argparse.Namespace, **compile_kwargs) -> NPUKernel:
    """The NPU kernel for `design` at `compile_kwargs`.

    Args:
        design: The `@iron.jit` design the test checks.
        opts: Parsed flags from `add_artifact_args`.
        **compile_kwargs: The design's CompileTime values, which must match the
            prebuilt artifacts when `opts.xclbin` names them.

    Returns:
        A kernel over `opts.xclbin`/`opts.instr` if given, else over the design
        compiled (or found in the JIT cache) at `compile_kwargs`.
    """
    if opts.xclbin is None:
        xclbin, instr = design.specialize(**compile_kwargs).compile()
    elif opts.instr is None:
        sys.exit("--xclbin needs --instr")
    else:
        xclbin, instr = opts.xclbin, opts.instr
    return NPUKernel(
        xclbin_path=str(xclbin), insts_path=str(instr), kernel_name="MLIR_AIE"
    )
