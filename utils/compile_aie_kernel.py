#!/usr/bin/env python3
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Compile one AIE core function to an object file.

CMake shim for ``compile_cxx_core_function``, used by ``add_aie_kernel_object()``
in ``programming_examples/common.cmake``. Same arrangement as
``utils/run_on_npu.py``: the logic stays in the Python package, and CMake only
shells out to it.

The point is that the Peano/Chess flag sets live in exactly one place. The
Makefiles spell them out as ``PEANOWRAP2_FLAGS`` / ``CHESSCCWRAP2P_FLAGS`` in
``programming_examples/makefile-common``, and re-encoding those in CMake would
give the CMake path its own copy to drift from.
"""

import argparse
import sys

from aie.utils.compile import compile_cxx_core_function

# Device family as the examples spell it -> the architecture the compilers take.
_ARCH = {"npu": "aie2", "npu2": "aie2p"}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", help="C++ kernel source (.cc)")
    p.add_argument("-o", "--output", required=True, help="output object (.o)")
    p.add_argument(
        "-d",
        "--device",
        required=True,
        choices=sorted(_ARCH),
        help="NPU device family the kernel is compiled for",
    )
    p.add_argument(
        "-D",
        "--define",
        action="append",
        default=[],
        metavar="NAME[=VALUE]",
        help="preprocessor definition (repeatable)",
    )
    p.add_argument(
        "-I",
        "--include-dir",
        action="append",
        default=[],
        metavar="DIR",
        help="additional include directory (repeatable)",
    )
    p.add_argument(
        "--chess",
        action="store_true",
        help="compile with xchesscc_wrapper instead of Peano",
    )
    args = p.parse_args(argv)

    compile_cxx_core_function(
        source_path=args.source,
        target_arch=_ARCH[args.device],
        output_path=args.output,
        include_dirs=args.include_dir,
        compile_args=[f"-D{d}" for d in args.define],
        use_chess=args.chess,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
