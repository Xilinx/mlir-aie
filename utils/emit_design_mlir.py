#!/usr/bin/env python3
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Run a design script and capture its MLIR into a file.

CMake shim for ``add_aie_mlir_design()`` in
``programming_examples/common.cmake``. The examples this serves print their
module on stdout, which the Makefiles capture with ``> build/aie.mlir``.
``add_custom_command(... VERBATIM)`` escapes ``>`` into a literal argument, and
dropping VERBATIM to get shell redirection would give up the quoting that makes
the rest of these rules safe on Windows. So the redirect happens here instead.

Writing through a temporary file keeps a failed run from leaving a truncated
.mlir behind that the build system would then treat as up to date.
"""

import argparse
import os
import subprocess
import sys
import tempfile


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("-o", "--output", required=True, help="file to write MLIR to")
    p.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="the design script invocation, e.g. python design.py -d npu2",
    )
    args = p.parse_args(argv)

    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        p.error("no command given")

    outdir = os.path.dirname(os.path.abspath(args.output)) or "."
    os.makedirs(outdir, exist_ok=True)

    fd, tmp = tempfile.mkstemp(dir=outdir, suffix=".mlir.tmp")
    try:
        with os.fdopen(fd, "wb") as out:
            ret = subprocess.run(command, stdout=out)
        if ret.returncode != 0:
            return ret.returncode
        os.replace(tmp, args.output)
        tmp = None
    finally:
        if tmp is not None:
            os.unlink(tmp)
    return 0


if __name__ == "__main__":
    sys.exit(main())
