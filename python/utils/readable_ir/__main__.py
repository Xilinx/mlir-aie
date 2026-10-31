# __main__.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""`python -m aie.utils.readable_ir FILE`: print FILE with locations as comments."""

import argparse
import sys

from aie.utils.readable_ir import readable


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Print MLIR with each op's location as a short trailing comment."
    )
    parser.add_argument("input", help="MLIR file printed with debug info, or -")
    parser.add_argument("-o", "--output", help="Write here instead of stdout")
    args = parser.parse_args(argv)
    if args.input == "-":
        text = sys.stdin.read()
    else:
        with open(args.input) as f:
            text = f.read()
    result = readable(text)
    if args.output:
        with open(args.output, "w") as f:
            f.write(result)
    else:
        sys.stdout.write(result)
    return 0


sys.exit(main())
