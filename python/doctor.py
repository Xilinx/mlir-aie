# doctor.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc.

"""``python -m aie.doctor`` -- report whether this host can run on the NPU.

Kept separate from aie.utils.probe so the library can be imported by the package
without the CLI being loaded twice when run with -m.
"""

from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    from .utils.probe import as_dict, first_actionable, summary

    if "--json" in argv:
        import json

        print(json.dumps(as_dict(), indent=2))
    else:
        print(summary())
    return 0 if first_actionable() is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
