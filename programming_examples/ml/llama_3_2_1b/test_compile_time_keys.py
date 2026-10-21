#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Each decode value that changes the build is a CompileTime value, so two
values give two JIT cache entries and two MLIR modules, never a stale one."""

import sys

import aie2_decode

CASES = [
    ("P_MAX", 4, 3),
    ("N_LAYERS", 2, 3),
    ("T", 128, 256),
    ("KSET", 8, 4),
]


def main():
    failed = 0
    for name, a, b in CASES:
        designs = [aie2_decode.build.specialize(**{name: v}).compilable for v in (a, b)]
        keys = {d.recipe_hash for d in designs}
        texts = {str(d.generate_mlir()) for d in designs}
        ok = len(keys) == 2 and len(texts) == 2
        print(f"{name}: {a} vs {b}: {'PASS' if ok else 'FAIL'}")
        failed += not ok
    print(f"compile_time_keys: {len(CASES) - failed}/{len(CASES)} PASS")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
