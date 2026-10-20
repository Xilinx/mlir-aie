#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Each llama design value that changes the build is a CompileTime value, so
two values give two JIT cache entries, never a stale one. A value the MLIR
carries also changes the MLIR; a gate scale goes only into the SiLU kernel's
compile flags, so for it the cache key is what differs."""

import sys

import aie2_decode
import aie2_layer_mh
import aie2_silu_mul_int8

CASES = [
    (aie2_decode.build, {}, "P_MAX", 4, 3, True),
    (aie2_decode.build, {}, "N_LAYERS", 2, 3, True),
    (aie2_decode.build, {}, "T", 128, 256, True),
    (aie2_decode.build, {}, "KSET", 8, 4, True),
    (aie2_decode.build, {}, "SILU_GATE_SCALE", 0.05, 0.04, True),
    (aie2_layer_mh.build, {}, "SILU_GATE_SCALE", 0.05, 0.04, True),
    (aie2_silu_mul_int8.build, {"SELFCAL": True}, "GATE_SCALE", 0.05, 0.04, False),
]


def main():
    failed = 0
    for design, base, name, a, b, in_mlir in CASES:
        designs = [design.specialize(**base, **{name: v}).compilable for v in (a, b)]
        keys = {d.recipe_hash for d in designs}
        texts = {str(d.generate_mlir()) for d in designs}
        ok = len(keys) == 2 and len(texts) == (2 if in_mlir else 1)
        print(
            f"{design.compilable.generator_name} {name}: {a} vs {b}: {'PASS' if ok else 'FAIL'}"
        )
        failed += not ok
    print(f"compile_time_keys: {len(CASES) - failed}/{len(CASES)} PASS")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
