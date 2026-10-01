# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Whole-array GEMM with DispatchTime M, K and N, checked without an NPU.

whole_array_dyn.py compiles once for a capacity and builds its instruction
stream per call from the live shape. This test asks it for that stream at
several shapes and checks, with aie.utils.txn_trace, that the DMA events are
the ones a fully static specialization of the same generator produces, and
the transfers the static whole_array.py issues for that shape. Dispatches that
break the shape constraints are refused by the guards. Run by dispatch_txn.lit.
"""

import sys
from pathlib import Path

import aie.iron as iron
from aie.iron.device import NPU1Col1
from aie.utils.hostruntime.hostruntime import HostRuntimeError
from aie.utils.txn_trace import compare, trace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from whole_array import whole_array  # noqa: E402
from whole_array_dyn import whole_array_dyn  # noqa: E402

iron.set_current_device(NPU1Col1())
TILE = dict(m=32, k=32, n=32, n_aie_cols=1, dtype_in_str="i16", dtype_out_str="i32")
CAPACITY = dict(A_elements=512 * 256, B_elements=256 * 256, C_elements=512 * 256)

dyn = whole_array_dyn.specialize(**CAPACITY, **TILE)


def pushes(words):
    return [e for e in trace(words) if e.kind == "push"]


for M, K, N in (
    (256, 128, 128),
    (384, 128, 128),
    (128, 128, 128),
    (512, 256, 256),
    (256, 64, 32),
    (768, 64, 32),
):
    words = dyn.instructions(M=M, K=K, N=N)
    static = whole_array_dyn.specialize(M=M, K=K, N=N, **CAPACITY, **TILE)
    diffs = compare(words, static.instructions(), names=("dynamic", "static"))
    line = f"M={M} K={K} N={N}: {len(pushes(words))} pushes, matches static specialization={not diffs}"
    # whole_array.py needs an even number of row blocks; where it can compile,
    # the dynamic design must produce exactly its DMA events, waits included.
    if (M // TILE["m"] // 4) % 2 == 0:
        original = whole_array.specialize(M=M, K=K, N=N, **TILE).instructions()
        line += f", same events as whole_array.py={not compare(words, original)}"
    print(line)
    for d in diffs:
        print(d)
# CHECK: M=256 K=128 N=128: 5 pushes, matches static specialization=True, same events as whole_array.py=True
# CHECK: M=384 K=128 N=128: {{[0-9]+}} pushes, matches static specialization=True
# CHECK: M=128 K=128 N=128: {{[0-9]+}} pushes, matches static specialization=True
# CHECK: M=512 K=256 N=256: {{[0-9]+}} pushes, matches static specialization=True, same events as whole_array.py=True
# CHECK: M=256 K=64 N=32: {{[0-9]+}} pushes, matches static specialization=True, same events as whole_array.py=True
# CHECK: M=768 K=64 N=32: {{[0-9]+}} pushes, matches static specialization=True, same events as whole_array.py=True

for bad, why in (
    (dict(M=100, K=128, N=128), "M not a multiple of m*rows"),
    (dict(M=256, K=100, N=128), "K not a multiple of k"),
    (dict(M=512, K=256, N=512), "over capacity"),
):
    try:
        dyn.instructions(**bad)
        print(f"{why}: accepted")
    except HostRuntimeError as e:
        print(f"{why}: refused: {e}")
# The refusal carries the message of the require() that failed.
# CHECK: M not a multiple of m*rows: refused: {{.*}}: M must be a multiple of m * n_aie_rows
# CHECK: K not a multiple of k: refused: {{.*}}: K must be a multiple of k
# CHECK: over capacity: refused: {{.*}}: B (K x N) does not fit B_elements
