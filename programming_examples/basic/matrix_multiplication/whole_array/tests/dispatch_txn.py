# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Whole-array GEMM with DispatchTime M, K and N, checked without an NPU.

Left unspecialized, whole_array.py compiles once for its host buffers and
builds its instruction stream per call from the live shape. This test asks it
for that stream at several shapes and checks, with aie.utils.txn_trace, that
the DMA events are the ones a fully static specialization of the same design
produces. Dispatches that break the shape constraints, or that do not fit the
buffers, are refused. Run by dispatch_txn.lit.
"""

import sys
from pathlib import Path

import aie.iron as iron
import numpy as np
from aie.iron.device import from_name
from aie.utils.hostruntime.hostruntime import HostRuntimeError
from aie.utils.txn_trace import compare, trace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from whole_array import whole_array  # noqa: E402

iron.set_current_device(from_name("npu"))
TILE = dict(m=32, k=32, n=32, n_aie_cols=1)
BUFFERS = dict(
    A=np.ndarray[(512, 256), np.dtype[np.int16]],
    B=np.ndarray[(256, 256), np.dtype[np.int16]],
    C=np.ndarray[(512, 256), np.dtype[np.int32]],
)

dyn = whole_array.specialize(**BUFFERS, **TILE)


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
    static = whole_array.specialize(M=M, K=K, N=N, **BUFFERS, **TILE)
    diffs = compare(words, static.instructions(), names=("dynamic", "static"))
    print(
        f"M={M} K={K} N={N}: {len(pushes(words))} pushes, "
        f"matches static specialization={not diffs}"
    )
    for d in diffs:
        print(d)
# CHECK: M=256 K=128 N=128: {{[0-9]+}} pushes, matches static specialization=True
# CHECK: M=384 K=128 N=128: {{[0-9]+}} pushes, matches static specialization=True
# CHECK: M=128 K=128 N=128: {{[0-9]+}} pushes, matches static specialization=True
# CHECK: M=512 K=256 N=256: {{[0-9]+}} pushes, matches static specialization=True
# CHECK: M=256 K=64 N=32: {{[0-9]+}} pushes, matches static specialization=True
# CHECK: M=768 K=64 N=32: {{[0-9]+}} pushes, matches static specialization=True

for bad, why in (
    (dict(M=100, K=128, N=128), "M not a multiple of m*rows"),
    (dict(M=256, K=100, N=128), "K not a multiple of k"),
    (dict(M=512, K=256, N=512), "over the B buffer"),
):
    try:
        dyn.instructions(**bad)
        print(f"{why}: accepted")
    except HostRuntimeError as e:
        print(f"{why}: refused: {e}")
# The refusal carries the message of the guard that failed.
# CHECK: M not a multiple of m*rows: refused: {{.*}}: M must be a multiple of m * n_aie_rows
# CHECK: K not a multiple of k: refused: {{.*}}: K must be a multiple of k
# CHECK: over the B buffer: refused: {{.*}}: a runtime DMA access runs past the end of its 65536-element host buffer
