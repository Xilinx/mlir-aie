# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %python %s | FileCheck %s
# REQUIRES: peano, hrxxclbinutil

"""Whole-array GEMM with DispatchTime M, K and N, checked without an NPU.

programming_examples/basic/matrix_multiplication/whole_array/whole_array_dyn.py
compiles once for a capacity and rebuilds its instruction stream per call from
the live shape. This test drives that host-side builder at several shapes and
checks, with aie.utils.txn_trace, that the DMA events it produces are the ones
a fully static specialization of the same generator produces, and that they
are the transfers the original static whole_array.py issues for that shape.
Dispatches that violate the shape constraints are refused by the guards.
"""

import sys
from pathlib import Path

import aie.iron as iron
import numpy as np
from aie.iron.device import NPU1Col1
from aie.utils.compile.jit._dispatch_bridge import DispatchBridge
from aie.utils.hostruntime.hostruntime import HostRuntimeError
from aie.utils.txn_trace import compare, trace

EXAMPLE = (
    Path(__file__).resolve().parents[2]
    / "programming_examples"
    / "basic"
    / "matrix_multiplication"
    / "whole_array"
)
sys.path.insert(0, str(EXAMPLE))
import whole_array as static_design  # noqa: E402
import whole_array_dyn as dynamic_design  # noqa: E402

iron.set_current_device(NPU1Col1())
TILE = dict(m=32, k=32, n=32, n_aie_cols=1, dtype_in_str="i16", dtype_out_str="i32")
CAPACITY = dict(M_max=512, K_max=256, N_max=256)

dyn = dynamic_design.whole_array_dyn.specialize(**CAPACITY, **TILE)
dyn.compile()
bridge = DispatchBridge(dyn.get_dispatch_lib_path(), dyn.compilable.dispatch_params)
print("dispatch params:", dyn.compilable.dispatch_params)
# CHECK: dispatch params: ['M', 'K', 'N']


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
    words = bridge.generate({"M": M, "K": K, "N": N})
    _, insts = dynamic_design.whole_array_dyn.specialize(
        M=M, K=K, N=N, **CAPACITY, **TILE
    ).compile()
    static = np.fromfile(insts, dtype=np.uint32)
    same = not compare(words, static)
    line = f"M={M} K={K} N={N}: {len(pushes(words))} pushes, matches static specialization={same}"
    # whole_array.py needs an even number of row blocks; where it can compile,
    # the dynamic design must produce exactly its DMA events, waits included.
    if (M // TILE["m"] // 4) % 2 == 0:
        _, orig = static_design.whole_array.specialize(M=M, K=K, N=N, **TILE).compile()
        original = np.fromfile(orig, dtype=np.uint32)
        line += f", same events as whole_array.py={not compare(words, original)}"
    print(line)
    for d in compare(words, static, names=("dynamic", "static")):
        print(d)
# CHECK: M=256 K=128 N=128: 5 pushes, matches static specialization=True, same events as whole_array.py=True
# CHECK: M=384 K=128 N=128: {{[0-9]+}} pushes, matches static specialization=True
# CHECK: M=128 K=128 N=128: {{[0-9]+}} pushes, matches static specialization=True
# CHECK: M=512 K=256 N=256: {{[0-9]+}} pushes, matches static specialization=True, same events as whole_array.py=True
# CHECK: M=256 K=64 N=32: {{[0-9]+}} pushes, matches static specialization=True, same events as whole_array.py=True
# CHECK: M=768 K=64 N=32: {{[0-9]+}} pushes, matches static specialization=True, same events as whole_array.py=True

for bad, why in (
    ({"M": 100, "K": 128, "N": 128}, "M not a multiple of m*rows"),
    ({"M": 256, "K": 100, "N": 128}, "K not a multiple of k"),
    ({"M": 512, "K": 256, "N": 512}, "over capacity"),
):
    try:
        bridge.generate(bad)
        print(f"{why}: accepted")
    except HostRuntimeError as e:
        print(f"{why}: refused: {e}")
# The refusal carries the message of the require() that failed.
# CHECK: M not a multiple of m*rows: refused: {{.*}}: M must be a multiple of m * n_aie_rows
# CHECK: K not a multiple of k: refused: {{.*}}: K must be a multiple of k
# CHECK: over capacity: refused: {{.*}}: B exceeds the compiled capacity
