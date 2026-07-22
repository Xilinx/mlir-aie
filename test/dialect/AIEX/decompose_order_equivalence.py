# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Differential regression check for aie-decompose-large-dma-bd (issue #2425).
#
# For each oversized, non-contiguous ND pattern we run aie-opt with the pass and
# expand BOTH the original op and the concatenation of the decomposed ops into
# the exact ORDERED element-address stream the hardware BD emits (d0 innermost /
# fastest ... d3 outermost / slowest). A decomposition that preserved only the
# address *set* but not the emission *order* would be caught here.
#
# RUN: %python %s | FileCheck %s
# CHECK: ALL ORDER-EQUIVALENT: True

import re
import shutil
import subprocess
import sys

AIE_OPT = shutil.which("aie-opt") or "aie-opt"

OP_RE = re.compile(r"dma_memcpy_nd\(%\w+\[([-\d, ]+)\]\[([-\d, ]+)\]\[([-\d, ]+)\]\)")


def emit(offset, sizes, strides):
    """Ordered element addresses for one ND access pattern (outermost-first)."""
    s3, s2, s1, s0 = sizes
    t3, t2, t1, t0 = strides
    out = []
    for i3 in range(s3):
        b3 = offset + i3 * t3
        for i2 in range(s2):
            b2 = b3 + i2 * t2
            for i1 in range(s1):
                b1 = b2 + i1 * t1
                for i0 in range(s0):
                    out.append(b1 + i0 * t0)
    return out


def module(rows, cols, pitch, nelem):
    return f"""module {{
  aie.device(npu2_1col) {{
    aie.runtime_sequence(%in : memref<{nelem}xi32>) {{
      aiex.npu.dma_memcpy_nd (%in[0,0,0,0][1,1,{rows},{cols}][0,0,{pitch},1])
        {{ metadata = @a, id = 0 : i64 }} : memref<{nelem}xi32>
    }}
    %t = aie.tile(0, 0)
    aie.shim_dma_allocation @a (%t, MM2S, 0)
  }}
}}
"""


def run_case(rows, cols, pitch):
    nelem = (rows - 1) * pitch + cols
    src = module(rows, cols, pitch, nelem)
    r = subprocess.run(
        [AIE_OPT, "--pass-pipeline=builtin.module(aie.device(aie-decompose-large-dma-bd))"],
        input=src, capture_output=True, text=True,
    )
    if r.returncode != 0:
        return None
    ops = []
    for m in OP_RE.finditer(r.stdout):
        offs = [int(x) for x in m.group(1).split(",")]
        szs = [int(x) for x in m.group(2).split(",")]
        strs = [int(x) for x in m.group(3).split(",")]
        flat = sum(o * s for o, s in zip(offs, strs))
        ops.append((flat, szs, strs))
    if not ops:
        return None
    orig = emit(0, [1, 1, rows, cols], [0, 0, pitch, 1])
    dec = []
    for flat, szs, strs in ops:
        dec += emit(flat, szs, strs)
    return orig == dec  # exact order equality


CASES = [
    (1080, 1920, 1921),  # the #2425 repro (factoring)
    (4, 2000, 2100),
    (64, 4096, 4097),
    (3, 1500, 2000),
    (2, 3000, 3001),     # large inner (factored)
    (1031, 2, 3),        # prime outer -> slicing / chain
    (7, 1024, 1100),
    (16, 2048, 2049),
]

all_ok = True
for rows, cols, pitch in CASES:
    ok = run_case(rows, cols, pitch)
    all_ok = all_ok and (ok is True)
    print(f"rows={rows} cols={cols} pitch={pitch} -> order_equivalent={ok}")

print(f"ALL ORDER-EQUIVALENT: {all_ok}")
sys.exit(0 if all_ok else 1)
