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
# The task path is checked the same way, for patterns of up to 6 dimensions.
# There one execution of a descriptor walks its innermost three dimensions, at
# its iteration dimension's index, and each descriptor counts its own
# executions. The decomposed tasks are executed in the order they are started.
#
# RUN: %python %s | FileCheck %s
# CHECK: ALL ORDER-EQUIVALENT: True

import math
import re
import shutil
import subprocess
import sys

AIE_OPT = shutil.which("aie-opt") or "aie-opt"

OP_RE = re.compile(r"dma_memcpy_nd\(%\w+\[([-\d, ]+)\]\[([-\d, ]+)\]\[([-\d, ]+)\]\)")
CONFIGURE_RE = re.compile(r"(%\w+) = aiex\.dma_configure_task_for @a \{")
BD_RE = re.compile(
    r"aie\.dma_bd\(%\w+ : memref<\w+> offset = (\d+) len = (\d+)"
    r"(?: sizes = \[([\d, ]+)\] strides = \[([\d, ]+)\])?"
)
START_RE = re.compile(r"aiex\.dma_start_task\((%\w+)\)(?: \{(.*)\})?")
REPEAT_RE = re.compile(r"repeat_count = (\d+)")


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


def module(offsets, sizes, strides, nelem, dtype):
    pattern = "".join(
        "[" + ",".join(map(str, values)) + "]" for values in (offsets, sizes, strides)
    )
    return f"""module {{
  aie.device(npu2_1col) {{
    aie.runtime_sequence(%in : memref<{nelem}x{dtype}>) {{
      aiex.npu.dma_memcpy_nd (%in{pattern})
        {{ metadata = @a, id = 0 : i64 }} : memref<{nelem}x{dtype}>
    }}
    %t = aie.tile(0, 0)
    aie.shim_dma_allocation @a (%t, MM2S, 0)
  }}
}}
"""


def run_case(offsets, sizes, strides, dtype="i32"):
    offset = sum(o * s for o, s in zip(offsets, strides))
    nelem = offset + sum((n - 1) * s for n, s in zip(sizes, strides)) + 1
    src = module(offsets, sizes, strides, nelem, dtype)
    r = subprocess.run(
        [
            AIE_OPT,
            "--pass-pipeline=builtin.module(aie.device(aie-decompose-large-dma-bd))",
        ],
        input=src,
        capture_output=True,
        text=True,
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
    orig = emit(offset, sizes, strides)
    dec = []
    for flat, szs, strs in ops:
        dec += emit(flat, szs, strs)
    return orig == dec  # exact order equality


CASES = [
    (1080, 1920, 1921),  # the #2425 repro (factoring)
    (4, 2000, 2100),
    (64, 4096, 4097),
    (3, 1500, 2000),
    (2, 3000, 3001),  # large inner (factored)
    (1031, 2, 3),  # prime outer -> slicing / chain
    (4093, 2, 3),  # prime outer -> four full chunks and a singleton tail
    (7, 1024, 1100),
    (16, 2048, 2049),
]

all_ok = True
for rows, cols, pitch in CASES:
    ok = run_case([0, 0, 0, 0], [1, 1, rows, cols], [0, 0, pitch, 1])
    all_ok = all_ok and (ok is True)
    print(f"rows={rows} cols={cols} pitch={pitch} -> order_equivalent={ok}")

for offsets, sizes, strides in [
    ([1, 0, 2, 3], [2, 1, 2, 2], [64, 0, 2097152, 1]),
    ([1, 0, 2, 3], [2, 1, 2, 4], [64, 0, 2097152, 1]),
    ([1, 0, 2, 3], [2, 1, 2, 1024], [2097152, 0, 2048, 1]),
    ([1, 2, 3, 4], [2, 2, 2, 2], [64, 32, 2097152, 1]),
    ([1, 0, 2, 3], [2, 1, 2, 2], [2097152, 0, 4194304, 1]),
    ([0, 0, 0, 0], [65, 1, 1, 2], [3, 0, 0, 2]),
    ([0, 0, 0, 0], [2, 1, 4099, 2], [20000, 0, 3, 1]),
]:
    ok = run_case(offsets, sizes, strides)
    all_ok = all_ok and (ok is True)
    print(f"offsets={offsets} sizes={sizes} strides={strides} -> order_equivalent={ok}")

for offsets, sizes, strides in [
    ([0, 0, 0, 0], [4, 1, 64, 512], [4194304, 0, 8192, 1]),
    ([1, 0, 2, 4], [2, 1, 2, 4], [64, 0, 4194304, 1]),
]:
    ok = run_case(offsets, sizes, strides, dtype="bf16")
    all_ok = all_ok and (ok is True)
    print(
        f"bf16 offsets={offsets} sizes={sizes} strides={strides} -> order_equivalent={ok}"
    )


def task_module(offset, sizes, strides, repeat_count, nelem, dtype):
    length = math.prod(sizes[-3:])
    pattern = f"sizes = {list(sizes)} strides = {list(strides)}"
    return f"""module {{
  aie.device(npu2_1col) {{
    %t = aie.tile(0, 0)
    aie.shim_dma_allocation @a (%t, MM2S, 0)
    aie.runtime_sequence(%in : memref<{nelem}x{dtype}>) {{
      %0 = aiex.dma_configure_task_for @a {{
        aie.dma_bd(%in : memref<{nelem}x{dtype}> offset = {offset} len = {length} {pattern})
        aie.end
      }} {{issue_token = true, repeat_count = {repeat_count} : i32}}
      aiex.dma_start_task(%0)
      aiex.dma_await_task(%0)
    }}
  }}
}}
"""


def execute(offset, sizes, strides, execution):
    """Addresses of one execution of a descriptor: the dimensions past the
    innermost three are indexed by the execution count, outermost slowest."""
    index = execution % math.prod(sizes[:-3])
    for size, stride in zip(reversed(sizes[:-3]), reversed(strides[:-3])):
        offset += (index % size) * stride
        index //= size
    return emit(offset, [1] + list(sizes[-3:]), [0] + list(strides[-3:]))


def parse_tasks(text):
    """The configured tasks, as {name: (bds, repeat_count)}, and the starts, as
    (name, repeat_count), in program order."""
    tasks, starts = {}, []
    lines = iter(text.splitlines())
    for line in lines:
        configure = CONFIGURE_RE.search(line)
        if configure:
            bds = []
            for body in lines:
                bd = BD_RE.search(body)
                if bd:
                    offset, length = int(bd.group(1)), int(bd.group(2))
                    if bd.group(3):
                        sizes = [int(x) for x in bd.group(3).split(",")]
                        strides = [int(x) for x in bd.group(4).split(",")]
                    else:
                        sizes, strides = [1, 1, 1, length], [0, 0, 0, 1]
                    bds.append((offset, sizes, strides))
                elif body.strip().startswith("}"):
                    repeat = REPEAT_RE.search(body)
                    tasks[configure.group(1)] = (
                        bds,
                        int(repeat.group(1)) if repeat else 0,
                    )
                    break
            continue
        start = START_RE.search(line)
        if start:
            repeat = REPEAT_RE.search(start.group(2) or "")
            starts.append((start.group(1), int(repeat.group(1)) if repeat else None))
    return tasks, starts


def run_task_case(offset, sizes, strides, repeat_count, dtype="i32"):
    nelem = offset + sum((n - 1) * s for n, s in zip(sizes, strides)) + 1
    src = task_module(offset, sizes, strides, repeat_count, nelem, dtype)
    r = subprocess.run(
        [
            AIE_OPT,
            "--pass-pipeline=builtin.module(aie.device(aie-decompose-large-dma-bd))",
        ],
        input=src,
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        return None
    tasks, starts = parse_tasks(r.stdout)
    if not starts:
        return None
    # Every descriptor has 4 dimensions after the pass.
    if any(len(bd[1]) != 4 for bds, _ in tasks.values() for bd in bds):
        return False
    orig = []
    for execution in range(repeat_count + 1):
        orig += execute(offset, sizes, strides, execution)
    dec = []
    executions = {}
    for name, override in starts:
        bds, task_repeat = tasks[name]
        for _ in range((task_repeat if override is None else override) + 1):
            for i, (bd_offset, bd_sizes, bd_strides) in enumerate(bds):
                count = executions.get((name, i), 0)
                dec += execute(bd_offset, bd_sizes, bd_strides, count)
                executions[(name, i)] = count + 1
    return orig == dec  # exact order equality


for offset, sizes, strides, repeat_count, dtype in [
    # 4 dimensions: factored, sliced into a chain, and sliced into tasks.
    (0, [1, 1, 1080, 1920], [0, 0, 1921, 1], 0, "i32"),
    (0, [4, 1, 1031, 2], [20000, 0, 3, 1], 3, "i32"),
    (0, [4, 1, 1031, 2], [20000, 0, 3, 1], 7, "i32"),
    (0, [4, 1, 4099, 2], [20000, 0, 3, 1], 3, "i32"),
    (0, [2, 1, 4099, 2], [20000, 0, 3, 1], 5, "i32"),
    # 5 dimensions: peeled, merged, a unit one dropped, and two passes.
    (0, [3, 4, 2, 32, 32], [262144, 32, 4096, 1024, 1], 11, "bf16"),
    (0, [2, 3, 1, 8, 16], [300, 100, 0, 32, 1], 5, "i32"),
    (0, [1, 3, 1, 8, 16], [0, 200, 0, 32, 1], 2, "i32"),
    (0, [2, 2, 1, 8, 16], [9000, 3500, 0, 32, 1], 7, "i32"),
    # Pieces that are decomposed further: sliced per index, and factored.
    (0, [2, 2, 1, 1031, 2], [9000, 3500, 0, 3, 1], 3, "i32"),
    (0, [2, 2, 1, 1031, 2], [9000, 3500, 0, 3, 1], 7, "i32"),
    (5, [3, 2, 1, 64, 2000], [300000, 140000, 0, 2100, 1], 5, "i32"),
    # 6 dimensions, from a nonzero base, and with a merge among them.
    (16, [3, 2, 2, 1, 8, 16], [20000, 7000, 1000, 0, 32, 1], 11, "i32"),
    (0, [2, 2, 3, 1, 8, 16], [9000, 900, 300, 0, 32, 1], 11, "i32"),
]:
    ok = run_task_case(offset, sizes, strides, repeat_count, dtype)
    all_ok = all_ok and (ok is True)
    print(
        f"task offset={offset} sizes={sizes} strides={strides} "
        f"repeat_count={repeat_count} -> order_equivalent={ok}"
    )

print(f"ALL ORDER-EQUIVALENT: {all_ok}")
sys.exit(0 if all_ok else 1)
