# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Differential test: Lean `Layout.slotElem` against each kernel contract's own `TensorLayout` codec."""

import argparse
import itertools
import subprocess

import aie.iron as iron
import aie.iron.kernels as kernels
import numpy as np
from aie.iron import str_to_dtype
from aie.iron.device import from_name

from difftest_walk import LEAN_EVAL

DTYPES = [
    ("i8", "i8"),
    ("i8", "i32"),
    ("i16", "i16"),
    ("i16", "i32"),
    ("bf16", "bf16"),
    ("bf16", "f32"),
]
TILES = [(64, 64, 32), (32, 32, 32), (16, 32, 16), (8, 8, 8), (64, 128, 64)]


def lean_line(layout) -> str:
    rows, cols = (1, *layout.shape) if len(layout.shape) == 1 else layout.shape
    fields = [rows, cols, int(layout.axes == (1, 0)), int(layout.inverse)]
    if layout.stream is not None:
        fields.append(int(layout.stream.offset))
        for size, stride in layout.stream.transformation_dims:
            fields += [int(size), int(stride)]
    return " ".join(map(str, fields))


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--devices", nargs="+", choices=["npu1", "npu2"], default=["npu1", "npu2"]
    )
    opts = p.parse_args()

    layouts, labels = [], []
    for device in opts.devices:
        iron.set_current_device(from_name(device, n_cols=None))
        for (din, dout), (m, k, n), bc, cc, emu in itertools.product(
            DTYPES, TILES, (0, 1), (0, 1), (0, 1)
        ):
            if emu and din != "bf16":
                continue
            try:
                fn = kernels.mm(
                    dim_m=m,
                    dim_k=k,
                    dim_n=n,
                    input_dtype=str_to_dtype(din),
                    output_dtype=str_to_dtype(dout),
                    b_col_maj=bool(bc),
                    c_col_maj=bool(cc),
                    emulate_bf16_mmul_with_bfp16=bool(emu),
                )
            except ValueError:
                continue
            r, s, t = fn.mac_dims
            if m % r or k % s or n % t:
                continue
            for i, layout in enumerate(fn.contract.layouts):
                layouts.append(layout)
                labels.append(
                    f"{device} {din}-{dout} {m}x{k}x{n} b{bc}c{cc}{' emu' if emu else ''} arg{i}"
                )
    out = subprocess.run(
        [LEAN_EVAL, "layouts"],
        input="\n".join(lean_line(l) for l in layouts) + "\n",
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    assert len(out) == len(
        layouts
    ), f"{len(out)} Lean results for {len(layouts)} layouts"
    bad = 0
    for layout, label, got in zip(layouts, labels, out):
        numel = int(np.prod(layout.shape))
        want = layout.encode(np.arange(numel))[0].tolist()
        if got.split() != [str(x) for x in want]:
            bad += 1
            print(f"MISMATCH {label}: lean {got[:60]} codec {want[:8]}")
    print(
        f"{len(layouts) - bad}/{len(layouts)} kernel operand layouts agree with their codecs"
    )
    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()
