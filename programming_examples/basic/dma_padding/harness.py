# dma_padding/harness.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Shared infra for the dma_padding designs.

Each interface design (ObjectFifo, TileDma) lives in its own file and defines
only the parts that differ. Everything common -- the padding geometry, the three
pad cases, and the run/verify/CLI sweep -- lives here so those files isolate the
actual API differences.

A design is supplied as a *factory* ``fn(elem_dtype) -> @iron.jit design`` whose
design takes ``(a_in, c_out, *, pad_value: CompileTime[int])``. ``main`` sweeps
every (api, pad-case) pair and verifies on device; ``--api`` / ``--pad`` filter
the sweep.
"""

import argparse

import aie.iron as iron
import numpy as np
from aie.utils.verify import assert_pass

REAL = 8  # real elements per transfer
PAD_BEFORE = 4
PAD_AFTER = 4
REGION = PAD_BEFORE + REAL + PAD_AFTER

# pad-case flag -> (element dtype, per-element pad value)
PAD_CASES = {
    "zero": (np.int32, 0),  # hardware default: no register write
    "int32": (np.int32, 1000),  # full 32-bit value held in the register
    "int8": (np.int8, 8),  # sub-word value replicated across the word (0x08080808)
}


def _run_case(api, factory, pad_key):
    elem_dtype, pad_value = PAD_CASES[pad_key]
    a_t = iron.arange(REAL, dtype=elem_dtype, device="npu")
    c_t = iron.zeros(REGION, dtype=elem_dtype, device="npu")

    factory(elem_dtype)(a_t, c_t, pad_value=pad_value)

    expected = np.array(
        [pad_value] * PAD_BEFORE + list(range(REAL)) + [pad_value] * PAD_AFTER,
        dtype=elem_dtype,
    )
    assert_pass(c_t.numpy(), expected, fail_msg=f"{api}/{pad_key}", print_pass=False)
    print(f"{api:>11} {pad_key:>5}: {c_t.numpy().tolist()}")


def main(apis):
    """Sweep (api, pad-case) and verify. ``apis`` maps name -> design factory.

    ``--api NAME`` and/or ``--pad {zero,int32,int8}`` narrow the sweep.
    """
    p = argparse.ArgumentParser()
    p.add_argument("--api", choices=list(apis), help="only this entrypoint")
    p.add_argument("--pad", choices=list(PAD_CASES), help="only this pad case")
    opts = p.parse_args()

    # Bind whatever NPU the runtime detects so the designs can lower and
    # device="npu" tensors resolve.
    iron.ensure_current_device()

    for api in [opts.api] if opts.api else list(apis):
        for pad_key in [opts.pad] if opts.pad else list(PAD_CASES):
            _run_case(api, apis[api], pad_key)
