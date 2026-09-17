# test_whole_array_matmul.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# REQUIRES: ryzen_ai
# RUN: %pytest %s

"""Whole-array matmul reconfigured via ctrlpkt, with control/data shim co-tenancy.

What: the real ``basic/matrix_multiplication/whole_array`` design (a 4xN_cols AIE
matmul) folded as a single-config ctrlpkt reconfiguration and run on device; the
result must equal ``A @ B``.

How: ``iron.Reconfiguration(method="ctrlpkt")`` folds the ``@iron.jit`` design and
its external matmul kernel into one full ELF, dispatched via ``pyxrt.runlist``.
Each column streams two shim inputs (A and B) into its memtile, claiming both shim
MM2S channels; aiecc's default-on auto-packetize packet-switches one leg per column
so the resident control overlay time-shares it, and the design-aware freeze pins
the control masters so each column's data routes around them.

Why: this is the full end-to-end demonstration -- a real, dense, multi-column
workload reconfigured on hardware -- and the only device exercise of the
auto-packetize + freeze co-tenancy that two data legs per column would otherwise
leave unroutable for control.
"""

import importlib.util
from pathlib import Path

import numpy as np

import aie.iron as iron
from aie.iron import str_to_dtype
from aie.iron.device import from_name
from aie.utils.compile.jit.compilabledesign import NPU_CACHE_HOME

from harness import dispatch_runlist, read_i32

# The real design lives under the basic examples (unrelated to this test dir); load
# it by path rather than duplicating ~600 lines of matmul dataflow.
_REPO = Path(__file__).resolve().parents[4]
_WHOLE_ARRAY = (
    _REPO
    / "programming_examples/basic/matrix_multiplication/whole_array/whole_array.py"
)

# Small pinned shape for a device test: 2 columns (4 rows fixed), micro-tile
# (m, k, n) = (8, 4, 16), M = K = N = 64, i16 -> i32.
_M = _K = _N = 64
_m, _k, _n = 8, 4, 16
_COLS = 2


def _load_whole_array():
    spec = importlib.util.spec_from_file_location("whole_array_design", _WHOLE_ARRAY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_whole_array_matmul_ctrlpkt():
    wa = _load_whole_array()

    # kernels.mm() reads its MMUL mac_dims from the current device (npu2 -> aie2p),
    # so set it before the fold generates the design's MLIR.
    iron.set_current_device(from_name("npu2", n_cols=None))

    rng = np.random.default_rng(1726250518)
    info = np.iinfo(np.int16)
    A_np = rng.integers(info.min // 4, info.max // 4, size=(_M, _K), dtype=np.int16)
    B_np = rng.integers(info.min // 4, info.max // 4, size=(_K, _N), dtype=np.int16)

    A = iron.tensor(A_np, dtype=str_to_dtype("i16"), device="npu")
    B = iron.tensor(B_np, dtype=str_to_dtype("i16"), device="npu")
    C = iron.zeros((_M, _N), dtype=str_to_dtype("i32"), device="npu")

    r = iron.Reconfiguration(
        "whole_array_ctrlpkt",
        method="ctrlpkt",
        output_dir=str(Path(NPU_CACHE_HOME) / "whole_array_ctrlpkt"),
    )
    r.add(
        wa.whole_array,
        A,
        B,
        C,
        M=_M,
        K=_K,
        N=_N,
        m=_m,
        k=_k,
        n=_n,
        n_aie_cols=_COLS,
        dtype_in_str="i16",
        dtype_out_str="i32",
    )
    elf = r.compile()

    per_ep = {ep: () if ep == elf.init else (A, B, C) for ep in elf.entrypoints}
    dispatch_runlist(elf, per_ep)

    expected = (A_np.astype(np.int64) @ B_np.astype(np.int64)).astype(np.int32)
    got = read_i32(C).reshape(_M, _N)
    np.testing.assert_array_equal(got, expected)
