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
so the resident control overlay time-shares it, and the design-aware pinning pins
the control masters so each column's data routes around them.

Why: this is the full end-to-end demonstration -- a real, dense, multi-column
workload reconfigured on hardware -- and the only device exercise of the
auto-packetize + pinning co-tenancy that two data legs per column would otherwise
leave unroutable for control.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

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


def _fold_whole_array(name, extra_aiecc_args=None):
    """Fold the real whole_array matmul at the pinned 2-column shape (i16->i32).
    Two shim inputs per column make the co-tenancy fire. Returns
    (elf, A_np, B_np, A, B, C)."""
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
        name,
        method="ctrlpkt",
        output_dir=str(Path(NPU_CACHE_HOME) / name),
        extra_aiecc_args=extra_aiecc_args,
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
    return r.compile(), A_np, B_np, A, B, C


def test_whole_array_matmul_ctrlpkt():
    elf, A_np, B_np, A, B, C = _fold_whole_array("whole_array_ctrlpkt")

    per_ep = {ep: () if ep == elf.init else (A, B, C) for ep in elf.entrypoints}
    dispatch_runlist(elf, per_ep)

    expected = (A_np.astype(np.int64) @ B_np.astype(np.int64)).astype(np.int32)
    got = read_i32(C).reshape(_M, _N)
    np.testing.assert_array_equal(got, expected)


def test_pinning_off_is_load_bearing():
    """Negative arm: --ctrlpkt-pinned-overlay=off skips the control-overlay pinning,
    so the same two-shim-input-per-column co-tenancy routes column data through the
    live control masters -- control packets never arrive and the dispatch fails
    (ERT_CMD_STATE_TIMEOUT). Proves the design-aware pinning (the default) is
    required, not cosmetic. The pinning is control routing, so the wedge is
    structural (not a timing race); the device recovers after the firmware
    command timeout, which is what runlist.wait() raises here."""
    elf, _, _, A, B, C = _fold_whole_array(
        "whole_array_pinoff", extra_aiecc_args=["--ctrlpkt-pinned-overlay=off"]
    )
    per_ep = {ep: () if ep == elf.init else (A, B, C) for ep in elf.entrypoints}
    with pytest.raises(Exception) as excinfo:
        dispatch_runlist(elf, per_ep)
    # The wedge must surface as a runtime/hardware error from runlist.wait()
    # (firmware command timeout), NOT a Python-level test bug -- otherwise this
    # arm could pass for an unintended reason (e.g. a harness typo). The exact
    # pyxrt exception type/message is driver-version dependent, so we don't pin
    # it, but a test-authoring error is never the wedge.
    assert not isinstance(
        excinfo.value,
        (AssertionError, TypeError, NameError, ImportError, AttributeError, KeyError),
    ), f"dispatch failed for a non-hardware reason: {excinfo.value!r}"
