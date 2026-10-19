# test_scalar_exp.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# RUN: %pytest %s
"""aie_kernels/common/scalar_exp.h, built for the host, against numpy's exp."""

import ctypes
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from aie.utils import config

HARNESS = """
#include "{header}"
extern "C" void exps(const float *x, float *y, long n) {{
  for (long i = 0; i < n; i++)
    y[i] = scalar_exp(x[i]);
}}
"""


@pytest.fixture(scope="module")
def scalar_exp(tmp_path_factory):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("no host C++ compiler")
    build = tmp_path_factory.mktemp("scalar_exp")
    header = Path(config.aie_kernels_dir()) / "common" / "scalar_exp.h"
    (build / "harness.cc").write_text(HARNESS.format(header=header))
    subprocess.run(
        [
            compiler,
            "-O2",
            "-std=c++20",
            "-ffp-contract=off",
            "-fno-fast-math",
            "-shared",
            "-fPIC",
            str(build / "harness.cc"),
            "-o",
            str(build / "harness.so"),
        ],
        check=True,
    )
    exps = ctypes.CDLL(str(build / "harness.so")).exps
    exps.argtypes = [
        np.ctypeslib.ndpointer(np.float32, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(np.float32, flags="C_CONTIGUOUS"),
        ctypes.c_long,
    ]

    def run(x):
        x = np.ascontiguousarray(x, np.float32)
        y = np.empty_like(x)
        exps(x, y, len(x))
        return y

    return run


def floats_between(lo, hi):
    """Every float32 from ``lo`` to ``hi``."""
    lo, hi = np.float32(lo).view(np.int32), np.float32(hi).view(np.int32)
    bits = np.arange(min(lo, hi), max(lo, hi) + 1, dtype=np.int32)
    return bits.view(np.float32)


def test_every_float_near_both_cutoffs(scalar_exp):
    x = np.concatenate(
        [
            floats_between(-87.5, -87.2),
            floats_between(88.6, 88.8),
            np.linspace(-100, 100, 20001, dtype=np.float32),
        ]
    )
    got = scalar_exp(x)
    exact = np.exp(x.astype(np.float64))
    with np.errstate(over="ignore"):
        rounded = exact.astype(np.float32)
    normal = (exact >= np.finfo(np.float32).tiny) & np.isfinite(rounded)
    assert np.all(got[~normal & (exact < 1)] == 0)
    assert np.all(np.isposinf(got[~normal & (exact > 1)]))
    ulps = np.abs(got[normal].astype(np.float64) - exact[normal]) / np.spacing(
        rounded[normal]
    )
    assert ulps.max() <= 2.0


def test_special_values(scalar_exp):
    got = scalar_exp(np.array([np.nan, np.inf, -np.inf, 0.0], np.float32))
    assert np.isnan(got[0]) and np.isposinf(got[1])
    assert got[2] == 0 and got[3] == 1
