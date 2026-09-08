# benchmarks/kernels/test_correctness.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Full correctness sweep of the registry: every case x data case x seed.

    pytest benchmarks/kernels/test_correctness.py -v            # everything
    pytest benchmarks/kernels/test_correctness.py -k "mm and bf16"
    pytest ... --data-cases random,nan_inf --seeds 3

This is the nightly's correctness gate, run on a device before any number is
recorded. The per-PR tier is ``test/python/npu/test_kernels_e2e.py`` (one
case per kernel); this file adds the shapes and the edge-case data. Both use
the same harness and the same kernel-owned tolerance, so a kernel that passes
there and fails here failed on data or shape, not on a different comparator.
"""

from __future__ import annotations

import numpy as np
import pytest
from aie.iron import kernels
from aie.iron.kernels._common import _detect_arch
from aie.utils import kernel_harness as kh

from . import registry

# --data-cases / --seeds are registered in conftest.py.


def _params(config):
    subset = config.getoption("--data-cases")
    subset = set(subset.split(",")) if subset else None
    seeds = config.getoption("--seeds")
    for case in registry.CASES:
        if not case.correctness:
            continue
        for dc in case.data_policy():
            if subset and dc not in subset:
                continue
            for seed in range(seeds if dc == "random" else 1):
                yield pytest.param(case, dc, seed, id=f"{case.name}/{dc}/s{seed}")


def pytest_generate_tests(metafunc):
    if {"case", "data_case", "seed"} <= set(metafunc.fixturenames):
        metafunc.parametrize("case,data_case,seed", list(_params(metafunc.config)))


def test_kernel(case, data_case, seed):
    if case.arch and _detect_arch() != case.arch:
        pytest.skip(f"{case.factory} binds a {case.arch}-only source")
    fn = case.fn()
    rng = np.random.default_rng(1000 + seed)
    inputs = registry.inputs_for(case, data_case, rng)
    design = kh.design(
        getattr(kernels, case.factory),
        **case.harness_opts(),
        params=kh.param_values(fn, inputs),
        **case.kwargs,
    )
    ref = kh.expected(fn, inputs, scalars=case.scalars)

    # The output is poisoned so a kernel that writes nothing cannot pass.
    out_n = kh.output_size(fn, calls=case.calls, shape=case.shape)
    out_dt = kh.output_dtype(fn, ref.dtype)
    got = kh.run(design, inputs, out_n, out_dt, poison=True, fn=fn)

    v = kh.judge(fn, got, ref, calls=case.calls)
    assert v, f"{case.name} [{data_case}]: {v.detail}"


def test_registry_sanity():
    names = [c.name for c in registry.CASES]
    assert len(names) == len(set(names)), "duplicate case names"
    assert registry.CANARY.name in names or registry.CANARY.perf
