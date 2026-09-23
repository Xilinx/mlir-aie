# test_kernels_bench.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

"""Time the kernel library on device and emit benchmark-action rows.

Every case in ``kernel_cases.py`` marked ``perf`` is one test: it checks the
kernel against its contract, then times it. Checking first is the point --
timings from a kernel that returns the wrong answer are noise, so the
assertion runs before anything is recorded, and a failing test leaves the
session non-zero, which is what stops ``pytest_sessionfinish`` writing the
JSON at all.

Run it the way the nightly workflow does::

    pytest test/python/npu/test_kernels_bench.py -m benchmark
        --bench-out bench.json --bench-meta meta.json --pmode turbo
        --warmup 10 --iters 50

``-k`` selects a subset. The series a row lands in is ``<case>/<metric>``;
``test_benchmark_series_names.py`` pins those names, because renaming one
restarts its chart on gh-pages.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
from aie.iron import kernels
from aie.iron.algorithms import kernel_design as kd
from aie.utils.benchmark import preflight, provenance, run_iters
from cases import Case, inputs_for
from kernel_cases import CASES

TRACE_SIZE = 16384

# A kernel whose cost is known well enough to catch a broken measurement.
# 270 cycles on Strix, identical across calls and across runs: 8 KB copied at
# about 30 B/cycle. The lower bound catches a decode that reports nothing
# rather than bounding the kernel, and the upper bound stays generous until
# nightly data has shown the spread across machines.
SMOKE_TEST = Case("passthrough", dict(tile_size=2048), calls=16)
SMOKE_CYCLE_BAND = (100, 2_000_000)


def _param(case: Case):
    """One case as a pytest param, keeping its name as the test id.

    ``devices`` becomes the ``supported_devices`` marker ``conftest.py``
    already applies, so a kernel whose source exists only for one generation
    skips on the other instead of being filtered by hand.
    """
    marks = [pytest.mark.supported_devices(*case.devices)] if case.devices else []
    return pytest.param(case, id=case.name, marks=marks)


_PERF_CASES = [_param(c) for c in CASES if c.perf and c.name != SMOKE_TEST.name]


def _measure(case: Case, config, workdir: Path) -> dict:
    """Build, check and time one case. Raises if it is wrong."""
    fn = case.fn()
    factory = getattr(kernels, case.factory)
    inputs = inputs_for(case, "random", np.random.default_rng(0))
    design = kd.design(
        factory,
        **case.harness_opts(),
        params=fn.param_values(inputs),
        aiecc_flags=["--get-core-elfs"],
        **case.kwargs,
    )

    ref = fn.expected(inputs, scalars=case.scalars)
    out_n = kd.output_size(fn, calls=case.calls)
    out_dt = fn.output_dtype()
    ins, out = kd.upload(inputs, out_n, out_dt, poison=True, fn=fn)
    outputs = out if isinstance(out, tuple) else (out,)

    measured: dict = {}
    if not config.getoption("--no-compile"):
        # A subdirectory per case: the cold rebuild bypasses the on-disk
        # xclbin cache, but compile_external_kernel still skips a kernel
        # object that already exists in its directory, so a shared directory
        # would let one case's leftovers make another's "cold" build look
        # faster than it is.
        measured["compile"] = design.measure_compile(workdir / case.name)

    design(*ins, *outputs)
    got = tuple(o.numpy() for o in outputs)
    verdict = fn.judge(got if len(got) > 1 else got[0], ref, calls=case.calls)
    assert verdict, f"{case.name}: {verdict.detail}"

    measured["wall"] = run_iters(
        design,
        *ins,
        *outputs,
        warmup=config.getoption("--warmup"),
        iters=config.getoption("--iters"),
    )
    if not config.getoption("--no-cycles"):
        # A separate traced run: tracing perturbs the timing above.
        per_call = kd.cycles_per_call(
            design,
            inputs,
            out_n,
            out_dt,
            trace_size=TRACE_SIZE,
            workdir=workdir,
            fn=fn,
            calls=case.calls,
        )
        if per_call:
            measured["cycles"] = int(np.median(per_call))
    return measured


def _record(record, case: Case, m: dict) -> None:
    """Emit the rows one measurement contributes, in series-name order."""
    if (cycles := m.get("cycles")) is not None:
        record(case.name, "cycles", "cycles", cycles)
        if ops := case.work():
            record(
                case.name,
                "cycles_per_kop",
                "cycles/1k-ops",
                round(1000.0 * cycles * case.kernel_calls() / ops, 3),
            )
    wall = m.get("wall")
    if wall and wall.npu:
        s = wall.npu
        record(
            case.name,
            "npu_us",
            "us",
            round(s.median_us, 2),
            f"min {s.min_us:.1f} max {s.max_us:.1f} n={s.n}",
        )
    if wall:
        s = wall.e2e
        record(
            case.name,
            "e2e_us",
            "us",
            round(s.median_us, 2),
            f"min {s.min_us:.1f} max {s.max_us:.1f} n={s.n}",
        )
    if compiled := m.get("compile"):
        seconds, xclbin, insts, elf = compiled
        record(case.name, "compile_s", "s", round(seconds, 2))
        record(case.name, "xclbin_bytes", "bytes", xclbin)
        record(case.name, "insts_bytes", "bytes", insts)
        record(case.name, "core_elf_bytes", "bytes", elf)


@pytest.fixture(scope="module")
def workdir():
    with tempfile.TemporaryDirectory(prefix="aie-bench-") as d:
        yield Path(d)


@pytest.fixture(scope="module", autouse=True)
def _preflight(request):
    """Describe the device, and refuse to measure it in the wrong power mode.

    A number taken at the wrong clock is worse than no number, because it
    lands in the same series as the right ones. Failing here fails every test
    in the module, which is what keeps the JSON unwritten.
    """
    config = request.config
    pre = preflight()
    required = config.getoption("--pmode")
    if required != "any" and pre.pmode != required:
        pytest.fail(
            f"power mode is {pre.pmode or 'unreadable'}, required '{required}' "
            "(set it with xrt-smi configure --pmode, or pass --pmode any)"
        )
    config._bench_meta["preflight"] = dict(vars(pre))
    config._bench_meta["provenance"] = provenance(device=pre.device, pmode=pre.pmode)
    return pre


@pytest.mark.benchmark
def test_measurement_is_sane(request, benchmark, workdir):
    """Time a kernel whose cost is known, so a broken clock fails loudly.

    Without this, a tracing path that decodes nothing reports ``None`` cycles
    for every kernel and the whole run charts as an improvement.
    """
    m = _measure(SMOKE_TEST, request.config, workdir)
    cycles = m.get("cycles")
    if not request.config.getoption("--no-cycles"):
        lo, hi = SMOKE_CYCLE_BAND
        assert cycles is not None, "traced run produced no cycle count"
        assert lo <= cycles <= hi, f"{cycles} cycles is outside {SMOKE_CYCLE_BAND}"
    _record(benchmark, SMOKE_TEST, m)


@pytest.mark.benchmark
@pytest.mark.parametrize("case", _PERF_CASES)
def test_kernel_benchmark(case, request, benchmark, workdir):
    _record(benchmark, case, _measure(case, request.config, workdir))
