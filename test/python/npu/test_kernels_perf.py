# test_kernels_perf.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

"""Time the kernel library on device and emit benchmark-action rows.

Every case in ``kernel_cases.py`` marked ``perf`` is one test: it checks the
kernel against its contract, then times it. Checking first is the point --
timings from a kernel that returns the wrong answer are noise, so the
assertion runs before anything is recorded. A failing kernel drops only its
own rows; the JSON is still written, unless the device itself is suspect --
preflight or ``test_measurement_is_sane`` did not pass -- in which case
``pytest_sessionfinish`` writes none of it.
With ``--correctness-results correctness.xml``, publication also excludes
cases with any extensive-suite failure or no passing correctness test.

Run it the way the nightly workflow does::

    pytest test/python/npu/test_kernels_perf.py -m perf
        --perf-out perf.json --perf-meta meta.json --pmode turbo
        --warmup 10 --iters 50

``-k`` selects a subset; include the sanity test, for example
``-k '(softmax) or test_measurement_is_sane'``, to allow publication.
The series a row lands in is ``<case>/<metric>``;
``test_perf_series_names.py`` pins those names, because renaming one
restarts its chart on gh-pages.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
from aie.iron import ExternalFunction, kernels
from aie.iron.algorithms import kernel_design as kd
from aie.utils.benchmark import preflight, provenance, run_iters
from cases import Case, inputs_for
from kernel_cases import CASES

TRACE_SIZE = 16384
# The add/256 case filled 16 KB after 91 intervals (180 B each); size for
# every declared interval with headroom, so the split sees whole calls.
TRACE_BYTES_PER_INTERVAL = 512
# A long kernel costs more trace bytes per interval than that (swiglu/256
# filled 128 KB after 84 calls), so a filled buffer is regrown from what it
# held and the traced run repeated, this many times at most.
TRACE_RETRIES = 3

# A kernel whose cost is known well enough to catch a broken measurement:
# 8 KB copied, identical across its calls and across nightlies (264 cycles
# on npu1, 138 on npu2, min = median = max over 256 calls). A band of about
# 25% either side catches a decode that reports nothing, a clock or trace
# path that reports something else, and a device that is not what the
# runner label promised, while leaving room for a host of the same
# generation to differ a little.
SMOKE_TEST = Case("passthrough", dict(tile_size=2048), calls=16)
SMOKE_CYCLE_BANDS = {"npu1": (200, 330), "npu2": (105, 175)}


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

    design(*ins, *outputs)
    # Copies: numpy() views the device buffer, which dies with this frame.
    got = tuple(o.numpy().copy() for o in outputs)
    verdict = fn.judge(
        got if len(got) > 1 else got[0],
        ref,
        calls=case.calls,
        inputs=inputs,
        scalars=case.scalars,
    )
    assert verdict, f"{case.name}: {verdict.detail}"
    measured: dict = {"outputs": got, "sizes": _sizes(design)}
    measured["wall"] = run_iters(
        design,
        *ins,
        *outputs,
        warmup=config.getoption("--warmup"),
        iters=config.getoption("--iters"),
    )
    if not config.getoption("--no-cycles"):
        # A separate traced run: tracing perturbs the timing above.
        intervals = kd.traced_intervals(fn, calls=case.calls)
        trace_size = max(TRACE_SIZE, TRACE_BYTES_PER_INTERVAL * intervals)
        for _ in range(1 + TRACE_RETRIES):
            traced = kd.cycles_per_call(
                design,
                inputs,
                out_n,
                out_dt,
                trace_size=trace_size,
                workdir=workdir,
                fn=fn,
                calls=case.calls,
            )
            # Every kernel checked here is timed; one that is not would chart
            # only wall clock and still pass.
            assert not traced.untimed, f"{case.name}: untimed, {traced.untimed}"
            if not traced.truncated:
                break
            # The buffer filled: the min is over a prefix of the calls. Ask
            # for every interval at the cost this run measured.
            seen = (
                len(traced.setup)
                + len(traced.kernel)
                + sum(len(v) for v in traced.initializers.values())
            )
            trace_size = kd.grow_trace_size(trace_size, seen=seen, expected=intervals)
        measured["cycles"] = traced
        measured["trace_size"] = trace_size
    return measured


def _sizes(design) -> tuple[int, int, int]:
    """The xclbin, instruction and core ELF bytes of the build ``design`` ran."""
    entry = design.compilable.get_cache_entry()
    elfs = sum(p.stat().st_size for p in entry.directory.glob("elfs_*/*.elf"))
    return entry.xclbin.stat().st_size, entry.insts.stat().st_size, elfs


def _cycles_span(traced: kd.CallCycles) -> str:
    """The kernel's spread, and any initializer's, beside its min."""
    k = traced.kernel
    parts = [f"median {int(np.median(k))} max {max(k)} n={len(k)}"]
    parts += [f"init[{i}] min {min(v)}" for i, v in traced.initializers.items() if v]
    if traced.truncated:
        parts.append("truncated")
    return "; ".join(parts)


def _detail(case: Case, m: dict) -> dict:
    """Return the distribution behind each row, for ``--perf-meta``.

    benchmark-action keeps one value and a text ``range`` per row; the
    numbers a reader needs to judge that value (how far the median and the
    slowest call sit above the min, how many calls the trace held) go here
    as fields.
    """
    detail: dict = {}
    if (traced := m.get("cycles")) is not None:
        k = traced.kernel
        detail["cycles"] = {
            "min": min(k),
            "median": int(np.median(k)),
            "max": max(k),
            "n": len(k),
            "calls": case.kernel_calls(),
            "truncated": traced.truncated,
            "trace_size": m.get("trace_size"),
        }
    if (wall := m.get("wall")) and (s := wall.npu):
        detail["npu_us"] = {
            "median": round(s.median_us, 2),
            "mad": round(s.mad_us, 2),
            "p95": round(s.p95_us, 2),
            "min": round(s.min_us, 2),
            "max": round(s.max_us, 2),
            "n": s.n,
        }
    return detail


def _record(record, case: Case, m: dict, meta: dict | None = None) -> None:
    """Emit the rows one measurement contributes, in series-name order.

    With ``meta`` (the ``--perf-meta`` dict), also file the case's
    :func:`_detail` under ``meta["cases"]``.
    """
    if meta is not None:
        meta.setdefault("cases", {})[case.name] = _detail(case, m)
    if (traced := m.get("cycles")) is not None:
        # The min: every call does the same work, so anything above it is
        # the core waiting (a stall, a refresh), not the kernel.
        cycles = min(traced.kernel)
        record(case.name, "cycles", "cycles", cycles, _cycles_span(traced))
        if ops := case.work():
            record(
                case.name,
                "cycles_per_kop",
                "cycles/1k-ops",
                round(1000.0 * cycles * case.kernel_calls() / ops, 3),
            )
    if (wall := m.get("wall")) and (s := wall.npu):
        record(
            case.name,
            "npu_us",
            "us",
            round(s.median_us, 2),
            f"± {s.mad_us:.1f}; min {s.min_us:.1f} max {s.max_us:.1f} n={s.n}",
        )
    if sizes := m.get("sizes"):
        xclbin, insts, elf = sizes
        record(case.name, "xclbin_bytes", "bytes", xclbin)
        record(case.name, "insts_bytes", "bytes", insts)
        record(case.name, "core_elf_bytes", "bytes", elf)


@pytest.fixture(scope="module")
def workdir():
    with tempfile.TemporaryDirectory(prefix="aie-perf-") as d:
        yield Path(d)


@pytest.fixture(scope="module", autouse=True)
def _preflight(request):
    """Describe the device, and refuse to measure it in the wrong power mode.

    A number taken at the wrong clock is worse than no number, because it
    lands in the same series as the right ones. Failing here fails every test
    in the module and leaves ``preflight`` out of the meta, which keeps the
    JSON unwritten.
    """
    config = request.config
    pre = preflight()
    required = config.getoption("--pmode")
    if required != "any" and pre.pmode != required:
        pytest.fail(
            f"power mode is {pre.pmode or 'unreadable'}, required '{required}' "
            "(set it with xrt-smi configure --pmode, or pass --pmode any)"
        )
    config._perf_meta["preflight"] = dict(vars(pre))
    config._perf_meta["provenance"] = provenance(device=pre.device, pmode=pre.pmode)
    return pre


@pytest.mark.perf
def test_measurement_is_sane(request, record_perf, workdir):
    """Time a kernel whose cost is known, so a broken clock fails loudly.

    Without this, a tracing path that decodes nothing reports ``None`` cycles
    for every kernel and the whole run charts as an improvement. Failing here
    withholds the whole JSON, not just this row.
    """
    meta = request.config._perf_meta
    meta["measurement_sane"] = False
    m = _measure(SMOKE_TEST, request.config, workdir)
    cycles = min(m["cycles"].kernel) if "cycles" in m else None
    if not request.config.getoption("--no-cycles"):
        band = SMOKE_CYCLE_BANDS[meta["preflight"]["npu"]]
        lo, hi = band
        assert cycles is not None, "traced run produced no cycle count"
        assert lo <= cycles <= hi, f"{cycles} cycles is outside {band}"
    meta["measurement_sane"] = True
    _record(record_perf, SMOKE_TEST, m, meta)


def _differing_words(a: np.ndarray, b: np.ndarray) -> int:
    """Output elements whose stored bits differ; a tolerance would hide a change."""
    word = np.dtype(f"u{a.itemsize}")
    return int(np.count_nonzero(a.view(word) != b.view(word)))


def _against_baseline(case: Case, config, workdir: Path, current: dict) -> None:
    """Measure ``case`` from the ``--baseline-sources`` tree beside the current one.

    Both sides get the same inputs, so their raw output words are compared
    exactly, and both must pass the contract. The rows stay the current
    tree's; the pair goes to ``--perf-meta`` and the terminal summary.
    """
    tree = config.getoption("--baseline-sources")
    # The baseline's kernels share their object names with this tree's but
    # not their sources; the registry would refuse them as a collision.
    ExternalFunction._instances.clear()
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("MLIR_AIE_KERNEL_SOURCES", tree)
        try:
            base = _measure(case, config, workdir / "baseline")
        except AssertionError as e:
            raise AssertionError(f"baseline tree {tree}: {e}") from None

    def cycles(m):
        return min(m["cycles"].kernel) if "cycles" in m else None

    def npu_us(m):
        return round(m["wall"].npu.min_us, 2) if m["wall"].npu else None

    words = [
        _differing_words(a, b) for a, b in zip(base["outputs"], current["outputs"])
    ]
    baseline = config._perf_meta.setdefault("baseline", {"sources": tree, "cases": {}})
    baseline["cases"][case.name] = {
        "cycles": [cycles(base), cycles(current)],
        "npu_us_min": [npu_us(base), npu_us(current)],
        "differing_words": sum(words),
    }


@pytest.mark.perf
@pytest.mark.parametrize("case", _PERF_CASES)
def test_kernel_perf(case, request, record_perf, workdir):
    m = _measure(case, request.config, workdir)
    _record(record_perf, case, m, request.config._perf_meta)
    if request.config.getoption("--baseline-sources"):
        _against_baseline(case, request.config, workdir, m)
