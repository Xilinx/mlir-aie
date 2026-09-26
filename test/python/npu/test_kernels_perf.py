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
preflight did not pass, or ``test_measurement_is_sane`` ran and failed -- in
which case ``pytest_sessionfinish`` writes none of it.
With ``--correctness-results correctness.xml``, publication also excludes
cases with any extensive-suite failure or no passing correctness test.

Run it the way the nightly workflow does::

    pytest test/python/npu/test_kernels_perf.py -m perf
        --perf-out perf.json --perf-meta meta.json --pmode turbo
        --warmup 10 --iters 50

``-k`` selects a subset. A subset that leaves out the sanity test still
writes its rows, with ``measurement_sane`` null in the meta; include it, for
example ``-k '(softmax) or test_measurement_is_sane'``, to vouch for them.
The series a row lands in is ``<case>/<metric>``;
``test_perf_series_names.py`` pins those names, because renaming one
restarts its chart on gh-pages.

With ``--baseline-sources DIR`` each case also runs with its kernels from
``DIR`` and the two arms' raw output words are compared; the ``check()``
cases, which are not timed, join that comparison untimed.
"""

from __future__ import annotations

import re
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest
from aie.iron import ExternalFunction, kernels
from aie.iron.algorithms import kernel_design as kd
from aie.utils.config import aie_kernels_dir, aie_runtime_lib_dir
from aie.utils.benchmark import preflight, provenance, run_iters
from aie.utils.compile.remarks import current_kernel_sources
from cases import Case, error_report, inputs_for
from kernel_cases import CASES

TRACE_SIZE = 16384
# The add/256 case filled 16 KB after 91 intervals (180 B each) when every
# default core event was traced; markers alone take about 5 B. Size for
# every declared interval with headroom, so the split sees whole calls.
TRACE_BYTES_PER_INTERVAL = 512

_STACK_SHORT = re.compile(
    r"stack_size = (\d+) is insufficient: this core needs (\d+) bytes"
)

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


def _measure(
    case: Case,
    config,
    workdir: Path,
    *,
    strict: bool = True,
    stack_bytes: int | None = None,
    timed: bool = True,
) -> dict:
    """Build, check and time one case. Raises if it is wrong, unless not ``strict``.

    Not ``strict``, a wrong output is timed anyway and its verdict's detail
    is the result's ``"failed"`` (None when it passed). ``stack_bytes``
    replaces the contract's core stack. Not ``timed``, it only runs once.
    """
    fn = case.fn()
    factory = getattr(kernels, case.factory)
    inputs = inputs_for(case, "random", np.random.default_rng(0))
    design = kd.design(
        factory,
        **case.harness_opts(),
        params=fn.param_values(inputs),
        aiecc_flags=["--get-core-elfs"],
        stack_bytes=stack_bytes,
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
    assert verdict or not strict, f"{case.name}: {verdict.detail}"
    measured: dict = {
        "outputs": got,
        "sizes": _sizes(design),
        "failed": None if verdict else verdict.detail,
    }
    measured["error"] = error_report(
        fn,
        got if len(got) > 1 else got[0],
        inputs,
        calls=case.calls,
        scalars=case.scalars,
    )
    if not timed:
        return measured
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
        traced = kd.cycles_per_call(
            design,
            inputs,
            out_n,
            out_dt,
            trace_size=max(TRACE_SIZE, TRACE_BYTES_PER_INTERVAL * intervals),
            workdir=workdir,
            fn=fn,
            calls=case.calls,
        )
        # Every kernel checked here is timed; one that is not would chart only
        # wall clock and still pass.
        assert not traced.untimed, f"{case.name}: untimed, {traced.untimed}"
        # A short trace would chart the min of the calls that fit.
        assert not traced.truncated, (
            f"{case.name}: {len(traced.kernel)} of {case.calls} "
            "calls traced; the rest never reached the host"
        )
        measured["cycles"] = traced
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
    return "; ".join(parts)


def _record(record, case: Case, m: dict) -> None:
    """Emit the rows one measurement contributes, in series-name order."""
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
        lo, hi = SMOKE_CYCLE_BAND
        assert cycles is not None, "traced run produced no cycle count"
        assert lo <= cycles <= hi, f"{cycles} cycles is outside {SMOKE_CYCLE_BAND}"
    meta["measurement_sane"] = True
    _record(record_perf, SMOKE_TEST, m)


def _differing_words(a: np.ndarray, b: np.ndarray) -> int:
    """Output elements whose stored bits differ; a tolerance would hide a change."""
    word = np.dtype(f"u{a.itemsize}")
    return int(np.count_nonzero(a.view(word) != b.view(word)))


def _measure_and_compare(
    case: Case,
    config,
    workdir: Path,
    *,
    tree: str | None = None,
    meta: dict | None = None,
    timed: bool = True,
) -> dict:
    """``_measure`` ``case``, compared with ``tree`` before it is judged.

    ``tree`` defaults to ``--baseline-sources`` and ``meta`` to the run's
    meta. A kernel that fails its contract still gets its words compared --
    a mutant has to show up in that diff for a ``same`` to mean anything --
    and then fails.
    """
    tree = tree or config.getoption("--baseline-sources")
    m = _measure(case, config, workdir, strict=not tree, timed=timed)
    if tree:
        entry = _compare(case, config, workdir, m, tree)
        meta = config._perf_meta if meta is None else meta
        if "baseline" not in meta:
            current_sources, suspect = current_kernel_sources(tree)
            meta["baseline"] = {
                "sources": tree,
                "current_sources": current_sources,
                "warning": suspect,
                "cases": {},
            }
        meta["baseline"]["cases"][case.name] = entry
    assert m["failed"] is None, f"{case.name}: {m['failed']}"
    return m


def _compare(case: Case, config, workdir: Path, current: dict, tree: str) -> dict:
    """Measure ``case`` from the kernel tree ``tree`` beside the current one.

    Both sides get the same inputs, so their raw output words are compared
    exactly. Both arms are judged by this tree's contract and only recorded
    here: the caller fails the current arm, and the baseline's miss can be a
    tolerance tightened along with the kernel. Returns the pair of each
    arm's cycles and npu_us (min, and min, max and n), each arm's error
    against the reference (``cases.error_report``), the differing words and
    each arm's verdict.

    The contract's stack is sized for this tree's kernel too, and aiecc
    refuses a baseline that needs more; that one is built with the stack
    aiecc measured, recorded as ``"baseline_stack"`` ``[contract, needed]``.
    The baseline is timed only if ``current`` was.
    """
    timed = "wall" in current
    # The baseline's kernels share their object names with this tree's but
    # not their sources; the registry would refuse them as a collision.
    ExternalFunction._instances.clear()
    stack = None
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("MLIR_AIE_KERNEL_SOURCES", tree)
        try:
            try:
                base = _measure(
                    case, config, workdir / "baseline", strict=False, timed=timed
                )
            except RuntimeError as e:
                if not (short := _STACK_SHORT.search(str(e))):
                    raise
                stack = [int(short[1]), int(short[2])]
                ExternalFunction._instances.clear()
                base = _measure(
                    case,
                    config,
                    workdir / "baseline",
                    strict=False,
                    stack_bytes=stack[1],
                    timed=timed,
                )
        except AssertionError as e:
            raise AssertionError(f"baseline tree {tree}: {e}") from None

    def cycles(m):
        if "cycles" not in m:
            return None
        k = m["cycles"].kernel
        return {"min": min(k), "max": max(k), "n": len(k)}

    def npu_us(m):
        if "wall" not in m or not (s := m["wall"].npu):
            return None
        return {"min": round(s.min_us, 2), "max": round(s.max_us, 2), "n": s.n}

    words = [
        _differing_words(a, b) for a, b in zip(base["outputs"], current["outputs"])
    ]
    cycles_range = [cycles(base), cycles(current)]
    npu_us_range = [npu_us(base), npu_us(current)]
    return {
        "cycles": [r and r["min"] for r in cycles_range],
        "npu_us_min": [r and r["min"] for r in npu_us_range],
        "cycles_range": cycles_range,
        "npu_us_range": npu_us_range,
        "differing_words": sum(words),
        "accuracy": [base["error"], current["error"]],
        "baseline_failed": base["failed"],
        "current_failed": current["failed"],
        "baseline_stack": stack,
    }


@pytest.mark.perf
@pytest.mark.parametrize("case", _PERF_CASES)
def test_kernel_perf(case, request, record_perf, workdir):
    m = _measure_and_compare(case, request.config, workdir)
    _record(record_perf, case, m)


def pytest_generate_tests(metafunc):
    if "check_case" in metafunc.fixturenames:
        compared = metafunc.config.getoption("--baseline-sources")
        cases = [c for c in CASES if not c.perf] if compared else []
        metafunc.parametrize("check_case", [_param(c) for c in cases])


@pytest.mark.perf
def test_kernel_same_as_baseline(check_case, request, workdir):
    """A case the run checks but does not time, compared with --baseline-sources.

    It gets the raw-word comparison and both arms' verdicts, not a row;
    without --baseline-sources nothing is collected.
    """
    _measure_and_compare(check_case, request.config, workdir, timed=False)


@pytest.mark.perf
def test_a_baseline_that_fails_the_contract_is_timed(request, workdir, tmp_path):
    """A baseline the current contract rejects is recorded and timed, not failed.

    The contract is this tree's, so a change that fixes a kernel and tightens
    its tolerance must still be able to time the kernel it replaced.
    """
    tree = tmp_path / "base"
    shutil.copytree(aie_kernels_dir(), tree / "aie_kernels")
    shutil.copytree(aie_runtime_lib_dir(), tree / "aie_runtime_lib")
    for header in ("relu_aie2.h", "relu_aie2p.h"):
        path = tree / "aie_kernels" / "eltwise" / header
        assert path.read_text().count("  event1();") == 1
        path.write_text(
            path.read_text().replace(
                "  event1();", "  c[0] = (bfloat16)-1.0f;\n  event1();"
            )
        )
    case = Case("relu", calls=4)
    current = _measure(case, request.config, workdir)
    entry = _compare(case, request.config, workdir, current, str(tree))
    assert "mismatches" in entry["baseline_failed"]
    assert entry["differing_words"] == case.calls
    assert entry["npu_us_min"][0] is not None
    if not request.config.getoption("--no-cycles"):
        assert entry["cycles"][0] is not None


@pytest.mark.perf
def test_a_baseline_that_needs_more_stack_is_timed(request, workdir, tmp_path):
    """A baseline over this tree's stack contract is built with the stack it needs.

    A change that shrinks a kernel's frame lowers its ``stack_bytes`` with
    it, and aiecc refuses the old kernel under the new contract.
    """
    tree = tmp_path / "base"
    shutil.copytree(aie_kernels_dir(), tree / "aie_kernels")
    shutil.copytree(aie_runtime_lib_dir(), tree / "aie_runtime_lib")
    frame = "  volatile int frame[1024];\n  for (int k = 0; k < 1024; k++)\n    frame[k] = k;\n"
    for header in ("relu_aie2.h", "relu_aie2p.h"):
        path = tree / "aie_kernels" / "eltwise" / header
        assert path.read_text().count("  event1();") == 1
        path.write_text(path.read_text().replace("  event1();", f"{frame}  event1();"))
    case = Case("relu", calls=4)
    current = _measure(case, request.config, workdir)
    entry = _compare(case, request.config, workdir, current, str(tree))
    contract, needed = entry["baseline_stack"]
    assert contract == kd._stack_bytes(case.fn()) and needed > 4096
    assert entry["baseline_failed"] is None and entry["differing_words"] == 0
    assert entry["npu_us_min"][0] is not None


@pytest.mark.perf
def test_a_wrong_kernel_is_compared_before_it_fails(request, workdir, tmp_path):
    """A kernel that fails its contract still gets its words compared, then fails.

    Otherwise a mutant, the one way to show the word diff sees a change,
    fails on the reference before the diff is taken.
    """
    trees = {}
    for name in ("base", "wrong"):
        trees[name] = tmp_path / name
        shutil.copytree(aie_kernels_dir(), trees[name] / "aie_kernels")
        shutil.copytree(aie_runtime_lib_dir(), trees[name] / "aie_runtime_lib")
    for header in ("relu_aie2.h", "relu_aie2p.h"):
        path = trees["wrong"] / "aie_kernels" / "eltwise" / header
        assert path.read_text().count("  event1();") == 1
        path.write_text(
            path.read_text().replace(
                "  event1();", "  c[0] = (bfloat16)-1.0f;\n  event1();"
            )
        )
    case = Case("relu", calls=4)
    meta: dict = {}
    # This tree's relu, and later tests', share the wrong one's object name.
    ExternalFunction._instances.clear()
    try:
        with pytest.MonkeyPatch.context() as mp:
            mp.setenv("MLIR_AIE_KERNEL_SOURCES", str(trees["wrong"]))
            with pytest.raises(AssertionError, match="mismatches"):
                _measure_and_compare(
                    case,
                    request.config,
                    workdir,
                    tree=str(trees["base"]),
                    meta=meta,
                    timed=False,
                )
    finally:
        ExternalFunction._instances.clear()
    entry = meta["baseline"]["cases"][case.name]
    assert entry["differing_words"] == case.calls
    assert "mismatches" in entry["current_failed"] and entry["baseline_failed"] is None
