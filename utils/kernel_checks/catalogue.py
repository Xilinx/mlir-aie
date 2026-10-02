#!/usr/bin/env python3
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Write one NPU's column of the kernel catalogue the results page shows.

For every factory in ``aie.iron.kernels``: its family, summary and sources,
the builds it offers on this NPU's architecture, and how its cases fared in
the nightly extensive sweep and performance checks. nightlyKernelChecks.yml
runs it on each NPU after timing, and publishKernelResults.yml installs the
result as ``kernel-checks/<npu>/catalogue.json`` beside that NPU's run
records (``utils/kernel_checks/publish.py``).

Per factory the column records, for this NPU:

* ``passed`` / ``failed``: cases the correctness run passed (count) and
  failed (names), including dedicated ``kernel_check`` tests. A case failed
  if any input or seed failed.
* ``timed``: cases the performance checks recorded rows for (count).
* ``timing_failed``: cases that passed the sweep but failed in the timing
  run (names, from ``meta.json``), so a series with a gap has a reason.
* ``untimed``: cases that passed the sweep and are not timed by design
  (``perf=False`` in ``kernel_cases.py`` or a dedicated check), so "N cases
  pass" and "M timed" add up.
* ``flaky``: cases that passed, but only when the runner retried a failed
  sweep input or timing test (names).
* ``reason``: why a factory with builds has no case at all.

When the timing run measured nothing at all (it refused the device's power
mode, or preflight did not pass), the column says why in ``timing_refused``
and no case is ``timing_failed``: one reason, not one per case.

The row assembly (:func:`rows`) needs only the standard library, so it is
tested on any host; only :func:`catalogue` imports the ``aie`` package.
"""

import argparse
import datetime
import inspect
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import pr_report

_LUT_SOURCE = re.compile(r'-DAIE_LUT_KERNEL_SOURCE="(.+)"')
_CASES_DIR = Path(__file__).resolve().parents[2] / "test/python/npu"
NO_CASE = "no case in test/python/npu/kernel_cases.py"


def _library_path(path: str) -> str:
    return path.rsplit("/aie_kernels/", 1)[-1]


def _sources(ef) -> list[str]:
    wrapped = [m[1] for f in ef.compile_flags if (m := _LUT_SOURCE.fullmatch(f))]
    if wrapped:
        return [_library_path(p) for p in wrapped]
    return [_library_path(ef.source_file)] if ef.source_file else []


def why_uncased(ef) -> str | None:
    """Return why the generic builder has no case for ``ef``, from its contract.

    The contract's ``unsupported`` reason (a cascade PUT half has no output
    to judge), else the reason its trace is ``none`` (``set_rounding`` runs
    once before the calls), else None.
    """
    contract = getattr(ef, "contract", None)
    if contract is None:
        return None
    if getattr(contract, "unsupported", None):
        return str(contract.unsupported)
    trace = getattr(contract, "trace", None)
    if trace is not None and getattr(trace, "shape", None) == "none":
        return getattr(trace, "reason", None) or None
    return None


def _by_factory(case_names) -> dict[str, set[str]]:
    grouped = defaultdict(set)
    for case in case_names:
        grouped[case.split("/", 1)[0]].add(case)
    return grouped


def swept(path) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    """Return generic and dedicated checks that passed and failed, by factory.

    A case failed if any input or seed failed, as the performance checks exclude it.
    """
    passed, failed = set(), set()
    for case, _, test in pr_report.sweep(path):
        (failed if pr_report.failed(test) else passed).add(case)
    return _by_factory(passed - failed), _by_factory(failed)


def timed(path) -> dict[str, set[str]]:
    with open(path) as f:
        rows_ = json.load(f)
    return _by_factory(row["name"].rsplit("/", 1)[0] for row in rows_)


def _timing_cases(nodeids) -> list[str]:
    cases = []
    for nodeid in nodeids:
        if "test_kernel_perf[" not in nodeid:
            continue
        match = pr_report._BRACKETED.search(nodeid)
        if match:
            cases.append(match[1])
    return cases


def timing_refused(path) -> str | None:
    """Return why the timing run measured nothing at all, or None if it measured."""
    with open(path) as f:
        meta = json.load(f)
    if meta.get("refused"):
        return str(meta["refused"])
    if "preflight" not in meta:
        return "the NPU preflight did not pass"
    return None


def timing_failed(path) -> dict[str, set[str]]:
    """Return the cases ``meta.json`` lists as failed in the timing run, by factory.

    The sweep's own failures reach the meta too; those are the correctness
    report's to attribute, so only the timing tests count here. A run that
    measured nothing failed every timing test for one reason
    (:func:`timing_refused`), so none of them counts.
    """
    if timing_refused(path):
        return {}
    with open(path) as f:
        failed = json.load(f).get("failed", [])
    return _by_factory(_timing_cases(failed))


def flaky(correctness=None, meta=None) -> dict[str, set[str]]:
    """Return the cases that passed only on a retry, by factory."""
    cases = set()
    if correctness:
        for case, _, test in pr_report.sweep(correctness):
            if pr_report.reruns(test) and not pr_report.failed(test):
                cases.add(case)
    if meta:
        with open(meta) as f:
            record = json.load(f)
        bad = set(record.get("failed", []))
        retried = [n for n in record.get("reruns", {}) if n not in bad]
        cases.update(_timing_cases(retried))
    return _by_factory(cases)


def rows(
    factories: list[dict],
    passed: dict[str, set[str]],
    failed: dict[str, set[str]],
    timed_: dict[str, set[str]],
    timing_failed_: dict[str, set[str]],
    declared: dict[str, dict[str, bool]],
    flaky_: dict[str, set[str]] | None = None,
) -> list[dict]:
    """Assemble the catalogue rows from what the run produced.

    ``factories`` carries each factory's ``factory``, ``family``,
    ``summary``, ``sources`` and ``builds``, and ``why`` when its contract
    says why the builder cannot run it; ``declared`` maps a factory to its
    cases on this NPU and whether each is timed (``perf``).
    """
    out = []
    for f in factories:
        f = dict(f)
        why = f.pop("why", None)
        name = f["factory"]
        cases = declared.get(name, {})
        ok = passed.get(name, set())
        row = {
            **f,
            "passed": len(ok),
            "failed": sorted(failed.get(name, ())),
            "timed": len(timed_.get(name, ())),
            "timing_failed": sorted(timing_failed_.get(name, ())),
            "untimed": sorted(c for c in ok if cases.get(c) is False),
            "flaky": sorted((flaky_ or {}).get(name, ())),
        }
        if f["builds"] and not cases:
            row["reason"] = why or NO_CASE
        out.append(row)
    return out


def _declared(npu: str, cases_dir: Path) -> dict[str, dict[str, bool]]:
    """Every case of ``kernel_cases.py`` this NPU supports: ``factory -> {name: perf}``."""
    sys.path.insert(0, str(cases_dir))
    try:
        # test/python/npu is put on the path above, at run time.
        from kernel_cases import CASES  # pyright: ignore[reportMissingImports]
    finally:
        sys.path.pop(0)
    declared: dict[str, dict[str, bool]] = defaultdict(dict)
    for case in CASES:
        if not case.supported_on(npu):
            continue
        try:
            name = case.name
        except NotImplementedError:
            continue  # the factory exists only for the other architecture
        declared[case.factory][name] = bool(case.perf)
    return declared


def catalogue(
    npu: str, correctness=None, perf=None, meta=None, cases_dir: Path = _CASES_DIR
) -> dict:
    """Return ``npu``'s column: every factory, and what this NPU offers and verified of it."""
    from aie.iron import kernels
    from aie.iron.device import from_name
    from aie.iron.kernels._common import ARCH_TRAITS
    from aie.utils import get_current_device
    from aie.utils.compile.remarks import kernel_builds
    from aie.utils.hostruntime import set_current_device

    arch = next(t.name for t in ARCH_TRAITS.values() if t.device == npu)
    previous = get_current_device(probe_runtime=False)
    set_current_device(from_name(npu, n_cols=1))
    try:
        builds = defaultdict(list)
        sources = defaultdict(set)
        why: dict[str, str] = {}
        for name, ef in kernel_builds():
            factory = name.split("/", 1)[0]
            builds[factory].append(name)
            sources[factory].update(_sources(ef))
            if factory not in why and (reason := why_uncased(ef)):
                why[factory] = reason
        declared = _declared(npu, cases_dir)
    finally:
        set_current_device(previous)

    if correctness:
        for case, variant, _ in pr_report.sweep(correctness, include_skipped=True):
            if variant == "dedicated":
                declared.setdefault(case.split("/", 1)[0], {})[case] = False

    factories = [
        {
            "factory": factory,
            "family": getattr(kernels, factory).__module__.rsplit(".", 1)[-1],
            "summary": (inspect.getdoc(getattr(kernels, factory)) or "")
            .split("\n", 1)[0]
            .replace("``", ""),
            "sources": sorted(sources[factory]),
            "builds": builds[factory],
            **({"why": why[factory]} if factory in why else {}),
        }
        for factory in kernels.factories()
    ]
    passed, failed = swept(correctness) if correctness else ({}, {})
    refused = timing_refused(meta) if meta else None
    return {
        "npu": npu,
        "arch": arch,
        "commit": os.environ.get("GITHUB_SHA", ""),
        "date": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "swept": correctness is not None,
        **({"timing_refused": refused} if refused else {}),
        "kernels": rows(
            factories,
            passed,
            failed,
            timed(perf) if perf else {},
            timing_failed(meta) if meta else {},
            declared,
            flaky(correctness, meta),
        ),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument("--npu", required=True, choices=["npu1", "npu2"])
    parser.add_argument("--correctness", help="junit XML of the extensive sweep")
    parser.add_argument("--perf", help="perf.json of the timed cases")
    parser.add_argument("--meta", help="meta.json of the timing run, for its failures")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    result = catalogue(args.npu, args.correctness, args.perf, args.meta)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
