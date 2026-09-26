#!/usr/bin/env python3
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Write one NPU's column of the kernel catalogue the results page shows.

For every factory in ``aie.iron.kernels``: its family, summary and sources,
the builds it offers on this NPU's architecture, and how its cases fared in
the nightly extensive sweep and performance checks. nightlyKernelChecks.yml
runs it on each NPU after timing, and publishKernelResults.yml installs the
result as ``kernel-checks/<npu>/catalogue.json`` beside that NPU's
``data.js``.
"""

import argparse
import datetime
import inspect
import json
import os
import re
import sys
from collections import defaultdict

import pr_report
from aie.iron import kernels
from aie.iron.device import from_name
from aie.iron.kernels._common import ARCH_TRAITS
from aie.utils import get_current_device
from aie.utils.compile.remarks import kernel_builds
from aie.utils.hostruntime import set_current_device

_LUT_SOURCE = re.compile(r'-DAIE_LUT_KERNEL_SOURCE="(.+)"')


def _library_path(path: str) -> str:
    return path.rsplit("/aie_kernels/", 1)[-1]


def _sources(ef) -> list[str]:
    wrapped = [m[1] for f in ef.compile_flags if (m := _LUT_SOURCE.fullmatch(f))]
    if wrapped:
        return [_library_path(p) for p in wrapped]
    return [_library_path(ef.source_file)] if ef.source_file else []


def _by_factory(case_names) -> dict[str, set[str]]:
    grouped = defaultdict(set)
    for case in case_names:
        grouped[case.split("/", 1)[0]].add(case)
    return grouped


def _swept(path) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    """Return the cases of the extensive sweep that passed and that failed, by factory.

    A case failed if any input or seed failed, as the performance checks exclude it.
    """
    passed, failed = set(), set()
    for case, _, test in pr_report.sweep(path):
        (failed if pr_report.failed(test) else passed).add(case)
    return _by_factory(passed - failed), _by_factory(failed)


def _timed(path) -> dict[str, set[str]]:
    with open(path) as f:
        rows = json.load(f)
    return _by_factory(row["name"].rsplit("/", 1)[0] for row in rows)


def catalogue(npu: str, correctness=None, perf=None) -> dict:
    """Return ``npu``'s column: every factory, and what this NPU offers and verified of it."""
    arch = next(t.name for t in ARCH_TRAITS.values() if t.device == npu)
    previous = get_current_device(probe_runtime=False)
    set_current_device(from_name(npu, n_cols=1))
    try:
        builds = defaultdict(list)
        sources = defaultdict(set)
        for name, ef in kernel_builds():
            factory = name.split("/", 1)[0]
            builds[factory].append(name)
            sources[factory].update(_sources(ef))
    finally:
        set_current_device(previous)

    passed, failed = _swept(correctness) if correctness else ({}, {})
    timed = _timed(perf) if perf else {}
    rows = []
    for factory in kernels.factories():
        f = getattr(kernels, factory)
        rows.append(
            {
                "factory": factory,
                "family": f.__module__.rsplit(".", 1)[-1],
                "summary": (inspect.getdoc(f) or "")
                .split("\n", 1)[0]
                .replace("``", ""),
                "sources": sorted(sources[factory]),
                "builds": builds[factory],
                "passed": len(passed.get(factory, ())),
                "failed": sorted(failed.get(factory, ())),
                "timed": len(timed.get(factory, ())),
            }
        )
    return {
        "npu": npu,
        "arch": arch,
        "commit": os.environ.get("GITHUB_SHA", ""),
        "date": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "swept": correctness is not None,
        "kernels": rows,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument(
        "--npu", required=True, choices=[t.device for t in ARCH_TRAITS.values()]
    )
    parser.add_argument("--correctness", help="junit XML of the extensive sweep")
    parser.add_argument("--perf", help="perf.json of the timed cases")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    result = catalogue(args.npu, args.correctness, args.perf)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
