# test_benchmark_series_names.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# RUN: %python %s
# REQUIRES: python_bindings

"""Pin the benchmark series keys.

``benchmarkKernels.yml`` feeds rows named ``<Case.name>/<metric>`` to
benchmark-action, which keys each chart on that string and keeps its history
under it. Renaming a case therefore does not rename a chart -- it abandons one
and starts another, silently, and the loss is only visible on the published
dashboard.

Nothing else ties those names down, so this does. It is a snapshot test on
purpose: a diff to ``benchmark_series.txt`` is the reviewable record of which
chart histories a change ends.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from cases import device_for  # noqa: E402
from kernel_cases import CASES  # noqa: E402

_SNAPSHOT = Path(__file__).parent / "benchmark_series.txt"


def _recorded() -> list[str]:
    lines = _SNAPSHOT.read_text().splitlines()
    return sorted(ln for ln in lines if ln and not ln.startswith("#"))


def _current() -> list[str]:
    names = []
    for case in CASES:
        with device_for(case.devices):
            names.append(case.name)
    return sorted(names)


def test_series_names_are_unchanged():
    recorded, current = _recorded(), _current()
    added = sorted(set(current) - set(recorded))
    removed = sorted(set(recorded) - set(current))
    assert not removed, (
        f"these benchmark series would lose their published history: {removed}. "
        f"If that is intended, update {_SNAPSHOT.name}."
    )
    assert not added, (
        f"new benchmark series: {added}. Add them to {_SNAPSHOT.name} so the "
        "next rename is still caught."
    )


def test_series_names_are_unique():
    """Two cases sharing a name would overwrite each other's chart."""
    names = _current()
    duplicates = sorted({n for n in names if names.count(n) > 1})
    assert not duplicates, f"cases share a benchmark series: {duplicates}"


if __name__ == "__main__":
    test_series_names_are_unchanged()
    test_series_names_are_unique()
    print("PASS!")
