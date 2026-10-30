# test_perf_series_names.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# RUN: %python %s
# REQUIRES: python_bindings

"""Pin the performance series keys.

``nightlyKernelChecks.yml`` publishes rows named ``<Case.name>/<metric>``,
and ``utils/kernel_checks/publish.py`` keys each chart's history on that
string. Renaming a case therefore does not rename a chart -- it abandons one
and starts another, silently, and the loss is only visible on the published
dashboard.

Nothing else ties those names down, so this does. It is a snapshot test on
purpose: a diff to ``perf_series.txt`` is the reviewable record of which
chart histories a change ends.
"""

import sys
from pathlib import Path

import numpy as np
from aie.iron import kernels
from aie.iron.algorithms import kernel_design as kd
from aie.iron.device import NPU2Col1
from aie.utils.hostruntime import set_current_device
from ml_dtypes import bfloat16

sys.path.insert(0, str(Path(__file__).parent))

from cases import Case, device_for  # noqa: E402
from kernel_cases import CASES  # noqa: E402
from test_kernels_perf import (  # noqa: E402
    _PERF_CASES,
    SMOKE_TEST,
    _differing_words,
    _kernel_object_size,
    _record,
)

_SNAPSHOT = Path(__file__).parent / "perf_series.txt"


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
        f"these performance series would lose their published history: {removed}. "
        f"If that is intended, update {_SNAPSHOT.name}."
    )
    assert not added, (
        f"new performance series: {added}. Add them to {_SNAPSHOT.name} so the "
        "next rename is still caught."
    )


def test_series_names_are_unique():
    """Two cases sharing a name would overwrite each other's chart."""
    names = _current()
    duplicates = sorted({n for n in names if names.count(n) > 1})
    assert not duplicates, f"cases share a performance series: {duplicates}"


def test_smoke_case_is_measured_once():
    names = [SMOKE_TEST.name, *(param.values[0].name for param in _PERF_CASES)]
    assert len(names) == len(set(names))
    assert set(names) == {case.name for case in CASES if case.perf}


def test_cycle_efficiency_is_independent_of_call_count():
    for calls in (1, 16, 256):
        case = Case("passthrough", dict(tile_size=2048), calls=calls)
        rows = []
        traced = kd.CallCycles(kernel=(270,) * calls)
        _record(lambda *row: rows.append(row), case, {"cycles": traced})
        assert rows[1][1:] == ("cycles_per_kop", "cycles/1k-ops", 131.836)


def test_cycles_row_is_the_kernel_min_with_its_spread_beside_it():
    case = Case("mm", dict(dim_m=32, dim_k=64, dim_n=32), calls=3, devices=("npu2",))
    # One call stalled; zero's intervals are their own population.
    traced = kd.CallCycles(kernel=(900, 1400, 905), initializers={2: (60, 61, 60)})
    rows = []
    _record(lambda *row: rows.append(row), case, {"cycles": traced})
    assert rows[0][1:] == (
        "cycles",
        "cycles",
        900,
        "median 905 max 1400 n=3; init[2] min 60",
    )


def test_size_records_the_bytes_the_tested_entry_links():
    """reduce_max.cc compiles every dtype's entry points into one object.

    Each build's row counts its own entry's code and data, not the object
    file, which all of them share.
    """
    sizes = {}
    with set_current_device(NPU2Col1()):
        for dtype in (np.int32, bfloat16):
            fn = kernels.reduce_max(dtype=dtype)
            design = kd.KernelDesign(fn).design()
            design.compilable.compile()
            obj = design.compilable.get_cache_entry().directory / fn.object_file_name
            sizes[dtype] = _kernel_object_size(design, fn)
            assert 0 < sizes[dtype] < obj.stat().st_size
        assert sizes[np.int32] != sizes[bfloat16]


def test_cycles_row_is_the_median_call_when_calls_do_unequal_work():
    case = Case("resample_quantize", calls=4, scalars=(4032, 912, 11, 4))
    traced = kd.CallCycles(kernel=(100, 5000, 7000, 9000))
    rows = []
    _record(lambda *row: rows.append(row), case, {"cycles": traced})
    assert rows[0][1:] == (
        "cycles",
        "cycles",
        6000,
        "median of calls of unequal work; min 100 max 9000 n=4",
    )


def test_raw_words_compare_bits_not_values():
    a = np.array([0.0, 1.0, np.nan], dtype=np.float32)
    b = np.array([-0.0, 1.0, np.nan], dtype=np.float32)
    assert _differing_words(a, a.copy()) == 0
    assert _differing_words(a, b) == 1


def test_matrix_series_keep_tile_geometry_and_call_count():
    cases = [
        Case("mm", dict(dim_m=32, dim_k=64, dim_n=32), calls=4, devices=("npu2",)),
        Case("mm", dict(dim_m=64, dim_k=32, dim_n=32), calls=4, devices=("npu2",)),
        Case("mm", dict(dim_m=32, dim_k=64, dim_n=32), calls=8, devices=("npu2",)),
    ]
    assert len({case.name for case in cases}) == len(cases)


if __name__ == "__main__":
    test_series_names_are_unchanged()
    test_series_names_are_unique()
    test_smoke_case_is_measured_once()
    test_cycle_efficiency_is_independent_of_call_count()
    test_cycles_row_is_the_kernel_min_with_its_spread_beside_it()
    test_size_records_the_bytes_the_tested_entry_links()
    test_cycles_row_is_the_median_call_when_calls_do_unequal_work()
    test_raw_words_compare_bits_not_values()
    test_matrix_series_keep_tile_geometry_and_call_count()
    print("PASS!")
