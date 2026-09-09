# test_kernel_bench.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# RUN: %pytest %s
"""The benchmark driver's gating, without a device.

``python -m aie.utils.kernel_harness`` takes its two hardware-touching steps,
``preflight_fn`` and ``measure_fn``, as arguments. The fakes below stand in
for them so the control flow can be pinned on a host:

* preflight failure or the wrong power mode -> exit 2, nothing written;
* a canary that is wrong or outside its cycle band -> exit 2;
* one kernel producing wrong output -> exit 3, nothing written;
* a valid run writes rows with the expected metric suffixes.

The case table is the real one, so the test also keeps its names unique.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from aie.utils.kernel_harness import bench
from aie.utils.kernel_harness.bench import Measurement, Preflight
from aie.utils.kernel_harness.cases import Case, load_cases

CASES_FILE = Path(__file__).parent / "npu" / "kernel_cases.py"

STRIX = Preflight(npu="npu2", arch="aie2p", device="NPU Strix", pmode="performance")


class Fake:
    """Stand-in preflight and measurement; records what was measured."""

    def __init__(self):
        self.calls: list[str] = []
        self.wrong: set[str] = set()
        self.canary_cycles = 20_000
        self.preflight: Preflight | Exception = STRIX

    def preflight_fn(self) -> Preflight:
        if isinstance(self.preflight, Exception):
            raise self.preflight
        return self.preflight

    def measure_fn(self, case: Case, **common) -> Measurement:
        self.calls.append(case.name)
        cycles = self.canary_cycles if case.name == bench.CANARY.name else 12_345
        return Measurement(
            case=case.name,
            device=common["device"],
            correct=case.name not in self.wrong,
            verdict="ok",
            cycles_median=cycles,
            work_ops=4096,
            compile_s=1.0,
            xclbin_bytes=10,
            insts_bytes=20,
            elf_bytes=30,
        )


@pytest.fixture
def fake():
    return Fake()


def _run(fake, tmp_path, *extra, cases=True):
    out, meta = tmp_path / "b.json", tmp_path / "m.json"
    argv = ["--out", str(out), "--meta", str(meta), "--no-cycles", *extra]
    if cases:
        argv += ["--cases", str(CASES_FILE)]
    code = bench.main(argv, measure_fn=fake.measure_fn, preflight_fn=fake.preflight_fn)
    return code, out, meta


def test_preflight_failure_writes_nothing(fake, tmp_path):
    fake.preflight = RuntimeError("no device")
    code, out, meta = _run(fake, tmp_path)
    assert code == 2 and not out.exists()
    assert "no device" in json.loads(meta.read_text())["preflight_error"]
    assert fake.calls == []


def test_wrong_power_mode_is_a_preflight_failure(fake, tmp_path):
    fake.preflight = Preflight(
        npu="npu2", arch="aie2p", device="NPU Strix", pmode="default"
    )
    code, out, meta = _run(fake, tmp_path)
    assert code == 2 and not out.exists()
    assert "power mode" in json.loads(meta.read_text())["preflight_error"]
    # `--pmode any` accepts whatever the runtime reports (including "unknown").
    code, out, _ = _run(fake, tmp_path, "--pmode", "any", "--only", "^passthrough/")
    assert code == 0 and out.exists()


def test_canary_out_of_band_writes_nothing(fake, tmp_path):
    fake.canary_cycles = bench.CANARY_CYCLE_BAND[1] * 10
    code, out, _ = _run(fake, tmp_path)
    assert code == 2 and not out.exists()
    assert fake.calls == [bench.CANARY.name]


def test_one_wrong_kernel_invalidates(fake, tmp_path):
    fake.wrong.add(
        next(c.name for c in load_cases(str(CASES_FILE)) if c.name.startswith("add/"))
    )
    code, out, meta = _run(fake, tmp_path)
    assert code == 3 and not out.exists()
    assert json.loads(meta.read_text())["summary"]["n_wrong"] == 1


def test_rows_have_expected_metrics(fake, tmp_path):
    code, out, _ = _run(fake, tmp_path, "--only", "^passthrough/")
    assert code == 0
    rows = json.loads(out.read_text())
    names = {r["name"].rsplit("/", 1)[-1] for r in rows}
    assert {
        "cycles",
        "cycles_per_kop",
        "compile_s",
        "xclbin_bytes",
        "insts_bytes",
        "core_elf_bytes",
    } <= names
    assert all(r["extra"].startswith("commit ") for r in rows)
    assert all("device NPU Strix" in r["extra"] for r in rows)
    assert all(r["value"] is not None for r in rows)


def test_only_filters_cases(fake, tmp_path):
    code, _, _ = _run(fake, tmp_path, "--only", "^mv/")
    assert code == 0
    assert fake.calls and all(
        c.startswith(("mv/", bench.CANARY.name)) for c in fake.calls
    )


def test_bare_kernel_names_run_at_their_default_shape(fake, tmp_path):
    code, out, _ = _run(fake, tmp_path, "add", "mul", "--calls", "4", cases=False)
    assert code == 0
    assert fake.calls[0] == bench.CANARY.name
    # factory/<elems>x<calls>/<dtype>: the shape comes from the factory's default tile.
    assert [c.split("/")[0] for c in fake.calls[1:]] == ["add", "mul"]
    assert all(re.fullmatch(r"\w+/\d+x4/bfloat16", c) for c in fake.calls[1:])


def test_device_restricted_cases_are_skipped_on_the_other_generation(fake, tmp_path):
    table = tmp_path / "cases.py"
    table.write_text(
        "from aie.utils.kernel_harness.cases import Case\n"
        "CASES = [Case('passthrough', calls=2, devices=('npu1',)),\n"
        "         Case('passthrough', calls=4, devices=('npu2',))]\n"
    )
    out, meta = tmp_path / "b.json", tmp_path / "m.json"
    code = bench.main(
        ["--out", str(out), "--meta", str(meta), "--cases", str(table)],
        measure_fn=fake.measure_fn,
        preflight_fn=fake.preflight_fn,
    )
    assert code == 0
    assert [c.split("/")[0] for c in fake.calls] == ["passthrough", "passthrough"]
    assert fake.calls[1].split("/")[1].endswith("x4")


def test_rows_for_skips_what_was_not_measured():
    m = Measurement(case="k/1/bfloat16", device="d", correct=True, verdict="ok")
    assert bench.rows_for(m, "x") == []
    m.cycles_median = 100
    assert [r["name"] for r in bench.rows_for(m, "x")] == ["k/1/bfloat16/cycles"]
    m.work_ops = 50_000
    per_kop = bench.rows_for(m, "x")[1]
    assert per_kop["name"] == "k/1/bfloat16/cycles_per_kop" and per_kop["value"] == 2.0
