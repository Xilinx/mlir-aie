# test_provenance_host.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# RUN: %pytest %s
"""A benchmark row names the XRT, the driver and the runner that measured it.

Nightly runs move between hosts of one generation, and a driver update
moves host-side timing the way a kernel regression would, so the row has to
say which machine and stack produced it.
"""

import subprocess
from types import SimpleNamespace

import aie.utils.benchmark as benchmark
import aie.utils.probe as probe
import pytest
from aie.utils.benchmark import parse_xrt_examine, provenance, xrt_versions

EXAMINE = """\
System Configuration
  OS Name              : Linux
  Release              : 6.14.0-29-generic
  Version              : #29~24.04.1-Ubuntu SMP
  Machine              : x86_64

XRT
  Version              : 2.20.0
  Branch               : master
  Hash                 : 7ba6be5
  Hash Date            : 2026-09-01 10:00:00
  amdxdna              : 2.20.0_20250915, 6.14.0-29-generic

Device(s) Present
|BDF             |Name          |
|----------------|--------------|
|[0000:c5:00.1]  |NPU Krackan 1 |
"""


def test_examine_is_read_from_its_xrt_section_only():
    # The OS "Version" line above the XRT heading is not XRT's.
    assert parse_xrt_examine(EXAMINE) == {"xrt": "2.20.0", "xdna": "2.20.0_20250915"}
    assert parse_xrt_examine("") == {}
    assert parse_xrt_examine("XRT\n  Version : 2.19.0\n") == {"xrt": "2.19.0"}
    assert parse_xrt_examine("XRT\n  Branch : master\n") == {}


def test_versions_and_host_reach_the_row(monkeypatch):
    monkeypatch.setattr(probe, "xrt_smi_path", lambda: "/opt/xilinx/xrt/bin/xrt-smi")
    calls = []

    def run(argv, **kwargs):
        calls.append(argv)
        return SimpleNamespace(stdout=EXAMINE)

    monkeypatch.setattr(benchmark.subprocess, "run", run)
    monkeypatch.setenv("RUNNER_NAME", "bench-aie2p-3")
    assert xrt_versions() == {"xrt": "2.20.0", "xdna": "2.20.0_20250915"}
    assert calls[-1] == ["/opt/xilinx/xrt/bin/xrt-smi", "examine"]
    fields = provenance(device="NPU Krackan 1", pmode="performance").split(" | ")
    assert "xrt 2.20.0" in fields
    assert "xdna 2.20.0_20250915" in fields
    assert "host bench-aie2p-3" in fields
    # Extra fields stay last, so the device still ends the line.
    assert fields[-2:] == ["device NPU Krackan 1", "pmode performance"]


@pytest.mark.parametrize(
    "failure", [OSError("gone"), subprocess.TimeoutExpired("x", 1)]
)
def test_a_failing_xrt_smi_is_left_out(monkeypatch, failure):
    monkeypatch.setattr(probe, "xrt_smi_path", lambda: "xrt-smi")

    def run(argv, **kwargs):
        raise failure

    monkeypatch.setattr(benchmark.subprocess, "run", run)
    assert xrt_versions() == {}


def test_without_xrt_or_a_runner_nothing_is_claimed(monkeypatch):
    monkeypatch.setattr(probe, "xrt_smi_path", lambda: None)
    monkeypatch.delenv("RUNNER_NAME", raising=False)
    assert xrt_versions() == {}
    keys = [f.split(" ", 1)[0] for f in provenance().split(" | ")]
    assert "xrt" not in keys and "xdna" not in keys and "host" not in keys
