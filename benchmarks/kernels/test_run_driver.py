# benchmarks/kernels/test_run_driver.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Host-only tests of run.py's control flow with the hardware layer mocked.

Verifies, without an NPU:
  * preflight failure -> exit 2, no bench.json
  * canary wrong / out of band -> exit 2, no bench.json
  * one wrong kernel -> exit 3, no bench.json
  * a valid run writes rows with the expected metric suffixes

Needs the ``aie`` package (the registry names real factories) but no device:
``measure`` and ``preflight`` are replaced, so nothing compiles or runs.
"""

from __future__ import annotations

import json
import types

import pytest

_kernels = pytest.importorskip(
    "aie.iron.kernels", reason="requires an installed aie package"
)
if not getattr(_kernels, "__file__", None):
    pytest.skip("aie.iron.kernels is a stub", allow_module_level=True)

from . import (
    _util,  # noqa: E402
    registry,  # noqa: E402
)
from . import run as run_mod  # noqa: E402
from .harness import Measurement  # noqa: E402


@pytest.fixture
def env(monkeypatch, tmp_path):
    """Mock preflight and measure; record the order entries were measured."""
    calls: list[str] = []
    state = {"wrong": set(), "canary_cycles": 20_000, "preflight_fail": False}

    def fake_preflight(pmode):
        if state["preflight_fail"]:
            raise run_mod.PreflightError("no device")
        return {"bdf": "0", "npu_name": "NPU Strix", "pmode": pmode}

    def fake_measure(case, **common):
        calls.append(case.name)
        cyc = state["canary_cycles"] if case.name == registry.CANARY.name else 12_345
        m = Measurement(
            case=case.name,
            device="NPU Strix",
            correct=case.name not in state["wrong"],
            verdict="ok",
            cycles_median=cyc,
            work_ops=4096,
        )
        m.compile_s, m.xclbin_bytes, m.insts_bytes, m.elf_bytes = 1.0, 10, 20, 30
        return m

    monkeypatch.setattr(run_mod, "preflight", fake_preflight)
    monkeypatch.setattr(run_mod, "measure", fake_measure)
    monkeypatch.setattr(run_mod, "provenance", lambda pm=None: "commit abc | peano p")
    return types.SimpleNamespace(
        calls=calls, state=state, out=tmp_path / "b.json", meta=tmp_path / "m.json"
    )


def _run(env, *extra):
    return run_mod.main(
        ["--out", str(env.out), "--meta", str(env.meta), "--no-cycles", *extra]
    )


def test_preflight_failure_writes_nothing(env):
    env.state["preflight_fail"] = True
    assert _run(env) == 2
    assert not env.out.exists() and env.meta.exists()


def test_canary_out_of_band_writes_nothing(env):
    env.state["canary_cycles"] = registry.CANARY_CYCLE_BAND[1] * 10
    assert _run(env) == 2
    assert not env.out.exists()
    assert env.calls == [registry.CANARY.name]


def test_one_wrong_kernel_invalidates(env):
    env.state["wrong"].add(next(registry.perf_cases("^add/")).name)
    assert _run(env) == 3
    assert not env.out.exists()
    assert json.loads(env.meta.read_text())["summary"]["n_wrong"] == 1


def test_rows_have_expected_metrics(env):
    assert _run(env, "--only", "^passthrough") == 0
    rows = json.loads(env.out.read_text())
    names = {r["name"].rsplit("/", 1)[-1] for r in rows}
    assert {
        "cycles",
        "cycles_per_kop",
        "compile_s",
        "xclbin_bytes",
        "insts_bytes",
        "core_elf_bytes",
    } <= names
    assert all("peano" in r["extra"] for r in rows)
    assert all(r["value"] is not None for r in rows)


def test_only_filters_entries(env):
    assert _run(env, "--only", "^mv/") == 0
    assert all(c.startswith(("mv/", registry.CANARY.name)) for c in env.calls)


def test_arch_gated_cases_are_skipped_on_the_other_device(env, monkeypatch):
    # The fake preflight reports a Strix (aie2p); an aie2-only case is skipped.
    aie2_only = registry.Case("passthrough", calls=2, arch="aie2")
    monkeypatch.setattr(registry, "CASES", registry.CASES + [aie2_only])
    assert _run(env, "--only", "^passthrough") == 0
    assert aie2_only.name not in env.calls


_XRT_SMI_EXAMINE = """
System Configuration
  OS Name              : Linux
Devices present
BDF             :  Name
------------------------------------
[0000:c5:00.1]  :  NPU Strix
"""


def test_preflight_keeps_the_whole_device_name(monkeypatch):
    # `NPU Strix`, not `NPU`: the arch gate reads the family from the name.
    # The platform-report line below assumes the pmode field's spelling
    # (see the TODO in _util.preflight); the device-name parse is what
    # this test pins.
    def fake_run(cmd):
        if "--report" in cmd:
            assert cmd[cmd.index("-d") + 1] == "0000:c5:00.1"
            return "Performance Mode : performance\n"
        return _XRT_SMI_EXAMINE

    monkeypatch.setattr(_util, "_run", fake_run)
    pre = _util.preflight("performance")
    assert pre == {
        "bdf": "0000:c5:00.1",
        "npu_name": "NPU Strix",
        "pmode": "performance",
    }
    assert run_mod._arch(pre) == "aie2p"


@pytest.mark.parametrize(
    "name,arch",
    [
        ("NPU Phoenix", "aie2"),
        ("NPU Strix", "aie2p"),
        ("NPU Strix Halo", "aie2p"),
        ("NPU Krackan", "aie2p"),
        ("NPU Gorgon Point", "aie2p"),
        ("RyzenAI-npu1", "aie2"),
        ("RyzenAI-npu4", "aie2p"),
        ("NPU", "aie2"),
    ],
)
def test_npu_arch_follows_the_device_family(name, arch):
    assert _util.npu_arch(name) == arch


def test_case_names_are_unique_and_spell_out_what_the_dtype_segment_cannot():
    # Every series is keyed by the case name; two cases sharing one would
    # overwrite each other's history. The residual dtype of bn_conv2dk1_skip
    # is neither the first input nor the output, so it goes into the name.
    names = [c.name for c in registry.CASES]
    dups = {n for n in names if names.count(n) > 1}
    assert not dups, f"duplicate case names: {sorted(dups)}"
    skip = [c.name for c in registry.CASES if c.kwargs.get("skip_dtype") is not None]
    assert skip and all("skip_dtype=int8" in n for n in skip)


def test_data_policy_is_derived_from_the_contract():
    from aie.iron import kernels

    from . import registry

    # integer kernels: the extremes; exact-copy bf16 kernels: IEEE edge data;
    # LUT activations: finite data only; matmul operands: no NaN; bfp16ebs8
    # operands: no "max" either; structured inputs: random only.
    assert registry.data_policy(kernels.passthrough()) == registry.INT_DATA
    assert registry.data_policy(kernels.add()) == registry.FLOAT_BASE + (
        "subnormal",
        "nan_inf",
    )
    assert registry.data_policy(kernels.gelu()) == registry.FLOAT_BASE
    assert registry.data_policy(kernels.mm()) == registry.MATRIX_DATA
    assert registry.data_policy(kernels.expand()) == ("random",)
    with registry._device_for("aie2p"):
        assert "max" not in registry.data_policy(kernels.mm_bfp())
    # A Case without data_cases resolves to the derived policy.
    case = next(c for c in registry.CASES if c.factory == "add")
    assert case.data_cases is None and case.data_policy() == registry.data_policy(
        kernels.add()
    )
