# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# RUN: %pytest %s
"""aiecc's --profile-json, grouped into compile stages, on real captured profiles."""

import importlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CHECKS = ROOT / "utils/component_checks"
# A shim-only design (test/aiecc/profile_json.mlir's shape), a cold
# mobilenet JIT compile at SA effort 0.05, and test/aiecc/
# cpp_full_elf_generation.mlir built with --get-full-elf, all captured from
# aiecc as is.
SHIM = CHECKS / "fixtures/aiecc_profile_shim.json"
MOBILENET = CHECKS / "fixtures/aiecc_profile_mobilenet.json"
FULL_ELF = CHECKS / "fixtures/aiecc_profile_full_elf.json"
AIECC = shutil.which("aiecc")


@pytest.fixture
def checks(monkeypatch):
    monkeypatch.syspath_prepend(str(CHECKS))
    return importlib.import_module("aiecc_stages"), importlib.import_module(
        "llama_chain_hw"
    )


@pytest.mark.parametrize(
    "fixture", [SHIM, MOBILENET, FULL_ELF], ids=["shim", "mobilenet", "full-elf"]
)
def test_every_captured_edge_has_a_stage(checks, fixture):
    stages, _ = checks
    edges = json.loads(fixture.read_text())["edges"]
    assert {
        e["name"]: stages.stage_of(e["name"])
        for e in edges
        if stages.stage_of(e["name"]) == "other"
    } == {}


def test_mobilenet_stages_are_its_edges_summed(checks):
    stages, _ = checks
    profile = json.loads(MOBILENET.read_text())
    ms = {e["name"]: e["ms"] for e in profile["edges"]}
    got = stages.stage_seconds(str(MOBILENET))
    assert got["place"] == ms["placed.mlir"] / 1000
    assert got["route"] == ms["input_physical.mlir"] / 1000
    assert got["kernels"] == ms["awaitLinkFiles"] / 1000
    assert got["wall_s"] == profile["wall_ms"] / 1000
    assert sum(got[s] for s in stages.STAGES) == pytest.approx(sum(ms.values()) / 1000)
    # Cold, its own kernel objects dominate; the SA placer is next.
    assert max(stages.STAGES, key=got.__getitem__) == "kernels"
    assert sorted(stages.STAGES, key=got.__getitem__)[-2] == "place"


@pytest.mark.parametrize(
    "edge, stage",
    [
        ("placed.mlir", "place"),
        ("input_physical.mlir", "route"),
        ("input_with_symbols.mlir", "lower"),
        ("design.xclbin", "package"),
        ("main.pdi", "package"),
        ("insts_chain_persist_grow_N16_PT4.bin", "package"),
        ("elfs_{0}.elf", "link"),
        ("instructions.elf", "package"),
        ("kernels_{0}.json", "package"),
        # Edges a full-ELF, multi-device, transaction or locmap build adds.
        ("perDeviceNPULowered_{0}.mlir", "lower"),
        ("perDeviceCompile_{0}.mlir", "lower"),
        ("perDeviceCompileMatching", "lower"),
        ("perDeviceArches_{0}.txt", "lower"),
        ("ctrlPktSeqs", "lower"),
        ("{0}_transaction.mlir", "package"),
        ("input_aie_partition_{0}.json", "package"),
        ("insts_{0}.bin.locmap.json", "package"),
        ("linked.mlir", "link"),
        ("a_new_aiecc_edge", "other"),
    ],
)
def test_output_names_aiecc_was_given_are_matched_by_kind(checks, edge, stage):
    stages, _ = checks
    assert stages.stage_of(edge) == stage


@pytest.mark.skipif(AIECC is None, reason="no aiecc on PATH")
def test_the_aiecc_built_here_names_only_known_edges(checks, tmp_path):
    stages, _ = checks
    profile = tmp_path / "profile.json"
    subprocess.run(
        [AIECC, "--get-npu-insts", "--no-progress", f"--profile-json={profile}"]
        + [f"--output-dir={tmp_path}", f"--tmpdir={tmp_path / 'work'}"]
        + [str(ROOT / "test/aiecc/profile_json.mlir")],
        check=True,
        capture_output=True,
    )
    got = json.loads(profile.read_text())
    assert got["version"] == stages.PROFILE_VERSION
    assert [
        e["name"] for e in got["edges"] if stages.stage_of(e["name"]) == "other"
    ] == []


def test_an_unknown_edge_is_counted_as_other(checks, tmp_path):
    stages, _ = checks
    profile = json.loads(SHIM.read_text())
    profile["edges"].append(
        {"name": "a_new_aiecc_edge", "ms": 250, "start_ms": 0, "end_ms": 250}
    )
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(profile))
    assert stages.stage_seconds(str(path))["other"] == 0.25


def test_a_profile_that_is_not_the_compiles_gives_no_stages(checks, tmp_path):
    stages, _ = checks
    missing = str(tmp_path / "missing.json")
    with pytest.raises(stages.ProfileError, match="no profile"):
        stages.stage_seconds(missing)
    assert stages.stage_rows(missing, 60.5, "_warm") == {
        "profile_problem_warm": f"no profile at {missing}"
    }
    newer = tmp_path / "newer.json"
    newer.write_text(json.dumps(dict(json.loads(SHIM.read_text()), version=2)))
    assert stages.stage_rows(str(newer), 60.5, "") == {
        "profile_problem": f"{newer} is version 2, this reads 1"
    }
    # aiecc runs inside the timed compile: a profile longer than the whole
    # compile is another compile's.
    (problem,) = stages.stage_rows(str(MOBILENET), 12.0, "").values()
    assert problem.endswith("spans 17.42 s of a 12.00 s compile: not this compile's")
    # A failed compile has no stages, and nothing to explain.
    assert stages.stage_rows(missing, None, "") == {}


def test_a_command_is_echoed_as_it_runs_and_measured(checks, capfd):
    stages, _ = checks
    measured = stages.run_measured(
        [sys.executable, "-c", "import sys; print('one'); sys.exit(3)"],
        None,
        None,
        60,
    )
    assert measured.process.returncode == 3 and measured.process.stdout == "one\n"
    assert capfd.readouterr().out == "one\n"
    assert measured.seconds > 0 and measured.peak_rss_mib > 0 and measured.cpu_s > 0


def test_a_command_past_its_timeout_is_killed_with_its_children(checks, capfd):
    stages, _ = checks
    start = time.perf_counter()
    measured = stages.run_measured(
        ["sh", "-c", "echo started; sleep 60 & sleep 60"], None, None, 1
    )
    assert time.perf_counter() - start < 30
    assert measured.process.returncode == stages.TIMED_OUT
    lines = measured.process.stdout.splitlines()
    assert lines[0] == "started" and lines[-1].endswith("timed out after 1 s")
    assert capfd.readouterr().out.splitlines() == lines


def test_a_warm_compile_keeps_only_the_kernel_objects_and_headers(checks, tmp_path):
    stages, _ = checks
    cache = tmp_path / "cache"
    for name in ("objects", "pch", "builds", "devices", "final.prj", "0123abcd"):
        (cache / name).mkdir(parents=True)
        (cache / name / "kept").write_text("")
    (cache / "final.xclbin").write_text("")
    elsewhere = tmp_path / "design.elf"
    elsewhere.write_text("")
    stages.clear_for_warm(str(cache), [str(elsewhere), str(tmp_path / "absent")])
    assert sorted(p.name for p in cache.iterdir()) == ["objects", "pch"]
    assert (cache / "objects/kept").exists() and (cache / "pch/kept").exists()
    assert not elsewhere.exists()


def test_llama_publishes_each_stage_cold_and_warm(checks):
    stages, llama = checks
    compiled = {
        "compile_s": 19.8,
        "compile_s_warm": 0.5,
        **stages.stage_rows(str(MOBILENET), 19.8, ""),
        **stages.stage_rows(str(SHIM), 0.5, "_warm"),
    }
    assert stages.stage_rows(str(MOBILENET), None, "") == {}
    rows = llama.aggregate(
        llama.parse_correctness("", 1),
        llama.parse_timing("", 1),
        llama.parse_generate("", 1),
        {},
        compiled,
    )
    by_name = {r["name"].rsplit("/", 1)[1]: r for r in rows}
    assert by_name["compile_place_s"]["value"] == 0.96
    assert by_name["compile_kernels_s"]["value"] == 15.72
    assert by_name["compile_aiecc_wall_s"]["value"] == 17.42
    assert by_name["compile_outside_aiecc_s"]["value"] == 2.38
    assert by_name["compile_place_s_warm"]["unit"] == "s"
    assert {f"{m}_warm" for m in stages.STAGE_METRICS} <= by_name.keys()


def test_yolo26n_publishes_each_stage_cold_and_warm(checks):
    stages, _ = checks
    yolo = importlib.import_module("yolo26n_hw")
    compiled = {
        "compile_s": 19.8,
        "compile_s_warm": 0.5,
        **stages.stage_rows(str(MOBILENET), 19.8, ""),
        **stages.stage_rows(str(SHIM), 0.5, "_warm"),
    }
    rows = yolo.aggregate(
        yolo.parse_correctness("", 1), yolo.parse_timing(""), {}, compiled
    )
    by_name = {r["name"].rsplit("/", 1)[1]: r for r in rows}
    assert all(r["name"].startswith("yolo26n/640/") for r in rows)
    assert by_name["compile_place_s"]["value"] == 0.96
    assert by_name["compile_kernels_s"]["value"] == 15.72
    assert by_name["compile_outside_aiecc_s"]["value"] == 2.38
    assert {
        f"{m}{suffix}" for m in stages.STAGE_METRICS for suffix in ("", "_warm")
    } <= by_name.keys()
    # A run that never verified still says so, with nothing to time.
    assert by_name["images_passed"] == {
        "name": "yolo26n/640/images_passed",
        "unit": "images",
        "value": 0,
        "range": "of 0",
    }
    assert not any(n.startswith(("us_per_image", "fps")) for n in by_name)


def test_yolo26n_reads_the_newest_build_in_its_cache(checks, tmp_path):
    stages, _ = checks
    yolo = importlib.import_module("yolo26n_hw")
    assert yolo.newest_build(str(tmp_path)) is None
    for name, mtime in (("older", 1_000_000), ("newer", 2_000_000)):
        build = tmp_path / name
        build.mkdir()
        (build / stages.PROFILE_NAME).write_text(SHIM.read_text())
        os.utime(build / stages.PROFILE_NAME, (mtime, mtime))
    (tmp_path / "objects").mkdir()
    newest = yolo.newest_build(str(tmp_path))
    assert newest == str(tmp_path / "newer")
    (tmp_path / "newer" / "design.elf").write_bytes(b"\0" * 4096)
    assert yolo.artifact_stats(newest) == {"elf_bytes": 4096}
    assert yolo.artifact_stats(str(tmp_path / "older")) == {}
    assert yolo.artifact_stats(None) == {}
