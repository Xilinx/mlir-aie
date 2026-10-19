# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# RUN: %pytest %s
"""aiecc's --profile-json, grouped into compile stages, on real captured profiles."""

import importlib
import json
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CHECKS = ROOT / "utils/component_checks"
# A shim-only design (test/aiecc/profile_json.mlir's shape) and a cold
# mobilenet JIT compile at SA effort 0.05, both captured from aiecc as is.
SHIM = CHECKS / "fixtures/aiecc_profile_shim.json"
MOBILENET = CHECKS / "fixtures/aiecc_profile_mobilenet.json"


@pytest.fixture
def checks(monkeypatch):
    monkeypatch.syspath_prepend(str(CHECKS))
    return importlib.import_module("aiecc_stages"), importlib.import_module(
        "llama_chain_hw"
    )


@pytest.mark.parametrize("fixture", [SHIM, MOBILENET], ids=["shim", "mobilenet"])
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
        ("a_new_aiecc_edge", "other"),
    ],
)
def test_output_names_aiecc_was_given_are_matched_by_kind(checks, edge, stage):
    stages, _ = checks
    assert stages.stage_of(edge) == stage


def test_an_unknown_edge_is_counted_as_other(checks, tmp_path):
    stages, _ = checks
    profile = json.loads(SHIM.read_text())
    profile["edges"].append(
        {"name": "a_new_aiecc_edge", "ms": 250, "start_ms": 0, "end_ms": 250}
    )
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(profile))
    assert stages.stage_seconds(str(path))["other"] == 0.25


def test_no_profile_means_no_stages(checks, tmp_path):
    stages, _ = checks
    missing = str(tmp_path / "missing.json")
    assert stages.stage_seconds(missing) is None
    assert stages.stage_rows(missing, 60.5, "") == {}


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
        {"tokens_passed": 4, "tokens_total": 4}, llama.parse_timing(""), {}, compiled
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
    (tmp_path / "newer" / "input_with_addresses.mlir").write_text(
        "%c = aie.core(%t0) {\n}\n%d = aie.core(%t1) {\n}\n"
    )
    assert yolo.artifact_stats(newest) == {"elf_bytes": 4096, "n_cores": 2}
    assert yolo.artifact_stats(str(tmp_path / "older")) == {}
    assert yolo.artifact_stats(None) == {}
