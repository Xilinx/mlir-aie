# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# RUN: %pytest %s
"""utils/kernel_checks/snapshot.py: the dashboard as one self-contained file."""

import base64
import gzip
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def snap():
    spec = importlib.util.spec_from_file_location(
        "snapshot", ROOT / "utils/kernel_checks/snapshot.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def published(root: Path) -> None:
    """A publication branch in miniature: CI health, one NPU, one check."""
    files = {
        "ci-health/status.json": {"schema": 1, "runners": []},
        "kernel-checks/npu1/latest.json": {"id": "40"},
        "component-checks/router/runs.json": {"runs": []},
        **{f"kernel-checks/npu1/runs/{i}.json": {"id": i} for i in (9, 10, 38, 40)},
    }
    for path, data in files.items():
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        (root / path).write_text(json.dumps(data))
    (root / "kernel-checks/index.html").write_text("<html></html>")


def test_only_the_latest_run_records_are_kept(snap, tmp_path):
    published(tmp_path)
    files = snap.collect(snap.Directory(tmp_path))
    # Run ids sort as numbers: 9 is older than 10. HTML is not data.
    assert sorted(p for p in files if "/runs/" in p) == [
        "kernel-checks/npu1/runs/10.json",
        "kernel-checks/npu1/runs/38.json",
        "kernel-checks/npu1/runs/40.json",
    ]
    assert "ci-health/status.json" in files
    assert not any(p.endswith(".html") for p in files)


@pytest.mark.parametrize("source", ["dir", "ref"])
@pytest.mark.parametrize("directory", ["kernel-checks/npu1", "component-checks/router"])
@pytest.mark.parametrize(
    "changes, selected, previous",
    [
        (
            [
                {"event": "schedule"},
                {"event": "schedule"},
                {"event": "workflow_dispatch"},
            ],
            "2",
            "1",
        ),
        ([{}, {"pmode": "turbo"}, {}], "3", "1"),
        ([{}, {"device": "Phoenix"}, {}], "3", "1"),
        ([{"device": "Phoenix"}, {"device": "Strix Halo"}, {}], "3", "2"),
        (
            [{}, {"event": "workflow_dispatch"}, {"published": False}, {}],
            "4",
            "1",
        ),
        ([{"event": "workflow_dispatch"} for _ in range(4)], "4", None),
        ([{}, {}], "2", "1"),
    ],
    ids=[
        "manual-after",
        "power",
        "same-device",
        "new-device",
        "refused",
        "manual-only",
        "legacy",
    ],
)
def test_required_comparison_records_survive_a_long_refusal_tail(
    snap, tmp_path, source, directory, changes, selected, previous
):
    runs = [
        {
            "id": str(i),
            "published": True,
            "pmode": "performance",
            "device": "Strix",
            **change,
        }
        for i, change in enumerate(changes, 1)
    ]
    latest = next(r for r in runs if r["id"] == selected)
    runs += [
        {"id": str(i), "published": False, "pmode": "performance", "device": "Strix"}
        for i in range(10, 30)
    ]
    data = {
        f"{directory}/latest.json": latest,
        f"{directory}/runs.json": {"runs": runs},
        **{
            f"{directory}/runs/{r['id']}.json": {**r, "detail": [r["id"]]} for r in runs
        },
    }
    expected = {"27", "28", "29", selected}
    if previous:
        expected.add(previous)
    for path, value in data.items():
        dest = tmp_path / path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(value), encoding="utf-8")
    if source == "dir":
        files = snap.collect(snap.Directory(tmp_path))
    else:
        # A publication branch with a page beside the data and a file outside it.
        (tmp_path / directory / "index.html").write_text("<html></html>")
        (tmp_path / "README.md").write_text("not published data")
        for args in (
            ["init", "-q", "-b", "publication"],
            ["add", "-A"],
            ["commit", "-q", "-m", "publication"],
        ):
            subprocess.run(
                ["git", "-c", "user.name=t", "-c", "user.email=t@example.com", *args],
                cwd=tmp_path,
                check=True,
                capture_output=True,
            )
        files = snap.collect(snap.GitRef("publication", tmp_path))
        assert set(files) <= set(data)
    kept = {Path(p).stem for p in files if "/runs/" in p}
    assert kept == expected
    assert len(kept) <= snap.RUNS_KEPT + 2
    for run_id in expected:
        assert files[f"{directory}/runs/{run_id}.json"]["detail"] == [run_id]


def test_an_overlays_latest_keeps_the_records_it_points_at(snap, tmp_path):
    base, overlay, out = tmp_path / "base", tmp_path / "overlay", tmp_path / "d.html"
    runs = [
        {"id": str(i), "published": True, "pmode": "performance", "device": "Strix"}
        for i in range(1, 11)
    ]
    data = {
        (base, "kernel-checks/npu2/latest.json"): runs[-1],
        (base, "kernel-checks/npu2/runs.json"): {"runs": runs},
        **{(base, f"kernel-checks/npu2/runs/{r['id']}.json"): r for r in runs},
        (overlay, "kernel-checks/npu2/latest.json"): runs[4],
        (overlay, "kernel-checks/npu2/runs/5.json"): {**runs[4], "from": "overlay"},
    }
    for (root, path), value in data.items():
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        (root / path).write_text(json.dumps(value), encoding="utf-8")
    assert snap.main(["--dir", str(base), "--with", str(overlay), "-o", str(out)]) == 0
    blob = re.search(r"atob\((\"[^\"]+\")\)", out.read_text(encoding="utf-8"))
    files = json.loads(gzip.decompress(base64.b64decode(json.loads(blob.group(1)))))
    assert files["kernel-checks/npu2/latest.json"]["id"] == "5"
    # The overlay's latest and its comparison, as well as the newest three.
    assert {Path(p).stem for p in files if "/runs/" in p} == {"4", "5", "8", "9", "10"}
    assert files["kernel-checks/npu2/runs/5.json"]["from"] == "overlay"


def test_a_latest_missing_from_the_index_searches_all_earlier_runs(snap):
    paths = [f"kernel-checks/npu1/runs/{i}.json" for i in range(1, 10)]
    metadata = {
        "kernel-checks/npu1/latest.json": {"id": "missing", "pmode": None},
        "kernel-checks/npu1/runs.json": {"runs": [{"id": "1", "published": True}]},
    }
    assert snap.kept(paths, metadata) == [paths[i] for i in (0, 6, 7, 8)]


def test_the_page_reads_its_data_from_the_file(snap, tmp_path):
    node = shutil.which("node")
    if not node:
        if os.environ.get("MLIR_AIE_REQUIRE_NODE"):
            pytest.fail("MLIR_AIE_REQUIRE_NODE is set but node is not on PATH")
        pytest.skip("node is required to run the snapshot's fetch")
    published(tmp_path)
    html = snap.snapshot(
        "<html><head><title>t</title></head></html>",
        snap.collect(snap.Directory(tmp_path)),
        "ci",
    )
    assert html.index("snapshot.py") < html.index("<title>")
    script = html.split("<script>", 1)[1].split("</script>", 1)[0]
    # The page's own requests, relative to dashboard/, answered from the
    # file; anything else is not in it.
    check = """
const assert = require('node:assert/strict');
global.location = { hash: '' };
global.history = { replaceState: (_s, _t, url) => { location.hash = url; } };
global.window = { fetch: async () => { throw new Error('no network'); } };
%s
(async () => {
  assert.equal(location.hash, '#view=ci');
  const ok = await window.fetch('../ci-health/status.json', { cache: 'no-cache' });
  assert.equal(ok.status, 200);
  assert.deepEqual(await ok.json(), { schema: 1, runners: [] });
  assert.equal((await window.fetch('../kernel-checks/npu1/runs/40.json')).status, 200);
  assert.equal((await window.fetch('../kernel-checks/npu1/runs/9.json')).status, 404);
})().catch(e => { console.error(e); process.exit(1); });
""" % script.replace("window.fetch.bind(window)", "window.fetch")
    subprocess.run([node, "-"], input=check, encoding="utf-8", check=True)


def test_a_page_without_a_head_is_refused(snap):
    with pytest.raises(ValueError):
        snap.snapshot("<html></html>", {})


def test_help_without_a_module_docstring():
    # -OO strips docstrings.
    result = subprocess.run(
        [sys.executable, "-OO", ROOT / "utils/kernel_checks/snapshot.py", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "--ref" in result.stdout
