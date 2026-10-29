# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""utils/kernel_checks/snapshot.py: the dashboard as one self-contained file."""

import importlib.util
import json
import os
import shutil
import subprocess
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
    files = snap.from_dir(tmp_path)
    # Run ids sort as numbers: 9 is older than 10. HTML is not data.
    assert sorted(p for p in files if "/runs/" in p) == [
        "kernel-checks/npu1/runs/10.json",
        "kernel-checks/npu1/runs/38.json",
        "kernel-checks/npu1/runs/40.json",
    ]
    assert "ci-health/status.json" in files
    assert not any(p.endswith(".html") for p in files)


def test_the_page_reads_its_data_from_the_file(snap, tmp_path):
    node = shutil.which("node")
    if not node:
        if os.environ.get("MLIR_AIE_REQUIRE_NODE"):
            pytest.fail("MLIR_AIE_REQUIRE_NODE is set but node is not on PATH")
        pytest.skip("node is required to run the snapshot's fetch")
    published(tmp_path)
    html = snap.snapshot(
        "<html><head><title>t</title></head></html>", snap.from_dir(tmp_path), "ci"
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
