#!/usr/bin/env python3
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Write the maintainer dashboard as one self-contained HTML file.

The page is this working tree's ``utils/kernel_checks/index.html``; its data
is the published ``ci-health/``, ``code-health/``, ``kernel-checks/`` and
``component-checks/``
of the publication branch (``origin/gh-pages`` by default), or of a directory
laid out the same way. The data goes into the file, gzipped, behind a
``fetch`` that answers the page's relative requests from it, so the file
opens anywhere with no server: in a browser, attached to a pull request for
a reviewer, in an editor's preview. Chart.js still comes from its CDN;
with no network the page draws its charts as SVG.

    python3 utils/kernel_checks/snapshot.py               # dashboard-snapshot.html
    python3 utils/kernel_checks/snapshot.py --fetch -o /tmp/d.html
    python3 utils/kernel_checks/snapshot.py --view performance
    python3 utils/kernel_checks/snapshot.py --view health --with /tmp/sample   # over the published

Unlike preview.py, it needs nothing but Python and git: no mlir_aie, no
synthetic nights, no server. A run's per-run records are many and large, so
only each check's latest few are kept; the page reads no older one.
"""

import argparse
import base64
import gzip
import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PAGE = HERE / "index.html"
DIRS = ("ci-health", "code-health", "kernel-checks", "component-checks")
# The page reads a check's latest record and the published one before it.
RUNS_KEPT = 3

SHIM = """<script>
// snapshot.py: the published files, answered from this file.
(function () {
  const VIEW = %(view)s;
  if (VIEW && !location.hash) history.replaceState(null, '', '#view=' + VIEW);
  const data = (async () => {
    const bytes = Uint8Array.from(atob(%(blob)s), c => c.charCodeAt(0));
    const text = new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'));
    return JSON.parse(await new Response(text).text());
  })();
  const network = window.fetch.bind(window);
  window.fetch = async (url, options) => {
    const u = String(url);
    if (/^(https?:|data:|blob:)/.test(u)) return network(url, options);
    const files = await data;
    const key = u.split('?')[0].replace(/^(\\.\\.\\/)+/, '').replace(/^\\.\\//, '');
    return key in files
      ? new Response(JSON.stringify(files[key]), { status: 200, headers: { 'Content-Type': 'application/json' } })
      : new Response('not in this snapshot', { status: 404 });
  };
})();
</script>
"""


def kept(paths: list[str]) -> list[str]:
    """The JSON files the page reads: all but the older per-run records."""
    runs: dict[str, list[str]] = {}
    out = []
    for p in paths:
        if not p.endswith(".json"):
            continue
        m = re.match(r"(.*)/runs/([^/]+)\.json$", p)
        if m:
            runs.setdefault(m.group(1), []).append(p)
        else:
            out.append(p)
    # Run ids are GitHub's, which grow: the newest last.
    for group in runs.values():
        group.sort(key=lambda p: [int(n) for n in re.findall(r"\d+", Path(p).stem)])
        out += group[-RUNS_KEPT:]
    return sorted(out)


def from_ref(ref: str) -> dict:
    """The published files at a git ref: {path: parsed JSON}."""
    git = ["git", "-C", str(ROOT)]
    listed = subprocess.run(
        [*git, "ls-tree", "-r", "--name-only", ref, "--", *DIRS],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.split()
    return {
        p: json.loads(
            subprocess.run(
                [*git, "show", f"{ref}:{p}"], check=True, capture_output=True
            ).stdout
        )
        for p in kept(listed)
    }


def from_dir(root: Path) -> dict:
    """The published files under a directory laid out as the branch is."""
    listed = [
        p.relative_to(root).as_posix()
        for d in DIRS
        if (root / d).is_dir()
        for p in (root / d).rglob("*.json")
    ]
    return {
        p: json.loads((root / p).read_text(encoding="utf-8")) for p in kept(listed)
    }


def snapshot(page: str, files: dict, view: str = "") -> str:
    """The page with its data in it."""
    blob = base64.b64encode(
        gzip.compress(json.dumps(files, separators=(",", ":")).encode(), 9)
    ).decode()
    shim = SHIM % {"view": json.dumps(view), "blob": json.dumps(blob)}
    if "<head>" not in page:
        raise ValueError("the page has no <head> to put its data in")
    return page.replace("<head>", "<head>\n" + shim, 1)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--ref", default="origin/gh-pages", help="publication ref")
    source.add_argument(
        "--dir", type=Path, help="a directory laid out as the branch is"
    )
    parser.add_argument(
        "--with",
        dest="overlay",
        type=Path,
        help="a directory laid out as the branch is whose files replace or add to the published ones",
    )
    parser.add_argument(
        "--fetch", action="store_true", help="git fetch origin gh-pages first"
    )
    parser.add_argument(
        "--view",
        default="",
        help="the tab to open on: overview, ci, night, performance, placement, health",
    )
    parser.add_argument(
        "-o", "--out", type=Path, default=Path("dashboard-snapshot.html")
    )
    args = parser.parse_args(argv)
    if args.fetch and not args.dir:
        subprocess.run(
            ["git", "-C", str(ROOT), "fetch", "origin", "gh-pages"], check=True
        )
    files = from_dir(args.dir) if args.dir else from_ref(args.ref)
    if args.overlay:
        files.update(from_dir(args.overlay))
    html = snapshot(PAGE.read_text(encoding="utf-8"), files, args.view)
    args.out.write_text(html, encoding="utf-8")
    size = len(html) / 1e6
    print(f"{args.out}: {len(files)} published files, {size:.1f} MB", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
