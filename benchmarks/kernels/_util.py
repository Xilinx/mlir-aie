# benchmarks/kernels/_util.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Small shared helpers: provenance string, benchmark-action rows, NPU preflight."""

from __future__ import annotations

import json
import os
import re
import subprocess
from importlib.metadata import version as pkg_version
from pathlib import Path


class PreflightError(RuntimeError):
    pass


# Mirrors NPU2_REGEX in utils/iron_setup.py (a setup script, not part of the
# wheel): the NPU2 family, whose kernels build for aie2p.
_NPU2_NAMES = re.compile(
    r"NPU Strix|NPU Strix Halo|NPU Krackan|NPU Gorgon Point|RyzenAI-npu[456]",
    re.IGNORECASE,
)


def npu_arch(npu_name: str) -> str:
    """``aie2p`` for the NPU2 family (Strix, Krackan, Gorgon Point), else ``aie2``."""
    return "aie2p" if _NPU2_NAMES.search(npu_name) else "aie2"


def _run(cmd: list[str]) -> str:
    return subprocess.run(cmd, check=True, capture_output=True, text=True).stdout


def _ver(pkg: str) -> str:
    try:
        return pkg_version(pkg)
    except Exception:  # noqa: BLE001
        return "unknown"


def provenance(pmode: str | None = None) -> str:
    """Goes into every row's `extra` (shown on graph hover)."""
    try:
        commit = (
            os.environ.get("GITHUB_SHA") or _run(["git", "rev-parse", "HEAD"]).strip()
        )
    except Exception:  # noqa: BLE001
        commit = "unknown"
    try:
        xrt = _run(["xrt-smi", "--version"]).strip().splitlines()[0]
    except Exception:  # noqa: BLE001
        xrt = "n/a"
    return (
        f"commit {commit[:10]} | peano {_ver('llvm-aie')} | mlir_aie {_ver('mlir_aie')} "
        f"| xrt {xrt}" + (f" | pmode {pmode}" if pmode else "")
    )


def row(name: str, unit: str, value, extra: str, rng: str | None = None) -> dict:
    r = {"name": name, "unit": unit, "value": value, "extra": extra}
    if rng:
        r["range"] = rng
    return r


def write_rows(path: str | Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("refusing to write an empty benchmark file")
    Path(path).write_text(json.dumps(rows, indent=1))


def preflight(required_pmode: str) -> dict:
    """Find the NPU and check its power mode. Raises PreflightError."""
    out = _run(["xrt-smi", "examine"])
    # `[0000:c5:00.1]  :  NPU Strix` -- the name runs to the end of the line.
    m = re.search(r"\[([0-9a-fA-F:.]+)\]\s*:\s*(.+?)\s*$", out, re.MULTILINE)
    if not m:
        raise PreflightError("no NPU device in `xrt-smi examine` output")
    bdf, name = m.group(1), m.group(2)
    # TODO(verify): exact field name in `--report platform` on Linux.
    rep = _run(["xrt-smi", "examine", "-d", bdf, "--report", "platform"])
    pm = re.search(r"(?i)(?:performance|power)\s*mode\s*:\s*(\S+)", rep)
    pmode = pm.group(1).lower() if pm else "unknown"
    if pmode != required_pmode:
        raise PreflightError(
            f"pmode is '{pmode}', required '{required_pmode}' "
            f"(sudo xrt-smi configure -d {bdf} --pmode {required_pmode})"
        )
    return {"bdf": bdf, "npu_name": name, "pmode": pmode}
